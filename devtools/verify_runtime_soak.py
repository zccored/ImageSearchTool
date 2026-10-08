# -*- coding: utf-8 -*-
# ImageSearchTool · 连续建库线程/PNG 缓冲与索引一致性回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""同进程重复生产建库；不改线程策略，不清 CUDA/系统缓存，不写原索引。

--out 指归档父目录；每次创建含参数清单、完整 SHA256 的独立批次。
RSS 是诊断量，不把分配器保留的内存直接定性为泄漏或设武断阈值。
"""
import argparse
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import sys
import shutil
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, compare_tile_arrays, run_one
from archive_batch import create_archive, digest
from profile_forward_phases import foreign_python


def snapshot():
    from hybrid_search import png_fast
    import cv2
    import torch
    process = psutil.Process()
    memory = process.memory_info()
    return {"epoch": time.time(), "rss_mb": round(memory.rss / 2**20, 2),
            "private_mb": round(memory.private / 2**20, 2) if hasattr(memory, "private") else None,
            "handles": process.num_handles() if hasattr(process, "num_handles") else None,
            "system_available_mb": round(psutil.virtual_memory().available / 2**20, 2),
            "native_threads": process.num_threads(),
            "python_threads": [{"name": t.name, "ident": t.ident}
                               for t in threading.enumerate()],
            "scratch_mb": png_fast.scratch_mb(), "opencv_threads": cv2.getNumThreads(),
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1080)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.n <= 0 or args.rounds < 2:
        parser.error("n 必须为正，rounds 至少 2")
    other = foreign_python()
    if other:
        raise RuntimeError(f"其他 Python 仍在运行 {other}，本轮未启动；不终止其他任务")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("此回归要求 CUDA")
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    # 初始化与首次模型载入放在资源基线前，避免把一次性库线程当作 worker 泄漏。
    warm_engine = HybridEngine(Config(prep_cache=False))
    warm_engine._get_extractor()
    del warm_engine
    gc.collect()
    sources = ["devtools/verify_runtime_soak.py", "devtools/ab_build_bench.py",
               "hybrid_search/config.py", "hybrid_search/runtime.py", "hybrid_search/fine.py",
               "hybrid_search/tile_index.py", "hybrid_search/coarse.py", "hybrid_search/store.py",
               "hybrid_search/png_fast.py"]
    work = create_archive(args.out, parameters={"script": "verify_runtime_soak.py",
                          "n": args.n, "rounds": args.rounds, "seed": 20260915,
                          "mode": "tiles", "png_decoder": "libdeflate", "prep_cache": False,
                          "opencv_threads": Config().opencv_threads,
                          "source_sha256": {p: digest(ROOT / p) for p in sources}})
    print("REPORT", work, flush=True)
    sample = build_sample(args.n, 0, 20260915)
    if len(sample) != args.n:
        raise AssertionError(("样本不足", len(sample), args.n))
    (work / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    before = snapshot()
    records = []
    started = time.monotonic()
    try:
        for i in range(args.rounds):
            other = foreign_python()
            if other:
                raise RuntimeError(f"发现其他 Python {other}，不启动下一轮；已完成报告保留，不自动重试")
            assert shutil.disk_usage(work).free > 1024**3, "归档盘空闲低于 1 GiB，停止追加"
            label = f"soak_{i}"
            if (work / f"tiles_{label}").exists():
                raise FileExistsError(label)
            round_start = time.monotonic()
            with (work / f"{label}.log").open("x", encoding="utf-8") as log, redirect_stdout(log):
                result = run_one("tiles", label, sample, str(work), False, True, 0,
                                 png_decoder="libdeflate")
            # 不人为清 CUDA allocator；只回收失去引用的 Python 对象。
            gc.collect()
            base_ids = {t["ident"] for t in before["python_threads"]}
            deadline = time.monotonic() + 5
            while any(t.ident not in base_ids for t in threading.enumerate()):
                if time.monotonic() >= deadline:
                    break
                time.sleep(.05)
            after = snapshot()
            record = {"result": result, "after_cleanup": after,
                      "round_total_s": time.monotonic()-round_start,
                      "elapsed_s": time.monotonic()-started}
            records.append(record)
            (work / "resources.json").write_text(
                json.dumps({"pid": psutil.Process().pid, "before": before, "rounds": records}, indent=2), encoding="utf-8")
            leaked = [t for t in after["python_threads"] if t["ident"] not in base_ids]
            assert not leaked, ("新增线程未退出", leaked)
            assert after["scratch_mb"] == before["scratch_mb"], "PNG scratch 未归还"
            assert after["opencv_threads"] == before["opencv_threads"] == Config().opencv_threads
            assert after["cuda_allocated_mb"] <= before["cuda_allocated_mb"] + 1, "存活 CUDA 张量增长"
            assert result["items"] > 0
            if i:
                assert result["items"] == records[0]["result"]["items"]
            print("ROUND", i, result["wall_s"], json.dumps(after), flush=True)
        # 比较在所有计时/资源采样后执行，避免加载索引污染下一轮 RSS 基线。
        with (work / "comparisons.log").open("x", encoding="utf-8") as log, redirect_stdout(log):
            for i in range(1, args.rounds):
                assert compare_tile_arrays(str(work), "soak_0", f"soak_{i}")
        # 额外确认所有索引的非有限值与粗细路径对齐，不仅检查不同运行的相似度。
        import numpy as np
        for i in range(args.rounds):
            directory = work / f"tiles_soak_{i}"
            with np.load(directory / "idx_tiles.coarse.npz", allow_pickle=True) as coarse, np.load(
                    directory / "idx_tiles.fine.npz", allow_pickle=True) as fine:
                assert np.array_equal(coarse["paths"], fine["paths"])
                assert np.isfinite(coarse["hu"]).all() and np.isfinite(fine["features"]).all()
        summary = {"passed": True, "images": len(sample), "rounds": args.rounds,
                   "items": records[0]["result"]["items"], "resources": "resources.json",
                   "total_build_s": sum(r["result"]["wall_s"] for r in records),
                   "total_elapsed_s": time.monotonic()-started, "finite_aligned_indices": True}
    except BaseException as exc:
        (work / "failure.json").write_text(json.dumps({"type": type(exc).__name__,
                "error": str(exc), "completed_rounds": len(records)}, indent=2), encoding="utf-8")
        raise
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", summary, "REPORT", work, flush=True)


if __name__ == "__main__":
    main()
