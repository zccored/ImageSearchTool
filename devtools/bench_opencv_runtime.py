# -*- coding: utf-8 -*-
# ImageSearchTool · 独立进程 OpenCV 线程策略 ABBA
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""独立进程 ABBA；默认比较 OpenCV 接管，或用 --decode-workers 比较外层线程。

指定 --decode-workers 后，两侧都保留现行 OpenCV 策略，只有候选覆盖外层线程数。
"""
import argparse
from contextlib import redirect_stdout
import json
import statistics
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from paths import GALLERY_ROOT
from archive_batch import create_archive, digest
from profile_forward_phases import foreign_python


def child(args):
    import cv2
    import torch
    from hybrid_search.engine import HybridEngine
    from hybrid_search.fine import ResNetExtractor
    from hybrid_search.config import Config
    from hybrid_search import png_fast
    original_engine = HybridEngine.__init__
    original_extractor = ResNetExtractor.__init__
    effective = {}

    def engine_init(engine, cfg):
        if args.decode_workers:
            if args.label.startswith("after"):
                cfg.decode_workers = args.decode_workers
        elif args.label.startswith("before"):
            cfg.opencv_threads = 0
        original_engine(engine, cfg)
        effective.update(requested=cfg.opencv_threads, actual=engine.opencv_threads,
                         cv2_version=cv2.__version__)

    def extractor_init(ex, cfg):
        original_extractor(ex, cfg)
        if ex.device != "cuda":
            raise RuntimeError("本性能基准需要 CUDA")
        data = torch.zeros((64 if args.mode == "tiles" else ex.batch, 3, 224, 224), device=ex.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
            for _ in range(3):
                ex.model(data)
        torch.cuda.synchronize()
        effective.update(device=ex.device, workers=ex.decode_workers, warmed=True)

    work = Path(args.work)
    sample = json.loads((work / "sample.json").read_text(encoding="utf-8"))
    with patch.object(HybridEngine, "__init__", engine_init), patch.object(
            ResNetExtractor, "__init__", extractor_init):
        result = run_one(args.mode, args.label, sample, str(work), False, True, 0,
                         png_decoder="libdeflate")
    assert effective.get("warmed"), "预热失败，不可比较"
    if args.decode_workers or args.label.startswith("after"):
        assert effective["requested"] == effective["actual"] == Config().opencv_threads
    if args.decode_workers and args.label.startswith("after"):
        assert effective["workers"] == args.decode_workers
    assert png_fast.scratch_mb() == 0
    result.update(effective=effective, variant=args.label.split("_")[0])
    (work / f"{args.label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RUN", args.label, result["wall_s"], result["cpu_s"], effective, flush=True)


def compare_whole(work, a, b):
    import numpy as np
    with np.load(work / f"whole_{a}" / "idx.coarse.npz", allow_pickle=True) as x, np.load(
            work / f"whole_{b}" / "idx.coarse.npz", allow_pickle=True) as y:
        assert set(x.files) == set(y.files)
        for key in x.files:
            assert np.array_equal(x[key], y[key]), key
    with np.load(work / f"whole_{a}" / "idx.fine.npz", allow_pickle=True) as x, np.load(
            work / f"whole_{b}" / "idx.fine.npz", allow_pickle=True) as y:
        assert np.array_equal(x["paths"], y["paths"])
        xf, yf = x["features"], y["features"]
        assert xf.shape == yf.shape
        cosine = np.sum(xf*yf, axis=1) / (np.linalg.norm(xf, axis=1)*np.linalg.norm(yf, axis=1))
        assert np.isfinite(cosine).all() and cosine.min() > .99999
        print("WHOLE_EQUAL", len(xf), "min_cosine", float(cosine.min()), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--n", type=int, default=1080)
    parser.add_argument("--mode", choices=("tiles", "whole"), default="tiles")
    parser.add_argument("--decode-workers", type=int, default=0,
                        help="候选外层线程数；0 保留原 OpenCV 策略 A/B 模式")
    parser.add_argument("--work")
    parser.add_argument("--label")
    args = parser.parse_args()
    if not 0 <= args.decode_workers <= 128 or args.n <= 0:
        parser.error("decode-workers 必须在 0..128，n 必须为正数")
    if args.work:
        work = Path(args.work).resolve()
        gallery = Path(GALLERY_ROOT).resolve()
        if work == gallery or gallery in work.parents:
            raise ValueError("禁止写图库")
        if args.label not in ("before_0", "after_1", "after_2", "before_3"):
            raise ValueError("无效的子进程标签")
        if (work / f"{args.mode}_{args.label}").exists():
            raise ValueError("拒绝覆写已有实验")
        child(args)
        return
    if not args.out:
        parser.error("需要 --out")
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("输出必须在图库外")
    sources = ["devtools/bench_opencv_runtime.py", "devtools/ab_build_bench.py",
               "hybrid_search/config.py", "hybrid_search/runtime.py", "hybrid_search/fine.py",
               "hybrid_search/tile_index.py", "hybrid_search/coarse.py", "hybrid_search/store.py"]
    work = create_archive(base, parameters={"script": Path(__file__).name, "n": args.n,
        "mode": args.mode, "decode_workers_candidate": args.decode_workers, "seed": 20260915,
        "order": ["before_0", "after_1", "after_2", "before_3"],
        "source_sha256": {p: digest(ROOT / p) for p in sources}})
    sample = build_sample(args.n, 0, 20260915)
    assert len(sample) == args.n
    (work / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    print("REPORT", work, flush=True)
    reports = []
    for i, variant in enumerate(("before", "after", "after", "before")):
        label = f"{variant}_{i}"
        other = foreign_python()
        if other:
            raise RuntimeError(f"其他 Python 仍在运行 {other}；未启动新基准，不终止其他任务")
        with (work / f"{label}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-E", "-B", "-X", "utf8", str(Path(__file__).resolve()),
                "--work", str(work), "--label", label, "--mode", args.mode,
                "--decode-workers", str(args.decode_workers)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (work / "failure.json").write_text(json.dumps({"label": label, "exit_code": result.returncode}), encoding="utf-8")
            raise RuntimeError(f"{label} 退出 {result.returncode}；保留日志，不自动重试")
        reports.append(json.loads((work / f"{label}.json").read_text(encoding="utf-8")))
        (work / "progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print("DONE", label, reports[-1]["wall_s"], reports[-1]["effective"], flush=True)
    with (work / "comparisons.log").open("x", encoding="utf-8") as log, redirect_stdout(log):
        for a, b in (("before_0", "after_1"), ("before_3", "after_2")):
            if args.mode == "tiles":
                assert compare_tile_arrays(str(work), a, b)
            else:
                compare_whole(work, a, b)
    summary = {"comparisons_passed": True, "mode": args.mode, "runs": reports,
               "decode_workers_candidate": args.decode_workers}
    for key in ("wall_s", "cpu_s"):
        before = statistics.median(r[key] for r in reports if r["variant"] == "before")
        after = statistics.median(r[key] for r in reports if r["variant"] == "after")
        summary[key] = {"before": before, "after": after, "change_percent": 100*(after/before-1)}
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), "REPORT", work, flush=True)


if __name__ == "__main__":
    main()
