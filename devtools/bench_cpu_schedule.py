# -*- coding: utf-8 -*-
# ImageSearchTool · CPU 并发与 GPU 发射的可复现对照
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""只在进程内替换并发配置；图库只读，原生产默认值保持不变。"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from hybrid_search.fine import ResNetExtractor
from paths import GALLERY_ROOT


def main():
    import cv2
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1080)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cv2-threads", type=int)
    parser.add_argument("--format", choices=("mixed", "png", "jpeg"), default="mixed")
    args = parser.parse_args()
    from hybrid_search.config import Config
    if hasattr(Config(), "opencv_threads"):
        parser.error("生产版已加入一次性线程配置，请使用 bench_opencv_runtime.py 的独立子进程对照；"
                     "此旧原型禁止绕过初始化策略")
    if args.workers is None and args.cv2_threads is None:
        parser.error("至少指定一个候选并发参数")
    if any(x is not None and x < 1 for x in (args.workers, args.cv2_threads)):
        parser.error("候选并发数必须为正整数")
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("输出必须在图库外")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cpu_schedule_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    if args.format != "mixed":
        extensions = {".png"} if args.format == "png" else {".jpg", ".jpeg"}
        sample = [p for p in sample if Path(p).suffix.lower() in extensions]
    if not sample:
        raise ValueError("没有符合格式的样本")
    original_init = ResNetExtractor.__init__
    original_forward = ResNetExtractor._forward
    original_cv_threads = cv2.getNumThreads()
    reports = []
    try:
        for i, variant in enumerate(("before", "after", "after", "before")):
            latencies = []
            effective = {}
            cv2.setNumThreads(args.cv2_threads if variant == "after" and args.cv2_threads is not None
                              else original_cv_threads)

            def init(ex, cfg):
                if variant == "after" and args.workers is not None:
                    cfg.decode_workers = args.workers
                original_init(ex, cfg)
                if ex.device != "cuda":
                    raise RuntimeError("本基准需要 CUDA")
                effective.update(workers=ex.decode_workers, cv2_threads=cv2.getNumThreads(),
                                 torch_threads=torch.get_num_threads(), tile_decode_slots=cfg.tile_decode_slots)
                data = torch.zeros((64, 3, 224, 224), device=ex.device)
                with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
                    for _ in range(3):
                        ex.model(data)
                torch.cuda.synchronize()

            def forward(ex, tensors):
                start = time.perf_counter()
                try:
                    return original_forward(ex, tensors)
                finally:
                    latencies.append({"rows": len(tensors), "seconds": time.perf_counter() - start})

            with patch.object(ResNetExtractor, "__init__", init), patch.object(
                    ResNetExtractor, "_forward", forward):
                result = run_one("tiles", f"{variant}_{i}", sample, str(work), False, True, 0,
                                 png_decoder="libdeflate")
            fixed = sorted(r["seconds"] for r in latencies if r["rows"] == 64)
            result.update(variant=variant, format=args.format, effective=effective,
                          batches=latencies, forward_total_s=sum(r["seconds"] for r in latencies),
                          full_batch_p95_s=fixed[min(len(fixed)-1, int(len(fixed)*.95))] if fixed else None)
            reports.append(result)
            (work / "abba.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            print("RUN", variant, result["wall_s"], result["cpu_s"], effective, flush=True)
        for a, b in ((reports[0], reports[1]), (reports[3], reports[2])):
            assert compare_tile_arrays(str(work), a["label"], b["label"])
        summary = {}
        for key in ("wall_s", "cpu_s", "forward_total_s", "full_batch_p95_s"):
            before = statistics.median(r[key] for r in reports if r["variant"] == "before")
            after = statistics.median(r[key] for r in reports if r["variant"] == "after")
            summary[key] = {"before": before, "after": after, "change_percent": 100 * (after / before - 1)}
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("SUMMARY", json.dumps(summary), "REPORT", work, flush=True)
    finally:
        cv2.setNumThreads(original_cv_threads)


if __name__ == "__main__":
    main()
