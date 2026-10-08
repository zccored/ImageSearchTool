# -*- coding: utf-8 -*-
# ImageSearchTool · 子图建库分段耗时回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""图库只读，产物写唯一新目录；并发耗时为累计墙钟，不能当作 CPU 核秒相减。"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import statistics
import threading
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from hybrid_search import tile_index as ti
from hybrid_search.fine import ResNetExtractor
from paths import GALLERY_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=540)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", help="修改前 tile_index.py；指定后运行 ABBA")
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("禁止在图库中写测试产物")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tile_profile_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    if args.baseline:
        spec = importlib.util.spec_from_file_location("hybrid_search._baseline_tile", args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        reports = []
        current_ingest = ti._ingest_tiles
        current_forward = ResNetExtractor._forward
        for index, variant in enumerate(("before", "after", "after", "before")):
            batches = []

            def forward(self, tensors):
                batches.append(len(tensors))
                return current_forward(self, tensors)

            label = f"{variant}_{index}"
            ingest = baseline._ingest_tiles if variant == "before" else current_ingest
            with patch.object(ti, "_ingest_tiles", ingest), \
                 patch.object(ResNetExtractor, "_forward", forward):
                result = run_one("tiles", label, sample, str(work), False, True, 0,
                                 png_decoder="libdeflate")
            result["variant"] = variant
            result["forward_batches"] = batches
            reports.append(result)
            print("RUN", label, result["wall_s"], result["cpu_s"], "batches", len(batches), flush=True)
            (work / "abba.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        for a, b in ((reports[0], reports[1]), (reports[3], reports[2])):
            assert compare_tile_arrays(str(work), a["label"], b["label"])
        summary = {}
        for key in ("wall_s", "cpu_s"):
            before = statistics.median(r[key] for r in reports if r["variant"] == "before")
            after = statistics.median(r[key] for r in reports if r["variant"] == "after")
            summary[key] = {"before": before, "after": after,
                            "change_percent": 100 * (after / before - 1)}
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("SUMMARY", json.dumps(summary), "REPORT", work, flush=True)
        return
    lock = threading.Lock()
    stats = {}

    def timed(fn, key):
        def wrapper(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                elapsed = time.perf_counter() - start
                with lock:
                    total, count = stats.get(key, (0.0, 0))
                    stats[key] = (total + elapsed, count + 1)
        return wrapper

    with patch.object(ti, "_decode_to_crops", timed(ti._decode_to_crops, "decode_crop")), \
         patch.object(ti, "_feature_one_tile", timed(ti._feature_one_tile, "tile_feature")), \
         patch.object(ti, "extract_binary_features", timed(ti.extract_binary_features, "binary_feature")), \
         patch.object(ResNetExtractor, "_forward", timed(ResNetExtractor._forward, "forward")):
        result = run_one("tiles", "profile", sample, str(work), False, True, 0,
                         png_decoder="libdeflate")
    report = {"result": result, "cumulative_wall_seconds_and_calls": stats,
              "note": "tile_feature contains binary_feature; concurrent timings overlap"}
    (work / "profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("REPORT", work, flush=True)


if __name__ == "__main__":
    main()
