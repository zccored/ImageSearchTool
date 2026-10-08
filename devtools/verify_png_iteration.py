# -*- coding: utf-8 -*-
# ImageSearchTool · PNG 探测优化 A/B 与缓存端到端验证
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""用修改前 io_utils.py 的备份隔离比较 _probe；只向新建报告目录写测试索引。

python -E -B devtools/verify_png_iteration.py --baseline-io <备份/io_utils.py>
性能数据取既有 ab_build_bench.run_one；ABBA 顺序，缓存关闭，加载模型不计时。
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import struct
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "devtools"))
from ab_build_bench import build_sample, run_one, compare_tile_arrays  # noqa: E402
from hybrid_search import io_utils as iu  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.store import IndexFiles  # noqa: E402
from paths import GALLERY_ROOT  # noqa: E402


def compare_whole(a, b):
    left, right = IndexFiles(str(a)), IndexFiles(str(b))
    ca, cb = left.load_coarse(), right.load_coarse()
    for key in ("paths", "md5s", "hu", "fp", "hu_mean", "hu_std"):
        np.testing.assert_array_equal(ca[key], cb[key], err_msg=key)
    fa, fb = left.load_fine(), right.load_fine()
    np.testing.assert_array_equal(fa["paths"], fb["paths"])
    np.testing.assert_allclose(fa["features"], fb["features"], rtol=0, atol=1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-io", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--pixel-samples", type=int, default=150)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("baseline_io", args.baseline_io)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    current_probe = iu._probe
    out = Path(tempfile.mkdtemp(prefix="iteration_png_", dir=ROOT / "perf_reports"))
    print("OUTPUT", out, flush=True)
    pngs = iu.collect_images(GALLERY_ROOT, [".png"])
    random.Random(7).shuffle(pngs)
    iu.set_png_decoder("libdeflate", silence_noise=True)
    checked, fast = 0, 0
    for path in pngs[:args.pixel_samples]:
        data = iu.read_bytes(path)
        with patch.object(iu, "_probe", baseline._probe):
            before = iu.decode_rgb(data)
        after = iu.decode_rgb(data)
        if before is None or after is None:
            assert before is None and after is None, "decode acceptance changed"
        else:
            np.testing.assert_array_equal(before, after)
        assert baseline._probe(data) == current_probe(data), "metadata changed"
        checked += 1
        fast += int(data[:8] == b"\x89PNG\r\n\x1a\n" and not iu._png_needs_exif_load(data))
    report = {"pixels_equal": checked, "fast_probe_images": fast,
              "sample_n": args.n, "runs": [], "comparisons": []}
    print("PIXELS_EQUAL", checked, "FAST_PROBE", fast, flush=True)
    sample = build_sample(args.n, 0, 20260915)
    for mode in ("whole", "tiles"):
        runs = []
        for i, variant in enumerate(("before", "after", "after", "before")):
            label = "%s_%s_%d" % (mode, variant, i)
            probe = baseline._probe if variant == "before" else current_probe
            with patch.object(iu, "_probe", probe):
                res = run_one(mode, label, sample, str(out), False, True, 0,
                              png_decoder="libdeflate")
            res["variant"] = variant
            runs.append(res)
            print("RUN", label, res["wall_s"], res["cpu_s"], flush=True)
        for a, b in ((runs[0], runs[1]), (runs[3], runs[2])):
            if mode == "tiles":
                assert compare_tile_arrays(str(out), a["label"], b["label"])
            else:
                compare_whole(out / (mode + "_" + a["label"]) / "idx",
                              out / (mode + "_" + b["label"]) / "idx")
        metrics = {"mode": mode}
        for metric in ("wall_s", "cpu_s"):
            before = statistics.median(r[metric] for r in runs if r["variant"] == "before")
            after = statistics.median(r[metric] for r in runs if r["variant"] == "after")
            metrics[metric] = {"before": before, "after": after,
                               "change_percent": (after / before - 1) * 100}
        report["runs"].extend(runs)
        report["comparisons"].append(metrics)
        print("SUMMARY", json.dumps(metrics), flush=True)
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 真正落盘的缓存往返；每轮新建引擎以排除 L2 内存命中。
    cache_paths = sample[:24]
    cold, warm = out / "cache" / "cold", out / "cache" / "warm"
    stats = []
    for prefix in (cold, warm):
        eng = HybridEngine(Config())
        assert eng.build(str(prefix), paths=cache_paths) == len(cache_paths)
        stats.append(eng._prep_cache.stats())
        eng.release_fine()
    compare_whole(cold, warm)
    codecs = []
    for path in (out / "cache" / "prep_cache").rglob("*.bin"):
        data = path.read_bytes()
        size = struct.unpack("<I", data[5:9])[0]
        codecs.append(json.loads(data[9:9 + size])["codec"])
    assert len(codecs) == len(cache_paths) and set(codecs) == {"ldf6"}
    assert stats[1]["hits"] == len(cache_paths)
    report["cache"] = {"stats": stats, "codecs": {"ldf6": len(codecs)},
                       "coarse_exact_fine_atol": 1e-5}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("CACHE_PASS", json.dumps(report["cache"]), flush=True)
    print("REPORT", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
