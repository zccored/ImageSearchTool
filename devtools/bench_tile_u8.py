# -*- coding: utf-8 -*-
# ImageSearchTool · CPU/GPU uint8 交接对照试验
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""加载备份 fine 模块对照当前实现，实际索引写入唯一新目录。"""
import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from paths import GALLERY_ROOT
from hybrid_search.fine import ResNetExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=540)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-fine", required=True)
    parser.add_argument("--format", choices=("mixed", "png", "jpeg"), default="mixed")
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("输出必须在图库外")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tile_u8_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    if args.format != "mixed":
        extensions = {".png"} if args.format == "png" else {".jpg", ".jpeg"}
        sample = [p for p in sample if Path(p).suffix.lower() in extensions]
    if not sample:
        raise ValueError("没有符合格式的样本")
    spec = importlib.util.spec_from_file_location("hybrid_search._baseline_fine", args.baseline_fine)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    original_init, original_forward = ResNetExtractor.__init__, ResNetExtractor._forward

    reports = []
    for i, variant in enumerate(("before", "after", "after", "before")):
        label = f"{variant}_{i}"
        with patch.object(ResNetExtractor, "__init__", original_init if variant == "after" else baseline.ResNetExtractor.__init__), \
             patch.object(ResNetExtractor, "_forward", original_forward if variant == "after" else baseline.ResNetExtractor._forward):
            result = run_one("tiles", label, sample, str(work), False, True, 0,
                             png_decoder="libdeflate")
        result["variant"] = variant
        result["format"] = args.format
        reports.append(result)
        (work / "abba.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print("RUN", label, result["wall_s"], result["cpu_s"], flush=True)
    for a, b in ((reports[0], reports[1]), (reports[3], reports[2])):
        assert compare_tile_arrays(str(work), a["label"], b["label"])
    summary = {}
    for key in ("wall_s", "cpu_s"):
        before = statistics.median(r[key] for r in reports if r["variant"] == "before")
        after = statistics.median(r[key] for r in reports if r["variant"] == "after")
        summary[key] = {"before": before, "after": after, "change_percent": 100 * (after / before - 1)}
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), "REPORT", work, flush=True)


if __name__ == "__main__":
    main()
