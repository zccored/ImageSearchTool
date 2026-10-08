# -*- coding: utf-8 -*-
# ImageSearchTool · 解码并发下 CUDA Graph 发射对照
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""进程内试验，只捕获 64 行模型前向，其余形状走原模型。"""
import argparse
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
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=540)
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("输出必须在图库外")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cuda_graph_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    original_init = ResNetExtractor.__init__

    capture_costs = []

    def init(ex, cfg, capture=True):
        original_init(ex, cfg)
        if ex.device != "cuda" or not ex.use_fp16:
            return
        static = torch.zeros((64, 3, 224, 224), device=ex.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            for _ in range(3):
                ex.model(static)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if not capture:
            return
        import time
        capture_start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            result = ex.model(static)
        torch.cuda.synchronize()
        capture_costs.append(time.perf_counter() - capture_start)
        eager = ex.model.forward

        def graph_forward(data):
            if data.shape != static.shape or data.dtype != static.dtype:
                return eager(data)
            static.copy_(data)
            graph.replay()
            return result

        ex.model.forward = graph_forward

    def baseline_init(ex, cfg):
        init(ex, cfg, capture=False)

    reports = []
    for i, variant in enumerate(("before", "after", "after", "before")):
        label = f"{variant}_{i}"
        with patch.object(ResNetExtractor, "__init__", init if variant == "after" else baseline_init):
            result = run_one("tiles", label, sample, str(work), False, True, 0,
                             png_decoder="libdeflate")
        result["variant"] = variant
        result["capture_s"] = capture_costs[-1] if variant == "after" else 0.
        result["wall_plus_capture_s"] = result["wall_s"] + result["capture_s"]
        reports.append(result)
        (work / "abba.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print("RUN", label, result["wall_s"], result["cpu_s"], "LOAD", result["load_s"], flush=True)
    for a, b in ((reports[0], reports[1]), (reports[3], reports[2])):
        assert compare_tile_arrays(str(work), a["label"], b["label"])
    summary = {}
    for key in ("wall_s", "wall_plus_capture_s", "cpu_s"):
        before = statistics.median(r[key] for r in reports if r["variant"] == "before")
        after = statistics.median(r[key] for r in reports if r["variant"] == "after")
        summary[key] = {"before": before, "after": after, "change_percent": 100 * (after / before - 1)}
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), "REPORT", work, flush=True)


if __name__ == "__main__":
    main()
