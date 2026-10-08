# -*- coding: utf-8 -*-
# ImageSearchTool · 有界 GPU 预准备实验
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""仅进程内原型：快照不消费队列；精确匹配下一批才复用，否则回退。"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from hybrid_search.fine import ResNetExtractor
from paths import GALLERY_ROOT


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=540)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("输出必须在图库外")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tile_prefetch_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    original_init = ResNetExtractor.__init__
    original_forward = ResNetExtractor._forward

    def init(ex, cfg):
        original_init(ex, cfg)
        if ex.device != "cuda" or not ex.norm_on_gpu:
            raise RuntimeError("需要 CUDA 归一化路径")
        data = torch.zeros((64, 3, 224, 224), device=ex.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
            for _ in range(3):
                ex.model(data)
        torch.cuda.synchronize()

    reports = []
    for i, variant in enumerate(("before", "after", "after", "before")):
        state = {"future": None, "hits": 0, "misses": 0, "submitted": 0,
                 "not_ready": 0, "future_wait_s": 0.0}
        latencies = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            def prepare(ex, tensors):
                stream = torch.cuda.Stream(priority=0)
                with torch.no_grad(), torch.cuda.stream(stream):
                    host = torch.stack(tensors).pin_memory()
                    data = host.to(ex.device, non_blocking=True)
                    data = ex._tile_scale_lut[data.long()]
                    data = data.sub(ex._mean_t).div_(ex._std_t)
                # 明确等待完成并保留 host/data/tensors，不让异步拷贝越过所有权。
                stream.synchronize()
                return tensors, data

            def forward(ex, tensors):
                cached = None
                future = state["future"]
                if future is not None:
                    if args.no_wait and not future.done():
                        state["not_ready"] += 1
                    else:
                        wait_start = time.perf_counter()
                        cached = future.result()
                        state["future_wait_s"] += time.perf_counter() - wait_start
                        state["future"] = None
                frame = sys._getframe(1)
                upcoming = []
                try:
                    while frame and frame.f_code.co_name != "_ingest_tiles":
                        frame = frame.f_back
                    if frame:
                        ready = frame.f_locals["ready_q"]
                        with ready.mutex:
                            for item in list(ready.queue)[:64]:
                                if item is None:
                                    break
                                upcoming.extend(t for t, _ in item[1])
                finally:
                    del frame
                if len(upcoming) == 64 and state["future"] is None:
                    state["future"] = pool.submit(prepare, ex, upcoming)
                    state["submitted"] += 1
                if cached and len(cached[0]) == len(tensors) and all(
                        a is b for a, b in zip(cached[0], tensors)):
                    state["hits"] += 1
                    with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
                        result = ex.model(cached[1])
                    return result.float().cpu().numpy()
                state["misses"] += 1
                return original_forward(ex, tensors)

            def measured_forward(ex, tensors):
                start = time.perf_counter()
                try:
                    return (forward if variant == "after" else original_forward)(ex, tensors)
                finally:
                    latencies.append(time.perf_counter() - start)

            with patch.object(ResNetExtractor, "__init__", init), patch.object(
                    ResNetExtractor, "_forward", measured_forward):
                result = run_one("tiles", f"{variant}_{i}", sample, str(work), False, True, 0,
                                 png_decoder="libdeflate")
            if state["future"]:
                state["future"].result()
            state.pop("future")
        ordered = sorted(latencies)
        result.update(variant=variant, prefetch=state, no_wait=args.no_wait,
                      forward_p95_s=ordered[min(len(ordered)-1, int(len(ordered)*.95))],
                      forward_total_s=sum(latencies))
        reports.append(result)
        (work / "abba.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print("RUN", variant, result["wall_s"], state, flush=True)
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
