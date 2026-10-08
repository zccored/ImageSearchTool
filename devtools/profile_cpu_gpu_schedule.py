# -*- coding: utf-8 -*-
# ImageSearchTool · CPU 解码供给与 GPU 前向分段诊断
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""独立测试索引；CPU 线程时间与 CUDA 事件分别计量，并发墙钟不可相加。"""
import argparse
import json
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one
from paths import GALLERY_ROOT
from hybrid_search import tile_index as ti
from hybrid_search.fine import ResNetExtractor
from hybrid_search import png_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=540)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("测试输出必须在图库之外")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cpu_gpu_schedule_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    stats = defaultdict(lambda: {"calls": 0, "wall_s": 0., "thread_cpu_s": 0.})
    lock = threading.Lock()
    batches = []

    def timed(fn, key_of):
        def wrapper(*a, **kw):
            key = key_of(*a, **kw)
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return fn(*a, **kw)
            finally:
                wall, cpu = time.perf_counter() - wall, time.thread_time() - cpu
                with lock:
                    row = stats[key]
                    row["calls"] += 1
                    row["wall_s"] += wall
                    row["thread_cpu_s"] += cpu
        return wrapper

    original_forward = ResNetExtractor._forward

    def forward(ex, tensors):
        import torch
        if ex.device != "cuda":
            return original_forward(ex, tensors)
        start = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        with torch.no_grad():
            batch = torch.stack(tensors, dim=0)
            stacked = time.perf_counter()
            events[0].record()
            batch = batch.to(ex.device)
            events[1].record()
            if batch.dtype == torch.uint8:
                batch = ex._tile_scale_lut[batch.long()]
            if ex.norm_on_gpu:
                batch = batch.sub(ex._mean_t).div_(ex._std_t)
            events[2].record()
            if ex.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = ex.model(batch)
            else:
                features = ex.model(batch)
            events[3].record()
            result = features.float().cpu().numpy()
        end = time.perf_counter()
        batches.append({"rows": len(tensors), "start": start, "end": end,
                        "stack_ms": 1000 * (stacked - start),
                        "h2d_ms": events[0].elapsed_time(events[1]),
                        "normalize_ms": events[1].elapsed_time(events[2]),
                        "model_ms": events[2].elapsed_time(events[3])})
        return result

    def codec(data):
        return "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg" if data[:2] == b"\xff\xd8" else "other"

    with ExitStack() as native_patches, \
         patch.object(ti, "decode_rgb", timed(ti.decode_rgb, lambda data: "decode_" + codec(data))), \
         patch.object(ti, "_feature_one_tile", timed(ti._feature_one_tile, lambda *a, **k: "tile_feature")), \
         patch.object(ResNetExtractor, "_forward", forward):
        pair = png_fast._load_libdeflate()
        native = png_fast._load_native()
        if pair:
            lib = pair[0]
            native_patches.enter_context(patch.object(
                lib, "libdeflate_zlib_decompress_ex", timed(
                    lib.libdeflate_zlib_decompress_ex, lambda *a: "png_libdeflate")))
        if native:
            native_patches.enter_context(patch.object(
                native, "unfilter_to_rgb", timed(
                    native.unfilter_to_rgb, lambda *a: "png_unfilter")))
        result = run_one("tiles", "profile", sample, str(work), False, True, 0,
                         png_decoder="libdeflate")
    totals = {key: sum(row[key] for row in batches) / 1000
              for key in ("stack_ms", "h2d_ms", "normalize_ms", "model_ms")}
    totals["forward_wall_s"] = sum(row["end"] - row["start"] for row in batches)
    totals["between_forward_s"] = sum(max(0., b["start"] - a["end"])
                                        for a, b in zip(batches, batches[1:]))
    report = {"result": result, "cpu_stages": dict(stats), "gpu_totals_s": totals,
              "batches": batches, "note": "CUDA event intervals include scheduling gaps; not kernel-active utilization"}
    (work / "profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"result": result, "cpu_stages": dict(stats), "gpu_totals_s": totals}, indent=2), flush=True)
    print("REPORT", work, flush=True)


if __name__ == "__main__":
    main()
