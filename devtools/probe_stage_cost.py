# -*- coding: utf-8 -*-
"""拆解单张图的融合建库各阶段耗时（读盘/解码/粗筛特征/ResNet 预处理/前向）。

用法: python devtools/probe_stage_cost.py [张数]
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.coarse import extract_binary_features  # noqa: E402
from hybrid_search.fine import ResNetExtractor  # noqa: E402
from hybrid_search.io_utils import collect_images, read_bytes  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
cfg = Config()
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)
print(f"样本 {len(paths)} 张（前 {N} 个自然序）")

ex = ResNetExtractor(cfg)
stats = {k: [] for k in ("read", "decode_gray", "decode_rgb", "coarse",
                         "transform", "forward")}
sizes = []
for p in paths:
    try:
        sizes.append(os.path.getsize(p))
        t0 = time.time()
        data = read_bytes(p)
        stats["read"].append(time.time() - t0)
        if data is None:
            continue
        from hybrid_search.io_utils import decode_gray, decode_rgb
        t0 = time.time()
        gray = decode_gray(data)
        stats["decode_gray"].append(time.time() - t0)
        t0 = time.time()
        rgb = decode_rgb(data)
        stats["decode_rgb"].append(time.time() - t0)
        if gray is None or rgb is None:
            continue
        t0 = time.time()
        extract_binary_features(gray, cfg)
        stats["coarse"].append(time.time() - t0)
        t0 = time.time()
        tensor = ex.transform(Image.fromarray(rgb))
        stats["transform"].append(time.time() - t0)
    except Exception as e:  # noqa: BLE001
        print("  跳过:", os.path.basename(p), e)

# GPU 前向：按 batch=64 折算每张成本
try:
    import torch
    batch = [ex.transform(Image.fromarray(np.zeros((224, 224, 3), np.uint8)))
             for _ in range(64)]
    for _ in range(3):                       # 预热
        ex._forward(batch)
    t0 = time.time()
    ex._forward(batch)
    fwd = (time.time() - t0) / 64
except Exception as e:  # noqa: BLE001
    print("  前向测量失败:", e)
    fwd = float("nan")

print(f"\n平均文件体积 {statistics.mean(sizes) / 2 ** 20:.1f} MB")
print(f"{'阶段':<16}{'均值(ms)':>10}{'中位(ms)':>10}{'P90(ms)':>10}")
for k, v in stats.items():
    if not v:
        continue
    print(f"{k:<16}{statistics.mean(v) * 1000:10.1f}"
          f"{statistics.median(v) * 1000:10.1f}"
          f"{np.percentile(v, 90) * 1000:10.1f}")
print(f"{'GPU前向(每张)':<16}{fwd * 1000:10.2f}")

cpu_per_img = sum(statistics.mean(v) for v in stats.values() if v)
print(f"\nCPU 合计 ≈ {cpu_per_img * 1000:.0f} ms/张  vs  GPU 前向 "
      f"{fwd * 1000:.2f} ms/张  →  GPU 理论利用率上限约 "
      f"{fwd / max(cpu_per_img, 1e-6) * 100:.1f}%（若只靠单线程 CPU 供料）")
