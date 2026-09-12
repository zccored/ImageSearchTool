# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器（二值法粗筛 + ResNet 精排）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""CPU 侧成本诊断：
  A) 真实样本（建库同序前缀）的单张各阶段耗时分布与文件体积分布
  B) 解码阶段的线程扩展性（1/4/8/14/20 线程）
  C) 融合建库期间的真实 CPU 并行度（wall 时间 vs 累计 CPU 时间）

用法: python devtools/probe_cpu_cost.py [张数] [--threads]
"""
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from PIL import Image  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.coarse import extract_binary_features  # noqa: E402
from hybrid_search.fine import ResNetExtractor, _PRE_DOWNSCALE_PX, _PRE_DOWNSCALE_SIDE  # noqa: E402
from hybrid_search.io_utils import collect_images, decode_gray, read_bytes  # noqa: E402

PROC = psutil.Process()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
DO_THREADS = "--threads" in sys.argv
cfg = Config()
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)
print(f"样本 {len(paths)} 张（建库同序前缀）")


def pct(v, p):
    return float(np.percentile(v, p)) if v else float("nan")


# ---------------- A) 单张成本分布 ----------------
print("\n=== A) 单张成本分布（单线程实测）===")
rows = []
for p in paths[:200]:
    try:
        sz = os.path.getsize(p)
        t0 = time.time(); data = read_bytes(p); t_read = time.time() - t0
        if data is None:
            continue
        t0 = time.time(); rgb = decode_gray(data); t_gray = time.time() - t0
        t0 = time.time(); rgb2 = decode_gray(data); t_gray2 = time.time() - t0
        # 实际融合路径只需解码一次 RGB（灰度再由 RGB 转），这里对齐真实路径：
        from hybrid_search.io_utils import decode_rgb
        t0 = time.time(); img = decode_rgb(data); t_rgb = time.time() - t0
        if img is None:
            continue
        if img.shape[0] * img.shape[1] > _PRE_DOWNSCALE_PX:
            s = _PRE_DOWNSCALE_SIDE / max(img.shape[:2])
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)),
                                   max(1, int(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        t0 = time.time(); gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); t_cvt = time.time() - t0
        t0 = time.time(); extract_binary_features(gray, cfg); t_coarse = time.time() - t0
        rows.append(dict(sz=sz, read=t_read, decode=t_rgb, cvt=t_cvt,
                         coarse=t_coarse, px=img.shape[0] * img.shape[1]))
    except Exception:  # noqa: BLE001
        continue

if rows:
    sz = [r["sz"] / 2 ** 20 for r in rows]
    px = [r["px"] / 1e6 for r in rows]
    print(f"  样本 {len(rows)} 张 | 体积 中位 {statistics.median(sz):.2f}MB "
          f"均值 {statistics.mean(sz):.2f}MB P90 {pct(sz, 90):.2f}MB "
          f"最大 {max(sz):.2f}MB")
    print(f"  解码后像素 中位 {statistics.median(px):.2f}MP "
          f"均值 {statistics.mean(px):.2f}MP P90 {pct(px, 90):.2f}MP")
    for k, label in (("read", "读盘"), ("decode", "解码RGB"),
                     ("cvt", "RGB→灰度"), ("coarse", "粗筛特征")):
        v = [r[k] * 1000 for r in rows]
        print(f"  {label:<10} 中位 {statistics.median(v):7.1f}ms  "
              f"均值 {statistics.mean(v):7.1f}ms  P90 {pct(v, 90):7.1f}ms  "
              f"合计占比 {sum(v) / sum([r['read'] + r['decode'] + r['cvt'] + r['coarse'] for r in rows]) * 100:4.0f}%")
    # 变换（PIL 预处理）
    ex = ResNetExtractor(cfg)
    t_tr = []
    for p in paths[:60]:
        img = decode_rgb(read_bytes(p))
        if img is None:
            continue
        if img.shape[0] * img.shape[1] > _PRE_DOWNSCALE_PX:
            s = _PRE_DOWNSCALE_SIDE / max(img.shape[:2])
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)),
                                   max(1, int(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        t0 = time.time(); ex.transform(Image.fromarray(img)); t_tr.append(time.time() - t0)
    print(f"  {'预处理(ResNet)':<10} 中位 {statistics.median(t_tr) * 1000:7.1f}ms  "
          f"均值 {statistics.mean(t_tr) * 1000:7.1f}ms  P90 {pct(t_tr, 90) * 1000:7.1f}ms")
    tot = [ (r["read"] + r["decode"] + r["cvt"] + r["coarse"]) * 1000 for r in rows]
    print(f"  {'单张 CPU 合计':<10} 中位 {statistics.median(tot):7.1f}ms  "
          f"均值 {statistics.mean(tot):7.1f}ms  P90 {pct(tot, 90):7.1f}ms")

# ---------------- B) 线程扩展性 ----------------
if DO_THREADS:
    print("\n=== B) 解码线程扩展性（同批 120 张）===")
    sub = paths[:120]
    blobs = [read_bytes(p) for p in sub]
    blobs = [b for b in blobs if b]

    def decode_all(_):
        for b in blobs:
            decode_rgb(b)

    for nt in (1, 4, 8, 14, 20):
        decode_all(0)                     # 预热
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=nt) as pool:
            list(pool.map(decode_all, range(nt)))
        dt = time.time() - t0
        print(f"  {nt:2d} 线程: {dt:6.2f}s  {len(blobs) / dt:6.1f} 张/秒  "
              f"加速比 {len(blobs) / dt / (len(blobs) / (dt * nt)) * 1:.2f}")

# ---------------- C) 真实建库的 CPU 并行度 ----------------
print("\n=== C) 融合建库真实并行度（400 张）===")
import shutil  # noqa: E402
import tempfile  # noqa: E402

from hybrid_search.engine import HybridEngine  # noqa: E402

sub_paths = paths[:400]
tmp = tempfile.mkdtemp(prefix="cpu_probe_")
try:
    eng = HybridEngine(cfg)
    c0 = PROC.cpu_times()
    t0 = time.time()
    eng.build(os.path.join(tmp, "b"), paths=sub_paths, force=True)
    dt = time.time() - t0
    c1 = PROC.cpu_times()
finally:
    shutil.rmtree(tmp, ignore_errors=True)
cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
print(f"  wall {dt:.1f}s | 进程 CPU {cpu_s:.1f}s | 平均并行度 "
      f"{cpu_s / dt:.1f} 核（{psutil.cpu_count()} 逻辑核）")
print(f"  吞吐 {len(sub_paths) / dt:.1f} 张/秒 | 折算单张 CPU {cpu_s / len(sub_paths) * 1000:.0f} ms")
