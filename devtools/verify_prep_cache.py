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

"""验证预处理缓存：冷建库 vs 热建库（命中缓存）的吞吐、CPU、GPU，以及
**索引是否逐位一致**（指纹/特征完全相同）。

用法: python devtools/verify_prep_cache.py [张数]
"""
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402
from hybrid_search.store import IndexFiles  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PROC = psutil.Process()
cfg = Config()
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)
print(f"样本 {len(paths)} 张")

try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001
    _h = None


def run_once(prefix: str, tag: str):
    samples, stop = [], [False]

    def sampler():
        while not stop[0]:
            if _h is not None:
                try:
                    samples.append(pynvml.nvmlDeviceGetUtilizationRates(_h).gpu)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.2)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        eng = HybridEngine(cfg)
        c0 = PROC.cpu_times()
        t0 = time.time()
        n = eng.build(prefix, paths=paths, force=True)
        dt = time.time() - t0
        c1 = PROC.cpu_times()
        cache = getattr(eng, "_prep_cache", None)
        stats = cache.stats() if cache else {}
    finally:
        stop[0] = True
        th.join(timeout=1)
    cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
    print(f"  [{tag}] {n} 张 {dt:.2f}s → {n / dt:.0f} 张/秒 | CPU {cpu_s:.1f}s"
          f"（{cpu_s / dt:.1f} 核，{cpu_s / max(n, 1) * 1000:.0f} ms/张）| "
          f"GPU 均值 {sum(samples) / max(len(samples), 1):.1f}% | "
          f"缓存 命中 {stats.get('hits', 0)} / 写入 {stats.get('writes', 0)}")
    return dt, cpu_s


work = tempfile.mkdtemp(prefix="prepcache_")
prefix = os.path.join(work, "idx", "gallery")
try:
    print("\n=== 1) 冷建库（空缓存，全部解码）===")
    dt1, cpu1 = run_once(prefix, "cold")
    files = IndexFiles(prefix)
    st_cold = files.load_coarse()
    hu1 = np.array(st_cold["hu"])
    fp1 = np.array(st_cold["fp"])
    fine1 = np.array(files.load_fine()["features"])
    print(f"  索引: 粗筛 {hu1.shape} / 指纹 {fp1.shape} / 精排 {fine1.shape}")

    print("\n=== 2) 热建库（缓存命中，跳过解码）===")
    dt2, cpu2 = run_once(prefix, "warm")
    st_warm = files.load_coarse()
    hu2 = np.array(st_warm["hu"])
    fp2 = np.array(st_warm["fp"])
    fine2 = np.array(files.load_fine()["features"])

    print("\n=== 3) 一致性（热建索引 vs 冷建索引）===")
    same_hu = bool(np.array_equal(hu1, hu2))
    same_fp = bool(np.array_equal(fp1, fp2))
    same_fine = bool(np.array_equal(fine1, fine2))
    max_cos_gap = float(np.abs((fine1 * fine2).sum(axis=1) - 1).max())
    print(f"  Hu 逐位一致: {same_hu} | 指纹逐位一致: {same_fp} | "
          f"精排特征逐位一致: {same_fine}（余弦最大偏差 {max_cos_gap:.2e}）")
    print(f"  行数一致: {hu1.shape[0] == hu2.shape[0]}")

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(prefix)), "prep_cache")
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _d, fs in os.walk(cache_dir) for f in fs)
    cnt = sum(len(fs) for _r, _d, fs in os.walk(cache_dir))
    print(f"\n=== 4) 收益 ===")
    print(f"  wall: {dt1:.2f}s → {dt2:.2f}s（提速 {dt1 / dt2:.1f}×）")
    print(f"  CPU : {cpu1:.1f}s → {cpu2:.1f}s（省 {cpu1 - cpu2:.1f}s，"
          f"{(1 - cpu2 / cpu1) * 100:.0f}%）")
    print(f"  单张: {cpu1 / len(paths) * 1000:.0f} ms → {cpu2 / len(paths) * 1000:.0f} ms")
    print(f"  缓存: {cnt} 个文件 / {total / 2 ** 20:.1f} MB"
          f"（{total / max(cnt, 1) / 1024:.0f} KB/张，全库 3.7 万张约 "
          f"{total / max(cnt, 1) * 37683 / 2 ** 30:.1f} GB）")
    _ok = same_hu and same_fp and same_fine
    print("\\n结果:", "✓ 缓存命中且索引与全新建库逐位一致" if _ok else "✗ 一致性校验失败")
finally:
    shutil.rmtree(work, ignore_errors=True)
