# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 探针：不同 decode_workers 下的 wall/CPU/吞吐对照
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""线程数对照：融合建库在不同 decode_workers 下的 wall / CPU / 吞吐。

用法: python devtools/probe_decode_threads.py [张数] [线程列表，如 6,10,14,20]
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import psutil  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
THREADS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 \
    else [6, 10, 14, 20]
PROC = psutil.Process()
print(f"样本 {N} 张 | 逻辑核 {psutil.cpu_count()} / 物理核 "
      f"{psutil.cpu_count(logical=False)}")

print(f"\n{'线程':>4}{'wall(s)':>9}{'吞吐(张/秒)':>12}{'CPU(s)':>9}"
      f"{'并行度':>8}{'CPU/张(ms)':>11}{'GPU 均值':>9}")
try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001
    _h = None

for nt in THREADS:
    cfg = Config()
    cfg.decode_workers = nt
    cfg.workers = min(8, nt)
    paths = collect_images(r"F:\视频", cfg.extensions, limit=N)
    tmp = tempfile.mkdtemp(prefix="thr_")
    gpu_samples = []
    stop = [False]
    import threading

    def sampler():
        while not stop[0]:
            if _h is not None:
                try:
                    gpu_samples.append(pynvml.nvmlDeviceGetUtilizationRates(_h).gpu)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.2)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        eng = HybridEngine(cfg)
        c0 = PROC.cpu_times()
        t0 = time.time()
        n = eng.build(os.path.join(tmp, "b"), paths=paths, force=True)
        dt = time.time() - t0
        c1 = PROC.cpu_times()
    finally:
        stop[0] = True
        th.join(timeout=1)
        shutil.rmtree(tmp, ignore_errors=True)
    cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
    gpu_avg = sum(gpu_samples) / max(len(gpu_samples), 1)
    print(f"{nt:>4}{dt:>9.2f}{n / dt:>12.0f}{cpu_s:>9.1f}"
          f"{cpu_s / dt:>8.1f}{cpu_s / max(n, 1) * 1000:>11.0f}{gpu_avg:>9.1f}")
