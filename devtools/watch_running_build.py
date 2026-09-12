# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 非侵入观测运行中的建库进程：读盘速率 → 估算 图片/秒、瓦片/秒
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""非侵入观测正在运行的建库进程：读盘速率 → 估算 图片/秒、瓦片/秒。

不启动任何建库、不写任何文件，只采样 Win32_Process 的 IO/内存计数。
用法: python devtools/watch_running_build.py <pid> [采样秒数]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import psutil  # noqa: E402

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
AVG_IMG_MB = 1.94          # 抽样实测：F:\视频 前 400 张平均 1.94MB
TILES_PER_IMG = 10.47      # 全库 444,235 瓦片 / 42,414 张

p = psutil.Process(PID)
print(f"观测 PID {PID}（{p.name()}）{DUR:.0f}s\n")
print(f"{'时刻':>7}{'读MB/s':>10}{'RSS(MB)':>10}{'RSS增速(MB/s)':>14}"
      f"{'CPU(核)':>9}{'估算图/s':>10}{'估算瓦片/s':>11}")
try:
    io0 = p.io_counters()
    rss0 = p.memory_info().rss
    c0 = p.cpu_times()
    t0 = time.time()
    last = (io0, rss0, c0, t0)
    rows = []
    while time.time() - t0 < DUR:
        time.sleep(5)
        now = time.time()
        io1 = p.io_counters()
        rss1 = p.memory_info().rss
        c1 = p.cpu_times()
        dt = now - last[3]
        rd = (io1.read_bytes - last[0].read_bytes) / 2 ** 20 / dt
        rss_rate = (rss1 - last[1]) / 2 ** 20 / dt
        cores = ((c1.user + c1.system) - (last[2].user + last[2].system)) / dt
        img_s = rd / AVG_IMG_MB
        print(f"{now - t0:6.0f}s{rd:10.1f}{rss1 / 2 ** 20:10.0f}"
              f"{rss_rate:14.1f}{cores:9.1f}{img_s:10.1f}"
              f"{img_s * TILES_PER_IMG:11.0f}")
        rows.append((rd, img_s, cores, rss_rate))
        last = (io1, rss1, c1, now)
finally:
    pass

if rows:
    import statistics
    rd = statistics.median(r[0] for r in rows)
    img = statistics.median(r[1] for r in rows)
    cores = statistics.median(r[2] for r in rows)
    rss_r = statistics.median(r[3] for r in rows)
    print(f"\n中位：读 {rd:.1f} MB/s → 图 {img:.1f} 张/s → 瓦片 "
          f"{img * TILES_PER_IMG:.0f} 块/s；CPU {cores:.1f} 核；"
          f"RSS 增速 {rss_r:.1f} MB/s")
    print(f"按当前 RSS 增速估算每瓦片内存 ≈ "
          f"{rss_r / max(img * TILES_PER_IMG, 1e-6) * 1024:.0f} KB")
