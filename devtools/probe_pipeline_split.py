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

"""定位融合建库的瓶颈：生产端（解码）还是消费端（粗筛入库 + GPU 前向）。

做法：monkeypatch 计时点，跑真实建库，报告
  * wall 时间
  * 消费端忙时 = Σ add_results + Σ _forward
  * 生产端等待 = wall − 消费端忙时（≈ 消费端在等批次 = GPU/主线程空窗）
  * 每批耗时分布

用法: python devtools/probe_pipeline_split.py [张数]
"""
import os
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.coarse import CoarseIndex  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.fine import ResNetExtractor  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
PROC = psutil.Process()
cfg = Config()
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)
print(f"样本 {len(paths)} 张（与建库同序前缀）")


def wrap(obj, name, acc, counter):
    orig = getattr(obj, name)

    def timed(*a, **kw):
        t0 = time.time()
        try:
            return orig(*a, **kw)
        finally:
            acc.append(time.time() - t0)
            counter[0] += 1
    setattr(obj, name, timed)
    return orig


add_t, fwd_t = [], []
add_n, fwd_n = [0], [0]
wrap(CoarseIndex, "add_results", add_t, add_n)
wrap(ResNetExtractor, "_forward", fwd_t, fwd_n)

tmp = tempfile.mkdtemp(prefix="split_")
try:
    eng = HybridEngine(cfg)
    c0 = PROC.cpu_times()
    t0 = time.time()
    n = eng.build(os.path.join(tmp, "b"), paths=paths, force=True)
    dt = time.time() - t0
    c1 = PROC.cpu_times()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
cons = sum(add_t) + sum(fwd_t)
print(f"\n入库 {n} 张 | wall {dt:.2f}s → {n / dt:.0f} 张/秒")
print(f"进程 CPU {cpu_s:.1f}s → 平均并行度 {cpu_s / dt:.1f} 核 "
      f"| 折算单张 CPU {cpu_s / max(n, 1) * 1000:.0f} ms")
print(f"\n消费端（主线程）：")
print(f"  粗筛入库 add_results: {sum(add_t):6.2f}s（{add_n[0]} 批，"
      f"每批中位 {statistics.median(add_t) * 1000 if add_t else 0:.1f}ms）")
print(f"  GPU 前向 _forward   : {sum(fwd_t):6.2f}s（{fwd_n[0]} 批，"
      f"每批中位 {statistics.median(fwd_t) * 1000 if fwd_t else 0:.1f}ms）")
print(f"  消费端合计         : {cons:6.2f}s（占 wall {cons / dt * 100:.0f}%）")
print(f"  等批次数（生产端慢）: {max(0.0, dt - cons):6.2f}s"
      f"（占 wall {max(0.0, dt - cons) / dt * 100:.0f}%）")
print(f"\n判定: " + ("消费端是瓶颈（主线程忙不过来）" if cons > dt * 0.6
                   else "生产端（解码）是瓶颈，消费端在等批次"))
