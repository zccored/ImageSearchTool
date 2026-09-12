# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 探针：GPU 利用率实测与空窗比例/最长连续空窗统计
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""GPU 利用率实测：融合建库期间采样 GPU/CPU，统计空窗比例与最长连续空窗。

用法: python devtools/probe_gpu_util.py [张数] [--tag 名称]
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

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402
import perfwatch  # noqa: E402

ROOT = r"F:\视频"
N = 2000
TAG = ""
if len(sys.argv) > 1:
    N = int(sys.argv[1])
if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]

paths = collect_images(ROOT, Config().extensions, limit=N)
print(f"样本 {len(paths)} 张真实图（{ROOT}）{('标签: ' + TAG) if TAG else ''}")

cfg = Config()
tmp = tempfile.mkdtemp(prefix="gpu_probe_")
prof = perfwatch.StageProfiler("index", f"GPU 利用率实测 {TAG}",
                               cfg=cfg, prefix=os.path.join(tmp, "b"))
prof.start()
try:
    t0 = time.time()
    eng = HybridEngine(cfg)
    eng.build(os.path.join(tmp, "b"), paths=paths, force=True,
              progress=lambda d, t, ph="fused": prof.bump(d, t))
    dt = time.time() - t0
finally:
    prof.mark("建库返回")
    path = prof.stop(note=f"{len(paths)} 张")
    shutil.rmtree(tmp, ignore_errors=True)

rows = prof.rows
gpu = [float(r["gpu"]) for r in rows if r.get("gpu") is not None]
cpu = [float(r["cpu_sys"]) for r in rows if r.get("cpu_sys") is not None]
mem = [float(r["cpu_proc"]) for r in rows if r.get("cpu_proc") is not None]
print(f"\n建库 {len(paths)} 张 用时 {dt:.1f}s → {len(paths) / dt:.0f} 张/秒")
if gpu:
    idle = [g for g in gpu if g < 5]
    busy = [g for g in gpu if g >= 50]
    # 最长连续空窗（采样间隔 0.4s）
    longest = cur = 0
    for g in gpu:
        cur = cur + 1 if g < 5 else 0
        longest = max(longest, cur)
    print(f"GPU%: 均值 {statistics.mean(gpu):.1f} 峰值 {max(gpu):.0f} | "
          f"<5% 占比 {len(idle) / len(gpu) * 100:.0f}% | "
          f">=50% 占比 {len(busy) / len(gpu) * 100:.0f}% | "
          f"最长连续空窗 {longest * 0.4:.1f}s")
    print(f"CPU%: 系统均值 {statistics.mean(cpu):.0f} 峰值 {max(cpu):.0f} | "
          f"进程均值 {statistics.mean(mem):.0f}")
    import numpy as np
    arr = np.array(gpu)
    print("GPU 时间片（每 0.4s，10 个一组的均值）:")
    for i in range(0, len(arr), 10):
        seg = arr[i:i + 10]
        bar = "█" * int(seg.mean() / 5)
        print(f"  {i * 0.4:6.1f}s {seg.mean():5.1f}% {bar}")
print(f"\n报告: {path}")
