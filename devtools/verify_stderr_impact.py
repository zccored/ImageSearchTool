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

"""对照：stderr 噪音过滤器开/关对融合建库吞吐的影响（判断是否反压阻塞）。

用法（两条独立进程各跑一次）:
  python devtools/verify_stderr_impact.py 600 on
  python devtools/verify_stderr_impact.py 600 off
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

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
MODE = sys.argv[2] if len(sys.argv) > 2 else "on"

cfg = Config()
cfg.silence_png_warnings = (MODE == "on")
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)
print(f"[{MODE}] 样本 {len(paths)} 张，silence_png_warnings={cfg.silence_png_warnings}")

tmp = tempfile.mkdtemp(prefix="impact_")
try:
    eng = HybridEngine(cfg)
    t0 = time.time()
    n = eng.build(os.path.join(tmp, "b"), paths=paths, force=True)
    dt = time.time() - t0
    print(f"[{MODE}] {n} 张 {dt:.1f}s → {n / dt:.0f} 张/秒")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
