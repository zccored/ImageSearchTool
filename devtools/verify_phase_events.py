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

"""验证阶段边界事件：融合建库/瓦片建库的进度回调必须以 save、done 收尾。

用法: python devtools/verify_phase_events.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search import tile_index as TI  # noqa: E402

work = tempfile.mkdtemp(prefix="phasetest_")
ok = True
try:
    paths = []
    rng = np.random.RandomState(0)
    for i in range(12):
        arr = rng.randint(0, 255, (900, 1200, 3), dtype=np.uint8)
        p = os.path.join(work, f"img{i}.jpg")
        Image.fromarray(arr).save(p, quality=90)
        paths.append(p)

    cfg = Config()
    cfg.workers = 4
    cfg.decode_workers = 2
    cfg.batch = 8

    def record():
        seq = []

        def cb(done, total, phase="fused"):
            if not seq or seq[-1][0] != phase:
                seq.append([phase, done, total])
            else:
                seq[-1][1] = done
                seq[-1][2] = total
        return seq, cb

    print("== 1) 融合建库（粗筛+ResNet 单遍）==")
    seq, cb = record()
    eng = HybridEngine(cfg)
    n = eng.build(os.path.join(work, "idx", "gallery"), paths=paths,
                  force=True, progress=cb)
    print(f"  入库 {n} 张；阶段序列: {[(s[0], s[2]) for s in seq]}")
    tail = [s[0] for s in seq][-2:]
    if tail == ["save", "done"]:
        print("  ✓ 以 save→done 收尾（可明确判定结束）")
    else:
        print(f"  ✗ 收尾阶段异常: {tail}")
        ok = False

    print("\n== 2) 瓦片建库 ==")
    seq2, cb2 = record()
    tp = os.path.join(work, "idx", "gallery_tiles")
    eng2 = HybridEngine(cfg)
    tp_n = TI.build_tiles(eng2, tp, paths=paths,
                          progress=lambda d, t, ph="tiles": cb2(d, t, ph))
    print(f"  入库 {tp_n} 块；阶段序列: {[(s[0], s[2]) for s in seq2]}")
    tail2 = [s[0] for s in seq2][-2:]
    if tail2 == ["save", "done"]:
        print("  ✓ 以 save→done 收尾")
    else:
        print(f"  ✗ 收尾阶段异常: {tail2}")
        ok = False

    print("\n== 3) 增量（add）==")
    seq3, cb3 = record()
    eng3 = HybridEngine(cfg)
    added = eng3.add(os.path.join(work, "idx", "gallery"), paths=paths)
    print(f"  重复路径增量: +{added} 张（应为 0，直接返回）")

    print("\n结果:", "全部通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(work, ignore_errors=True)
