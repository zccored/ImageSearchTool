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

"""真实索引基准：整图库(37,683)检索延迟 + 瓦片库(444,235)局部检索延迟。

只读：打开现有索引并检索，不写图库/索引。
用法: python devtools/bench_real_index.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search import tile_index as TI  # noqa: E402

ROOT = r"F:\视频"
FP = os.path.join(ROOT, ".gallery_index", "gallery")
TP = os.path.join(ROOT, ".gallery_index", "gallery_tiles")
CROP = r"F:\靶子\77C93F54F2277365732D6E39B73878E4.png"
cfg = Config()


def bench_fused(n: int):
    """真实图融合建库吞吐（粗筛+ResNet 单遍解码），临时索引测完自删。"""
    import shutil
    import tempfile

    from hybrid_search.io_utils import collect_images

    paths = collect_images(ROOT, cfg.extensions, limit=n)
    tmp = tempfile.mkdtemp(prefix="fused_bench_")
    try:
        eng = HybridEngine(cfg)
        t0 = time.time()
        got = eng.build(os.path.join(tmp, "b"), paths=paths)
        dt = time.time() - t0
        print(f"  融合建库 {got} 张 {dt:.1f}s → {got / max(dt, 1e-6):.0f} 张/秒"
              f"（{cfg.model}，{'GPU' if eng._get_extractor().device == 'cuda' else 'CPU'}）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if "--fused" in sys.argv:
    i = sys.argv.index("--fused")
    bench_fused(int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 2000)
    sys.exit(0)

print("== 1) 整图索引：加载 ==")
eng = HybridEngine(cfg)
t0 = time.time()
eng.open(FP)
print(f"  加载 {time.time() - t0:.2f}s（{eng.coarse.size:,} 张，存储 {eng._storage}）")

print("== 2) 整图索引：检索延迟（5 个真实查询）==")
queries = [p for p in eng.coarse.paths[:5]]
lat = []
for q in queries:
    t0 = time.time()
    o = eng.search(q, coarse_k=cfg.coarse_k, top_k=10)
    dt = time.time() - t0
    lat.append(dt)
    top = os.path.basename(o.hits[0].path) if o.hits else "-"
    print(f"  {dt * 1000:7.1f} ms  top1={top}")
print(f"  中位数 {statistics.median(lat) * 1000:.1f} ms | 均值 "
      f"{statistics.mean(lat) * 1000:.1f} ms（库 {eng.coarse.size:,} 张）")

print("== 3) 瓦片索引：加载 ==")
eng_t = HybridEngine(cfg)
t0 = time.time()
eng_t.open(TP)
print(f"  加载 {time.time() - t0:.2f}s（{eng_t.coarse.size:,} 块，"
      f"存储 {eng_t._storage}）")

print("== 4) 瓦片索引：局部查询（查询侧切块）==")
for q in (CROP, queries[0]):
    t0 = time.time()
    o = TI.search_tiles_tiled(eng_t, q, top_k=10, coarse_k=cfg.coarse_k,
                              method="lsh")
    dt = time.time() - t0
    t = o.times
    print(f"  {os.path.basename(q)[:40]:<40} {dt:5.2f}s  "
          f"（切块 {t.get('查询切块×特征', 0):.2f}s + 聚合 "
          f"{t.get('分块候选聚合', 0):.2f}s + 精排 "
          f"{t.get('收敛精排(全瓦片×查询块)', 0):.2f}s）")
    if o.hits:
        h = o.hits[0]
        print(f"      top1={os.path.basename(h.path)} cos={h.fine_score:.4f} "
              f"box={h.box}")
