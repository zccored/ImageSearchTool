# -*- coding: utf-8 -*-
"""复核真实索引（侧车格式）：加载耗时/RSS + 三种检索模式。

只读：仅打开与检索，不写图库/索引。
用法: python devtools/check_real_index.py
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search import tile_index as TI  # noqa: E402

ROOT = r"F:\视频"
FP = os.path.join(ROOT, ".gallery_index", "gallery")
TP = os.path.join(ROOT, ".gallery_index", "gallery_tiles")
Q = r"F:\靶子\77C93F54F2277365732D6E39B73878E4.png"
PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 2 ** 20


def open_engine(prefix: str):
    base = rss()
    eng = HybridEngine(Config())
    t0 = time.time()
    eng.open(prefix)
    dt = (time.time() - t0) * 1000
    gc.collect()
    print(f"  open {os.path.basename(prefix)}: 存储={eng._storage} "
          f"n={eng.coarse.size:,} 耗时={dt:.0f}ms "
          f"RSS {base:.0f}->{rss():.0f}MB (+{rss() - base:.0f})")
    return eng


print("== 1) 加载 ==")
eng_t = open_engine(TP)
eng_f = open_engine(FP)

print("== 2) 瓦片检索（靶子 PNG）==")
t0 = time.time()
o = TI.search_tiles_tiled(eng_t, Q, top_k=3, coarse_k=300)
print(f"  {time.time() - t0:.2f}s")
for h in o.hits:
    print(f"  cos={h.fine_score:.4f} box={h.box} {os.path.basename(h.path)}")

print("== 3) 整图检索 ==")
t0 = time.time()
o = eng_f.search(Q, coarse_k=300, top_k=3)
print(f"  {time.time() - t0:.2f}s")
for h in o.hits:
    print(f"  cos={h.fine_score:.4f} {os.path.basename(h.path)}")

print("== 4) 混合检索 ==")
t0 = time.time()
o_f = eng_f.search(Q, coarse_k=300, top_k=20)
o_t = TI.search_tiles_tiled(eng_t, Q, top_k=20, coarse_k=300)
o = TI.merge_hybrid(o_f, o_t, Q, top_k=20)
print(f"  {time.time() - t0:.2f}s")
for h in o.hits[:3]:
    print(f"  cos={h.fine_score:.4f} kind={h.match_kind} "
          f"{os.path.basename(h.path)}")
print("== 完成 ==")
