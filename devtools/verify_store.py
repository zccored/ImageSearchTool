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

"""验证侧车(.npy)存储：加载耗时/内存、检索结果一致性、增量、真实索引回归。"""
import gc
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, r"D:\code\新的代码\全栈图库管理器 v3.2bata\image-search")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search import tile_index as TI  # noqa: E402
from hybrid_search.store import IndexFiles, compact  # noqa: E402

PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 2 ** 20


def sig(o):
    return [(os.path.basename(h.path), h.box,
             None if h.fine_score is None else round(float(h.fine_score), 5))
            for h in o.hits]


cfg = Config()
cfg.workers = 6
cfg.decode_workers = 4
cfg.batch = 16

work = tempfile.mkdtemp(prefix="storetest_")
print("工作目录:", work)
try:
    rng = np.random.RandomState(3)
    imgs = []
    for i in range(10):
        base = rng.randint(0, 90, (900, 1200, 3), dtype=np.uint8)
        cv2.rectangle(base, (100 + i * 40, 150), (600 + i * 40, 650),
                      (200, 60 + i * 15, 240), -1)
        cv2.putText(base, f"IMG{i}", (150, 820), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (255, 255, 255), 10)
        p = os.path.join(work, f"img{i}.jpg")
        cv2.imwrite(p, base)
        imgs.append(p)
    # 查询图：img0 的左上 700x600 裁切（应命中 img0 的左上瓦片）
    full = cv2.imread(imgs[0])
    qpng = os.path.join(work, "query.png")
    cv2.imwrite(qpng, full[60:660, 80:780])

    tp = os.path.join(work, "idx", "gallery_tiles")
    print("\n== 1) 用 npz 格式建瓦片索引 ==")
    eng = HybridEngine(cfg)
    t0 = time.time()
    n = TI.build_tiles(eng, tp, paths=imgs)
    print(f"  建库 {time.time() - t0:.1f}s，{n} 瓦片")
    o = TI.search_tiles_tiled(eng, qpng, top_k=5, coarse_k=100)
    res_npz = sig(o)
    print("  npz 检索 top3:", res_npz[:3])
    del eng
    gc.collect()

    print("\n== 2) 旧 npz 加载耗时/内存 ==")
    eng = HybridEngine(cfg)
    t0 = time.time()
    eng.open(tp)
    dt_npz, rss_npz = time.time() - t0, rss()
    o = TI.search_tiles_tiled(eng, qpng, top_k=5, coarse_k=100)
    assert sig(o) == res_npz
    print(f"  open={dt_npz * 1000:.0f}ms  rss={rss_npz:.0f}MB  检索一致=OK")
    del eng
    gc.collect()

    print("\n== 3) compact 转换 ==")
    t0 = time.time()
    r = compact(tp)
    print(f"  {r}  用时 {time.time() - t0:.1f}s")
    side = [f for f in os.listdir(os.path.dirname(tp))
            if f.endswith(".npy")]
    print("  侧车文件:", sorted(side))

    print("\n== 4) 侧车加载耗时/内存 + 结果一致性 ==")
    eng = HybridEngine(cfg)
    t0 = time.time()
    eng.open(tp)
    dt_side, rss_side = time.time() - t0, rss()
    o = TI.search_tiles_tiled(eng, qpng, top_k=5, coarse_k=100)
    res_side = sig(o)
    print(f"  open={dt_side * 1000:.0f}ms  rss={rss_side:.0f}MB")
    print(f"  加速 {dt_npz / max(dt_side, 1e-6):.1f}x  省内存 "
          f"{rss_npz - rss_side:.0f}MB  结果一致={res_side == res_npz}")
    if res_side != res_npz:
        print("   npz :", res_npz)
        print("   side:", res_side)

    print("\n== 5) 侧车索引上增量（框混合路径）==")
    extra = []
    for i in (10, 11):
        base = rng.randint(0, 90, (900, 1200, 3), dtype=np.uint8)
        cv2.putText(base, f"NEW{i}", (150, 820), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (255, 255, 255), 10)
        p = os.path.join(work, f"new{i}.jpg")
        cv2.imwrite(p, base)
        extra.append(p)
    added = TI.add_tiles(eng, tp, paths=imgs + extra)
    print(f"  新增瓦片 {added}（应=2 图 × 6 块=12）")
    o = TI.search_tiles_tiled(eng, qpng, top_k=5, coarse_k=100)
    print("  增量后 top1:", sig(o)[0])
    print("  旧结果仍在前列:", sig(o)[0][0] == res_npz[0][0])
    n_after = eng.coarse.size
    del eng
    gc.collect()
    # 增量写盘后重开：行数必须包含新增（验证侧车/ npz 写入一致）
    eng = HybridEngine(cfg)
    eng.open(tp)
    print(f"  重开：存储={eng._storage} 行数={eng.coarse.size}"
          f"（内存中={n_after}） 一致={eng.coarse.size == n_after}")
    o = TI.search_tiles_tiled(eng, qpng, top_k=5, coarse_k=100)
    print("  重开检索 top1:", sig(o)[0])
    del eng
    gc.collect()

    print("\n== 6) 增量后再次 compact（幂等）==")
    r2 = compact(tp)
    print("  ", r2)
    eng = HybridEngine(cfg)
    eng.open(tp)
    o = TI.search_tiles_tiled(eng, qpng, top_k=5, coarse_k=100)
    print("  侧车重开 top1:", sig(o)[0], " 与增量后一致=",
          sig(o)[0][0] == "img0.jpg")
    del eng
    gc.collect()

    print("\n== 7) 整图索引 npz -> compact -> 检索 ==")
    fp = os.path.join(work, "idx", "gallery")
    eng = HybridEngine(cfg)
    eng.build(fp, paths=imgs, force=True)
    eng.open(fp)
    o1 = eng.search(qpng, coarse_k=50, top_k=5)
    s1 = [(os.path.basename(h.path), round(float(h.fine_score), 5))
          for h in o1.hits]
    print("  npz 检索:", s1[:3])
    del eng
    gc.collect()
    r3 = compact(fp)
    print("  compact:", r3)
    eng = HybridEngine(cfg)
    eng.open(fp)
    o2 = eng.search(qpng, coarse_k=50, top_k=5)
    s2 = [(os.path.basename(h.path), round(float(h.fine_score), 5))
          for h in o2.hits]
    print("  侧车检索:", s2[:3], " 一致=", s1 == s2)
    del eng
    gc.collect()

    print("\n== 8) 真实 444k 瓦片索引（npz，只读）回归 ==")
    eng = HybridEngine(cfg)
    t0 = time.time()
    eng.open(r"F:\视频\.gallery_index\gallery_tiles")
    print(f"  open={time.time() - t0:.2f}s 存储={eng._storage} n={eng.coarse.size}")
    o = TI.search_tiles_tiled(
        eng, r"F:\靶子\77C93F54F2277365732D6E39B73878E4.png",
        top_k=3, coarse_k=300)
    for h in o.hits[:2]:
        print(f"  {os.path.basename(h.path)} cos={h.fine_score:.4f} box={h.box}")
    ok = o.hits and "77C93F54F2277365732D6E39B73878E4.jpg" in o.hits[0].path
    print("  靶子 Top1 正确 =", bool(ok))
finally:
    shutil.rmtree(work, ignore_errors=True)
    print("\n完成（临时目录已清理）")
