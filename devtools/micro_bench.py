# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool — P0/P1 微基准（解码层与预处理层的交替配对测量）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""解码/预处理的**交替配对微基准**：同一张图紧邻跑 A、B 两个实现，取配对比值中位。

为什么这么做：单次"先跑 A 再跑 B"会被 CPU 时钟漂移、邻居负载、页缓存状态污染
（实测同一实现两次测量可差 50%）。交替 + 配对能把这些系统误差消掉。

用法:
  python -E devtools/micro_bench.py png-rgb       # P0-b：cv2 换通道 vs RGB 直出
  python -E devtools/micro_bench.py decode         # 各格式解码耗时（现状实现）
  python -E devtools/micro_bench.py md5-tile       # P0-a：逐块哈希 vs 复用哈希对象
"""
import os
import random
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

GALLERY_INDEX = os.path.join(GALLERY_ROOT, ".gallery_index")


def load_blobs(kind="png", n=40, seed=7):
    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                    allow_pickle=True)]
    if kind == "png":
        pool = [p for p in paths if p.lower().endswith(".png")]
    elif kind == "jpeg":
        pool = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    else:
        pool = paths
    random.seed(seed)
    blobs = []
    for p in random.sample(pool, min(n, len(pool))):
        try:
            blobs.append(open(p, "rb").read())
        except OSError:
            pass
    return blobs


def bench_png_rgb(rounds=3):
    from hybrid_search import io_utils as iu
    iu.silence_png_noise(True)
    blobs = load_blobs("png", 40)
    print("样本 %d 张 PNG，均 %.2f MB；%d 轮交替配对"
          % (len(blobs), sum(map(len, blobs)) / len(blobs) / 2 ** 20, rounds))
    t_off = t_on = 0.0
    ratios = []
    for rnd in range(rounds):
        for i, b in enumerate(blobs):
            order = (False, True) if (i + rnd) % 2 == 0 else (True, False)
            ts = {}
            for flag in order:
                iu.set_cv2_rgb_direct(flag)
                t0 = time.perf_counter()
                iu.decode_rgb(b)
                ts[flag] = time.perf_counter() - t0
            t_off += ts[False]
            t_on += ts[True]
            ratios.append(ts[True] / ts[False])
    iu.set_cv2_rgb_direct(True)
    med = statistics.median(ratios)
    print("  旧(BGR+换通道): %.1f ms/张   新(RGB 直出): %.1f ms/张"
          % (t_off / len(ratios) * 1000, t_on / len(ratios) * 1000))
    print("  配对中位比值 %.4f → 解码时间 %+.1f%%；均值 %.4f（样本 %d）"
          % (med, 100 * (med - 1), statistics.mean(ratios), len(ratios)))


def bench_decode():
    from hybrid_search import io_utils as iu
    iu.silence_png_noise(True)
    for kind in ("png", "jpeg"):
        blobs = load_blobs(kind, 40)
        t0 = time.perf_counter()
        for b in blobs:
            iu.decode_rgb(b)
        el = time.perf_counter() - t0
        mb = sum(map(len, blobs)) / 2 ** 20
        print("  %-5s %2d 张 / %.1f MB：总 %.2f s，单张中位 %.1f ms，%.0f MB/s(压缩流)"
              % (kind, len(blobs), mb, el, el / len(blobs) * 1000, mb / el))


def bench_md5_tile(rounds=5):
    import hashlib
    from hybrid_search.tile_index import _tile_md5, _tile_md5_base, _tile_md5_of
    blobs = load_blobs("png", 20)
    n_tiles = 13
    boxes = [(384 * i % 1500, 0, 384 * i % 1500 + 512, 512) for i in range(n_tiles)]
    t_old = t_new = 0.0
    for _ in range(rounds):
        for b in blobs:
            t0 = time.perf_counter()
            for box in boxes:
                _tile_md5(b, box)
            t_old += time.perf_counter() - t0
            t0 = time.perf_counter()
            base = _tile_md5_base(b)
            for box in boxes:
                _tile_md5_of(base, box)
            t_new += time.perf_counter() - t0
    r = t_new / t_old
    print("  %d 张 × %d 块 × %d 轮" % (len(blobs), n_tiles, rounds))
    print("  逐块整文件哈希: %.3f s   复用哈希对象: %.3f s → %+.1f%%（%.1f× 提速）"
          % (t_old, t_new, 100 * (r - 1), t_old / t_new))


def main() -> int:
    what = (sys.argv[1] if len(sys.argv) > 1 else "png-rgb").lower()
    if what == "png-rgb":
        bench_png_rgb()
    elif what == "decode":
        bench_decode()
    elif what == "md5-tile":
        bench_md5_tile()
    else:
        print("未知：%s（可选 png-rgb / decode / md5-tile）" % what)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
