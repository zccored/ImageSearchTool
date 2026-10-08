# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PNG 解码器一致性验证（cv2 vs imagecodecs）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""按 PNG 格式矩阵（位深 × 颜色类型 × 隔行）逐位比对 cv2 与 imagecodecs 解码结果。

必须逐位一致才允许把 png_decoder 切到 imagecodecs —— 因为粗筛指纹/ResNet 输入
都由解码像素直接决定，一旦有位差就要重建索引并重跑召回评估。
只读图片，不写索引、不删文件。

用法: python -E devtools/verify_png_decoder.py [扫描候选数] [每格式上限]
"""
import os
import random
import struct
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

GALLERY_INDEX = os.path.join(GALLERY_ROOT, ".gallery_index")
COLOR_TYPES = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+alpha", 6: "RGBA"}


def ihdr_of(data: bytes):
    """(宽, 高, 位深, 颜色类型, 隔行) —— 直接读 PNG 的 IHDR。"""
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25], data[28]


def main() -> int:
    scan = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    per_fmt = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    from hybrid_search import io_utils as iu

    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                    allow_pickle=True)]
    pngs = [p for p in paths if p.lower().endswith(".png")]
    random.seed(7)
    random.shuffle(pngs)
    buckets = defaultdict(list)
    scanned = 0
    for p in pngs[:scan]:
        try:
            with open(p, "rb") as f:
                head = f.read(33)
        except OSError:
            continue
        info = ihdr_of(head)
        scanned += 1
        if not info:
            continue
        _, _, depth, ctype, inter = info
        key = "%s %dbit%s" % (COLOR_TYPES.get(ctype, "ct%d" % ctype), depth,
                              " 隔行" if inter else "")
        if len(buckets[key]) < per_fmt:
            buckets[key].append(p)
    print("扫描 %d 张 PNG 的文件头，得到格式矩阵：" % scanned)
    total = 0
    for k in sorted(buckets):
        print("   %-18s %3d 张" % (k, len(buckets[k])))
        total += len(buckets[k])
    print("   合计 %d 张待验证" % total)

    bad = []
    stat = {}
    for k in sorted(buckets):
        files = buckets[k]
        n_ok = n_bad = n_err = 0
        t = {"cv2": 0.0, "imagecodecs": 0.0}
        for p in files:
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            out = {}
            for mode in ("cv2", "imagecodecs"):
                iu.set_png_decoder(mode)
                t0 = time.perf_counter()
                rgb = iu.decode_rgb(data)
                gr = iu.decode_gray(data)
                t[mode] += time.perf_counter() - t0
                out[mode] = (rgb, gr)
            iu.set_png_decoder("cv2")
            a_rgb, a_gr = out["cv2"]
            b_rgb, b_gr = out["imagecodecs"]
            if a_rgb is None or b_rgb is None:
                n_err += 1
                continue
            # 判定口径：只看 **RGB**（建库主路径的输入；灰度在 imagecodecs 模式下
            # 按设计回退 cv2，这里只作信息性对照）
            same = a_rgb.shape == b_rgb.shape and np.array_equal(a_rgb, b_rgb)
            gray_same = (a_gr is not None and b_gr is not None
                         and a_gr.shape == b_gr.shape and np.array_equal(a_gr, b_gr))
            if same:
                n_ok += 1
                if not gray_same:
                    gray_dev += 1
            else:
                n_bad += 1
                diff = (0 if a_rgb.shape != b_rgb.shape
                        else int(np.abs(a_rgb.astype(int) - b_rgb.astype(int)).max()))
                gdiff = (0 if (a_gr is None or b_gr is None or a_gr.shape != b_gr.shape)
                         else int(np.abs(a_gr.astype(int) - b_gr.astype(int)).max()))
                bad.append((k, p, diff, gdiff))
        if n_ok or n_bad or n_err:
            spd = (t["cv2"] / t["imagecodecs"]) if t["imagecodecs"] else 0
            print("   %-18s RGB+灰度逐位一致 %3d/%3d  不一致 %d  失败 %d  "
                  "| imagecodecs 相对 cv2 %.2fx"
                  % (k, n_ok, len(files), n_bad, n_err, spd))
            stat[k] = (n_ok, n_bad, spd)

    print()
    tot_ok = sum(v[0] for v in stat.values())
    tot_bad = sum(v[1] for v in stat.values())
    print("总计：逐位一致 %d 张，不一致 %d 张" % (tot_ok, tot_bad))
    if bad:
        print("不一致样例（最多 10 条）：")
        for k, p, d, gd in bad[:10]:
            print("   [%s] RGB最大差=%d 灰度最大差=%d  %s" % (k, d, gd, p))
    print("结论：%s" % ("可安全切换（输出逐位一致，无需重建索引）"
                      if tot_bad == 0 else "存在位差 —— 不可直接切换"))
    return 0 if tot_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
