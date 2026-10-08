# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PNG 反滤波内核单元测试（合成数据）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""原生反滤波内核的**合成数据**单元测试（不依赖任何真实图片）。

思路：按 PNG 规范正向施加 filter 0..4 得到"应该 inflate 出来的扫描线"，再喂给内核，
要求解出的像素与原始像素**逐位相同**；同时要求 SIMD 内核与纯标量实现结果一致。
覆盖：RGB/RGBA、多种宽度（含 16/32 字节块的边界与尾部残块）、多行（Up/Paeth 的跨行依赖）、
以及非法滤波类型必须被拒（返回负值，调用方据此回退 cv2）。

用法: python -E devtools/test_png_filters.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "hybrid_search", "native"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import _pngfast                                                    # noqa: E402


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def forward_filter(px: np.ndarray, ft: int) -> np.ndarray:
    """按规范施加滤波：返回 (h, w*ch+1) 的扫描线（第 0 列是滤波类型）。"""
    h, w, ch = px.shape
    f = np.zeros((h, w * ch + 1), np.uint8)
    f[:, 0] = ft
    for y in range(h):
        row = f[y, 1:].reshape(w, ch)
        for x in range(w):
            for k in range(ch):
                a = int(px[y, x - 1, k]) if x > 0 else 0
                b = int(px[y - 1, x, k]) if y > 0 else 0
                c = int(px[y - 1, x - 1, k]) if (x > 0 and y > 0) else 0
                v = int(px[y, x, k])
                if ft == 0:
                    d = v
                elif ft == 1:
                    d = v - a
                elif ft == 2:
                    d = v - b
                elif ft == 3:
                    d = v - ((a + b) >> 1)
                else:
                    d = v - paeth(a, b, c)
                row[x, k] = d & 0xFF
    return f


def main() -> int:
    rng = np.random.RandomState(20260926)
    widths = [1, 2, 3, 4, 5, 7, 8, 11, 15, 16, 17, 19, 31, 32, 33, 47, 64, 65, 100, 127, 128, 129]
    height = 23
    fails = []
    cases = 0
    print("SIMD/指令集位：%d（bit0=SSE2 bit1=SSSE3 bit2=AVX2）" % _pngfast.cpu_features())
    for ch in (3, 4):
        for w in widths:
            px = rng.randint(0, 256, size=(height, w, ch)).astype(np.uint8)
            # 逐行混合滤波类型：更接近真实 PNG（逐行自适应选择）
            mixed = np.zeros((height, w * ch + 1), np.uint8)
            for y in range(height):
                ft = int(rng.randint(0, 5))
                mixed[y] = forward_filter(px, ft)[y]
            for ft in list(range(5)) + ["mixed"]:
                src = forward_filter(px, int(ft)) if ft != "mixed" else mixed
                src = np.ascontiguousarray(src).reshape(-1)   # 内核吃一维扫描线缓冲
                exp = px[:, :, :3]
                out_s = np.zeros((height, w, 3), np.uint8)
                out_c = np.zeros((height, w, 3), np.uint8)
                r1 = _pngfast.unfilter_to_rgb(src, out_s)
                r2 = _pngfast.unfilter_to_rgb_scalar(src, out_c)
                cases += 1
                if r1 != 0 or r2 != 0 or not np.array_equal(out_s, exp) \
                        or not np.array_equal(out_c, exp):
                    fails.append(("ch=%d w=%d ft=%s" % (ch, w, ft), r1, r2,
                                  np.array_equal(out_s, exp), np.array_equal(out_c, exp)))

    # 非法滤波类型必须被拒（5 与 255）
    px = rng.randint(0, 256, size=(4, 8, 4)).astype(np.uint8)
    bad = forward_filter(px, 0).reshape(-1)
    bad[2 * (8 * 4 + 1)] = 5                      # 第 3 行的滤波类型字节
    out = np.zeros((4, 8, 3), np.uint8)
    rc_bad = _pngfast.unfilter_to_rgb(bad, out)
    bad[2 * (8 * 4 + 1)] = 255
    rc_bad2 = _pngfast.unfilter_to_rgb(bad, out)
    # 长度不符必须被拒
    rc_len = _pngfast.unfilter_to_rgb(np.zeros(10, np.uint8), out)
    print("用例 %d 个；失败 %d 个" % (cases, len(fails)))
    for f in fails[:10]:
        print("   FAIL", f)
    print("非法滤波类型返回：%d / %d（应为负值）；长度不符返回 %d（应为负值）"
          % (rc_bad, rc_bad2, rc_len))
    ok = (not fails) and rc_bad < 0 and rc_bad2 < 0 and rc_len < 0
    print("结论：%s" % ("全部逐位一致，SIMD 与标量结果相同" if ok else "存在失败用例 —— 不可用"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
