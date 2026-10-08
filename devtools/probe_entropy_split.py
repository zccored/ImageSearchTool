# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PNG 解码成本拆分探针（DEFLATE vs 其余）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""PNG 解码成本拆分：DEFLATE（Huffman + LZ77 回引用）占多少，滤波/色彩/拷贝占多少。

用途：回答"GPU 熵解码（DietGPU/rANS、Huffman gap-array、cudaCompress、GPUJPEG）
能不能帮到 PNG"——它们最多只能覆盖 DEFLATE 这一段，而 DEFLATE 里还有一半是
**不可并行**的 LZ77 回引用。本探针给出这段的实测占比与速度上限。

用法:
  python -E devtools/probe_entropy_split.py [png张数]
"""
import os
import random
import statistics
import struct
import sys
import time
import zlib

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


def idat_of(data: bytes) -> bytes:
    """把 PNG 的所有 IDAT 块拼起来（就是那个 zlib 流）。"""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return b""
    i, out = 8, []
    while i + 8 <= len(data):
        (ln,) = struct.unpack(">I", data[i:i + 4])
        typ = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + ln]
        if typ == b"IDAT":
            out.append(body)
        elif typ == b"IEND":
            break
        i += 12 + ln
    return b"".join(out)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    import cv2
    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                    allow_pickle=True)]
    pool = [p for p in paths if p.lower().endswith(".png")]
    random.seed(11)
    sample = random.sample(pool, min(n, len(pool)))
    print("样本 %d 张 PNG（随机取自真实索引）" % len(sample))
    rows = []
    for p in sample:
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        idat = idat_of(data)
        if not idat:
            continue
        d = zlib.decompressobj()
        t0 = time.perf_counter()
        raw = d.decompress(idat)
        t_inflate = time.perf_counter() - t0
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR_RGB)
        t_total = time.perf_counter() - t0
        if img is None:
            continue
        rows.append((len(data), len(idat), len(raw), t_inflate, t_total))
    if not rows:
        print("无有效样本")
        return 2
    total_src = sum(r[0] for r in rows)
    total_idat = sum(r[1] for r in rows)
    total_raw = sum(r[2] for r in rows)
    ti = sum(r[3] for r in rows)
    tt = sum(r[4] for r in rows)
    print()
    print("压缩文件合计 %.1f MB；IDAT(zlib流) %.1f MB；解出滤波数据 %.1f MB"
          % (total_src / 2 ** 20, total_idat / 2 ** 20, total_raw / 2 ** 20))
    print("逐张中位：文件 %.2f MB → IDAT %.2f MB → 原始 %.2f MB"
          % (statistics.median(r[0] for r in rows) / 2 ** 20,
             statistics.median(r[1] for r in rows) / 2 ** 20,
             statistics.median(r[2] for r in rows) / 2 ** 20))
    print()
    print("DEFLATE（Huffman + LZ77 回引用）: %.2f s  → 单张中位 %.1f ms  | "
          "%.0f MB/s(解出字节)" % (ti, statistics.median(r[3] for r in rows) * 1000,
                                 total_raw / ti / 2 ** 20))
    print("完整 PNG 解码（cv2）          : %.2f s  → 单张中位 %.1f ms  | "
          "%.0f MB/s(解出字节)" % (tt, statistics.median(r[4] for r in rows) * 1000,
                                 total_raw / tt / 2 ** 20))
    print("滤波重建/色彩转换/拷贝等其余  : %.2f s（占解码 %.0f%%）"
          % (tt - ti, 100 * (tt - ti) / tt))
    print(" → 任何「熵解码加速」最多只能覆盖 %.0f%% 的解码时间；"
          % (100 * ti / tt))
    print("   而其中 LZ77 回引用部分（32KB 滑动窗，逐字节依赖）不可并行，")
    print("   可并行的只有 Huffman 符号解码一段，故真实上限还要再打折。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
