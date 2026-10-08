# -*- coding: utf-8 -*-
"""PNG 行滤波类型分布探针：决定"能否用 numpy/Cython 自建 libdeflate 解码路径"。

PNG 解码 = inflate + 逐行反滤波（filter 0..4）。inflate 换成 libdeflate 后，反滤波若留在
Python/numpy 层：filter 0/1/2 可向量化（cumsum），filter 3/4（Average/Paeth）**行内串行**，
只能靠 C 层循环。所以分布决定自建路径的难度与收益上限。

用法: python -E devtools/probe_png_filters.py [每档张数=8] [单张上限MB=256]
"""
import os
import random
import struct
import sys
import zlib
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

EXTS = [".png"]
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAX_RAW_MB = int(sys.argv[2]) if len(sys.argv) > 2 else 256
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}
BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
NAMES = {0: "None", 1: "Sub", 2: "Up", 3: "Average", 4: "Paeth"}


def ihdr(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25], data[28]


def idat_of(data):
    i, out = 8, []
    while i + 8 <= len(data):
        (ln,) = struct.unpack(">I", data[i:i + 4])
        typ = data[i + 4:i + 8]
        if typ == b"IDAT":
            out.append(data[i + 8:i + 8 + ln])
        elif typ == b"IEND":
            break
        i += 12 + ln
    return b"".join(out)


def main() -> int:
    from hybrid_search.io_utils import collect_images
    pngs = collect_images(GALLERY_ROOT, EXTS)
    random.seed(17)
    random.shuffle(pngs)
    buckets = defaultdict(list)
    for p in pngs:
        try:
            with open(p, "rb") as f:
                info = ihdr(f.read(33))
        except OSError:
            continue
        if not info or info[2] != 8 or info[4] != 0:      # 只看 8bit 非交错
            continue
        mp = info[0] * info[1] / 1e6
        bn = "0-1MP" if mp < 1 else "1-4MP" if mp < 4 else "4-12MP" if mp < 12 else "12+MP"
        k = "%s/%s" % (CT.get(info[3], "?"), bn)
        if len(buckets[k]) < N_EACH:
            buckets[k].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 8:
            break

    rows = defaultdict(int)
    by_bucket = defaultdict(lambda: defaultdict(int))
    files = 0
    bytes_by_type = defaultdict(int)
    for k in sorted(buckets):
        for p in buckets[k]:
            data = open(p, "rb").read()
            info = ihdr(data)
            w, h, depth, ctype = info[0], info[1], info[2], info[3]
            bpp = max(1, BPP.get(ctype, 4) * depth // 8)
            stride = w * bpp
            if (stride + 1) * h > MAX_RAW_MB * 2 ** 20:
                continue
            try:
                payload = zlib.decompress(idat_of(data))
            except Exception:                             # noqa: BLE001
                continue
            files += 1
            for y in range(h):
                off = y * (stride + 1)
                ft = payload[off]
                rows[ft] += 1
                by_bucket[k][ft] += 1
                bytes_by_type[ft] += stride
            del payload

    total = sum(rows.values()) or 1
    print("样本 %d 张，%d 行，%.2f GB 扫描线数据" % (files, total,
                                                 sum(bytes_by_type.values()) / 2 ** 30))
    print("滤波类型分布（按行 / 按字节）：")
    for ft in sorted(rows, key=lambda x: -rows[x]):
        print("  %-8s(%d): %6.2f%% 行  %6.2f%% 字节   %s"
              % (NAMES.get(ft, "?"), ft, 100.0 * rows[ft] / total,
                 100.0 * bytes_by_type[ft] / max(sum(bytes_by_type.values()), 1),
                 "可向量化(cumsum)" if ft in (0, 1, 2) else "行内串行(需 C 层)"))
    print("\n分档（每档行占比，None/Sub/Up 可向量化 vs Average/Paeth 串行）：")
    for k in sorted(by_bucket):
        tot = sum(by_bucket[k].values()) or 1
        vec = sum(v for f, v in by_bucket[k].items() if f in (0, 1, 2))
        print("  %-18s 行数 %7d  可向量化 %5.1f%%  串行 %5.1f%%"
              % (k, tot, 100.0 * vec / tot, 100.0 * (tot - vec) / tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
