# -*- coding: utf-8 -*-
"""把图库里"吃 DEFLATE 的图"做预分类：PNG（zlib/DEFLATE）按位深/颜色类型/像素档分组。

用于给解码器对比加权——不同档的 DEFLATE 行为差别很大（压缩率、子流结构、解出字节量）。
只读：只读文件头 + stat，不写不删。
用法: python -E devtools/classify_deflate_targets.py [全库扫描上限]
"""
import os
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search.io_utils import collect_images  # noqa: E402
from paths import GALLERY_ROOT  # noqa: E402

ROOT = GALLERY_ROOT
EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}


def ihdr(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25], data[28]


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    paths = collect_images(ROOT, EXTS)
    if limit:
        paths = paths[:limit]
    mp_bins = [(0, 1), (1, 4), (4, 12), (12, 1e9)]
    stats = defaultdict(lambda: [0, 0, 0.0])     # key -> [张数, 像素总数, 文件字节]
    fmt_count = defaultdict(int)
    png_total = png_bytes = png_mp = 0.0
    jpg_total = jpg_bytes = 0.0
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        try:
            nb = os.path.getsize(p)
        except OSError:
            continue
        if ext != ".png":
            fmt_count[ext] += 1
            jpg_total += 1
            jpg_bytes += nb
            continue
        try:
            with open(p, "rb") as f:
                head = f.read(33)
        except OSError:
            continue
        info = ihdr(head)
        if not info:
            fmt_count[".png(坏头)"] += 1
            continue
        w, h, depth, ctype, inter = info
        mp = w * h / 1e6
        png_total += 1
        png_bytes += nb
        png_mp += mp
        for lo, hi in mp_bins:
            binname = "%g-%g MP" % (lo, hi if hi < 1e8 else float("inf"))
            if lo <= mp < hi:
                break
        key = "%-6s %2dbit%-4s %-8s" % (CT.get(ctype, "ct%d" % ctype), depth,
                                       "隔行" if inter else "", binname)
        s = stats[key]
        s[0] += 1
        s[1] += mp
        s[2] += nb
    print("图库 %s：共 %d 张" % (ROOT, len(paths)))
    print("  PNG（吃 DEFLATE）: %d 张 / %.1f GB / %.0f MP"
          % (png_total, png_bytes / 2 ** 30, png_mp))
    print("  其余格式        : %d 张 / %.1f GB（JPEG 等，不涉及 DEFLATE）"
          % (jpg_total, jpg_bytes / 2 ** 30))
    print()
    print("PNG 预分类（按 位深×颜色类型×隔行×像素档）：")
    print("  %-34s %6s %10s %10s %9s" % ("分类", "张数", "占比", "总MB", "平均MP"))
    for k in sorted(stats, key=lambda x: -stats[x][2]):
        n, mp, nb = stats[k]
        print("  %-34s %6d %9.1f%% %10.1f %9.2f"
              % (k, n, 100 * n / max(png_total, 1), nb / 2 ** 20, mp / max(n, 1)))
    print()
    print("其它格式分布:", dict(sorted(fmt_count.items(), key=lambda kv: -kv[1])[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
