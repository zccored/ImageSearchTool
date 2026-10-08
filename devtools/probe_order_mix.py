# -*- coding: utf-8 -*-
"""按建库顺序（与 GUI 扫描同序）统计图片构成，用于判断"吞吐下滑是否由内容变重导致"。

只 stat + 读文件头（不动文件、不删任何东西）。
用法: python -E devtools/probe_order_mix.py [每桶张数] [文件头抽样步长]
"""
import os
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image  # noqa: E402

from hybrid_search.io_utils import collect_images  # noqa: E402
from paths import GALLERY_ROOT  # noqa: E402

ROOT = GALLERY_ROOT
EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]
BUCKET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 4      # 每 STEP 张读一次文件头

paths = collect_images(ROOT, EXTS)
print("图库 %s：%d 张（建库顺序）" % (ROOT, len(paths)))
print("每桶 %d 张；文件头抽样步长 %d" % (BUCKET, STEP))
print()
print(" 桶   张数区间       格式(PNG%%)  平均MB  中位MB  >4MP%%  >12MP%%  平均MP   估算解码ms/张")
tot_ms = 0.0
rows = []
for i in range(0, len(paths), BUCKET):
    chunk = paths[i:i + BUCKET]
    sizes = []
    fmts = Counter()
    mps = []
    for j, p in enumerate(chunk):
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            sizes.append(0)
        fmts[os.path.splitext(p)[1].lower()] += 1
        if j % STEP == 0:
            try:
                with Image.open(p) as im:
                    mps.append(im.size[0] * im.size[1] / 1e6)
            except Exception:                 # noqa: BLE001
                pass
    n = len(chunk)
    png_share = 100 * (fmts[".png"]) / max(n, 1)
    avg_mp = st.mean(mps) if mps else 0.0
    gt4 = 100 * sum(1 for v in mps if v > 4) / max(len(mps), 1)
    gt12 = 100 * sum(1 for v in mps if v > 12) / max(len(mps), 1)
    # 粗略解码成本模型（基于本机实测）：PNG≈26 ms/MP，JPEG≈16 ms/MP 且长边>5120 按 1/4 计像素
    est = 0.0
    for j, p in enumerate(chunk):
        if j % STEP:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:                     # noqa: BLE001
            continue
        mp = w * h / 1e6
        ext = os.path.splitext(p)[1].lower()
        if ext == ".png":
            est += 26.0 * mp
        else:
            if max(w, h) > 5120:
                mp /= 4.0
            est += 16.0 * mp
    est = est / max(1, len(mps)) if mps else 0.0
    tot_ms += est * n
    rows.append((i + 1, i + n, png_share, st.mean(sizes) / 2 ** 20,
                 st.median(sizes) / 2 ** 20, gt4, gt12, avg_mp, est))
    print(" %3d  %6d-%-6d  %5.1f%%  %6.2f %6.2f  %5.1f  %6.1f  %6.2f  %7.1f"
          % (len(rows), i + 1, i + n, png_share, st.mean(sizes) / 2 ** 20,
             st.median(sizes) / 2 ** 20, gt4, gt12, avg_mp, est))

print()
print("全库加权平均估算解码：%.1f ms/张 → 39854 张合计约 %.0f 核·秒（单线程口径）"
      % (tot_ms / len(paths), tot_ms / 1000))
