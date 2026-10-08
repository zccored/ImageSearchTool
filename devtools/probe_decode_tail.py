# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 单张解码耗时分布探针（长尾假说验证）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""验证"解码池打不满是由长尾（大图拖尾）造成"的假说。

不改主程序逻辑：用与建库同规模的线程池（默认 18 路，可调）跑同一批真实图，
逐张记录解码耗时，输出整体与分档的 p50/p95/p99、极差、以及"尾部单张 vs 中位"的倍数。
判据（用户提出）：**p99/p50 > 5 → 长尾显著**，此时应给大图开专用低并发通道，
而不是继续在 cv2 线程数上抠 1~2 个核。

用法: python -E devtools/probe_decode_tail.py [样本数=540] [线程数=18] [格式集=all|png|jpg]
"""
import json
import os
import random
import statistics as st
import struct
import sys
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 540
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 18
KIND = sys.argv[3] if len(sys.argv) > 3 else "all"


def ihdr(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25]


def main() -> int:
    from hybrid_search.io_utils import collect_images, decode_rgb

    paths = collect_images(GALLERY_ROOT, EXTS)
    if KIND == "png":
        paths = [p for p in paths if p.lower().endswith(".png")]
    elif KIND == "jpg":
        paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    random.seed(11)
    random.shuffle(paths)
    sample = paths[:N]
    print("样本 %d 张（%s），线程池 %d 路；图库根 %s" % (len(sample), KIND, WORKERS, GALLERY_ROOT))

    lock = threading.Lock()
    rows = []
    idx = [0]

    def work():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(sample):
                return
            p = sample[i]
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            t0 = time.perf_counter()
            arr = decode_rgb(data)
            dt = (time.perf_counter() - t0) * 1e3
            if arr is None:
                continue
            h, w = arr.shape[0], arr.shape[1]
            ext = os.path.splitext(p)[1].lower()
            bucket = ("0-1MP" if w * h < 1e6 else "1-4MP" if w * h < 4e6
                      else "4-12MP" if w * h < 12e6 else "12+MP")
            with lock:
                rows.append({"ms": dt, "px": w * h, "ext": ext, "bucket": bucket,
                             "name": os.path.basename(p)})

    ths = [threading.Thread(target=work, daemon=True) for _ in range(WORKERS)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0

    def pct(v, q):
        if not v:
            return 0.0
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))]

    def brief(tag, v):
        if not v:
            return None
        p50, p95, p99 = pct(v, .5), pct(v, .95), pct(v, .99)
        return {"n": len(v), "p50": round(p50, 1), "p95": round(p95, 1),
                "p99": round(p99, 1), "max": round(max(v), 1),
                "mean": round(st.mean(v), 1), "p99_over_p50": round(p99 / max(p50, 1e-9), 2)}

    all_ms = [r["ms"] for r in rows]
    out = {"overall": brief("all", all_ms), "wall_s": round(wall, 2),
           "imgs_per_s": round(len(rows) / wall, 1),
           "workers": WORKERS, "by_bucket": {}, "by_ext": {}}
    for key, grp in (("bucket", "bucket"), ("ext", "ext")):
        d = defaultdict(list)
        for r in rows:
            d[r[grp]].append(r["ms"])
        for k in sorted(d):
            out["by_bucket" if key == "bucket" else "by_ext"][k] = brief(k, d[k])
    print("\n== 整体：%d 张 / %.2f s（%.1f 张/s，%d 路池）" %
          (len(rows), wall, len(rows) / wall, WORKERS))
    o = out["overall"]
    print("   p50 %.1f ms | p95 %.1f | p99 %.1f | max %.1f | mean %.1f | **p99/p50 = %.2f**"
          % (o["p50"], o["p95"], o["p99"], o["max"], o["mean"], o["p99_over_p50"]))
    print("== 分档（**观察长尾**）")
    for k in sorted(out["by_bucket"]):
        b = out["by_bucket"][k]
        print("   %-8s n=%4d p50 %7.1f p95 %7.1f p99 %7.1f max %8.1f  p99/p50 %5.2f"
              % (k, b["n"], b["p50"], b["p95"], b["p99"], b["max"], b["p99_over_p50"]))
    print("== 按格式")
    for k in sorted(out["by_ext"]):
        b = out["by_ext"][k]
        print("   %-8s n=%4d p50 %7.1f p95 %7.1f p99 %7.1f  p99/p50 %5.2f"
              % (k, b["n"], b["p50"], b["p95"], b["p99"], b["p99_over_p50"]))
    print("== 最慢 8 张")
    for r in sorted(rows, key=lambda x: -x["ms"])[:8]:
        print("   %8.1f ms  %-9s %6.2f MP  %s" % (r["ms"], r["bucket"], r["px"] / 1e6, r["name"]))
    # 长尾对总时间的贡献：最慢 5% 占了多少
    v = sorted(all_ms)
    top5 = v[int(len(v) * .95):]
    print("== 最慢 5%% 的 %d 张占总解码时间的 %.1f%%（若 >25%% 说明长尾显著）"
          % (len(top5), 100.0 * sum(top5) / max(sum(v), 1e-9)))
    os.makedirs(os.path.join(_HERE, "perf_reports"), exist_ok=True)
    jf = os.path.join(_HERE, "perf_reports",
                      "decode_tail_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(jf, "w", encoding="utf-8") as f:
        json.dump({"ts": time.strftime("%Y%m%d-%H%M%S"), "kind": KIND, "workers": WORKERS,
                   "result": out, "top5_pct_time": round(100.0 * sum(top5) / max(sum(v), 1e-9), 1),
                   "rows": sorted(rows, key=lambda x: -x["ms"])[:200]}, f,
                  ensure_ascii=False, indent=2)
    print("JSON:", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
