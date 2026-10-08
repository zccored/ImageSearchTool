# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — libdeflate PNG 路径一致性验证
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""验证 libdeflate PNG 路径：**(a) 全格式矩阵**逐位对齐 cv2；**(b) 大样本**覆盖与逐位一致。

判定口径：只看 RGB（建库主路径的输入是 RGB；灰度在 libdeflate 模式下按设计回退 cv2）。
回退必须"安全"：回退的文件在 cv2 模式下仍能正常解码，否则管线结果会变。

只读图片，不写索引、不删文件。
用法: python -E devtools/verify_png_fast.py [矩阵扫描候选数=2500] [每格式上限=40] [大样本数=800]
"""
import json
import os
import random
import struct
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

COLOR_TYPES = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+alpha", 6: "RGBA"}
OUT_DIR = os.path.join(_HERE, "perf_reports")


def ihdr_of(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25], data[28]


def reason_delta(before, after):
    for k, v in after.items():
        if k.startswith("fallback_") and v > before.get(k, 0):
            return k[len("fallback_"):]
    return "unknown"


def main() -> int:
    scan = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    per_fmt = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    big_n = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    from hybrid_search import png_fast
    from hybrid_search.io_utils import collect_images

    print(png_fast.describe())
    paths = collect_images(GALLERY_ROOT, [".png"])
    pngs = [p for p in paths if p.lower().endswith(".png")]
    random.seed(7)
    random.shuffle(pngs)

    # ---------------- (a) 全格式矩阵 ----------------
    buckets = defaultdict(list)
    for p in pngs[:scan]:
        try:
            with open(p, "rb") as f:
                info = ihdr_of(f.read(33))
        except OSError:
            continue
        if not info:
            buckets["坏头"].append(p)
            continue
        _, _, depth, ctype, inter = info
        key = "%s %dbit%s" % (COLOR_TYPES.get(ctype, "ct%d" % ctype), depth,
                              " 隔行" if inter else "")
        if len(buckets[key]) < per_fmt:
            buckets[key].append(p)
    print("\n=== (a) 格式矩阵（扫描 %d 张）" % scan)
    matrix = {}
    bad_all = []
    for k in sorted(buckets):
        n_ok = n_handled_bad = n_fb_ok = n_fb_broken = n_both_none = 0
        reasons = Counter()
        for p in buckets[k]:
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            before = png_fast.stats()
            a = png_fast.decode_rgb(data)
            after = png_fast.stats()
            b = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR_RGB)
            if a is not None:
                if b is not None and a.shape == b.shape and np.array_equal(a, b):
                    n_ok += 1
                else:
                    n_handled_bad += 1
                    d = -1 if (b is None or a.shape != b.shape) else int(
                        np.abs(a.astype(int) - b.astype(int)).max())
                    bad_all.append((k, p, d))
            else:
                reasons[reason_delta(before, after)] += 1
                if b is None:
                    n_both_none += 1
                else:
                    n_fb_ok += 1
            del data
        matrix[k] = {"files": len(buckets[k]), "ok": n_ok, "mismatch": n_handled_bad,
                     "fallback_cv2_ok": n_fb_ok, "fallback_cv2_none": n_both_none,
                     "reasons": dict(reasons)}
        print("   %-18s 走新路径且逐位一致 %3d | 走新路径但不一致 %d | 回退(其余正常) %3d "
              "| 回退且两边都失败 %d | 原因 %s"
              % (k, n_ok, n_handled_bad, n_fb_ok, n_both_none,
                 ",".join("%s×%d" % kv for kv in reasons.most_common()) or "-"))

    # ---------------- (b) 大样本（经 io_utils 管线口径） ----------------
    print("\n=== (b) 大样本 %d 张（走 io_utils.decode_rgb，cv2 vs libdeflate）" % big_n)
    from hybrid_search import io_utils as iu
    sample = pngs[:big_n]
    n_ok = n_bad = n_none = 0
    px_ok = px_total = 0
    t_cv2 = t_ldf = 0.0
    bad_samples = []
    png_fast.reset_stats()
    for p in sample:
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        iu.set_png_decoder("cv2", silence_noise=True)
        t0 = time.perf_counter()
        a = iu.decode_rgb(data)
        t_cv2 += time.perf_counter() - t0
        iu.set_png_decoder("libdeflate", silence_noise=False)
        t0 = time.perf_counter()
        b = iu.decode_rgb(data)
        t_ldf += time.perf_counter() - t0
        if a is None or b is None:
            n_none += 1
            continue
        if a.shape == b.shape and np.array_equal(a, b):
            n_ok += 1
            px_ok += int(a.shape[0]) * int(a.shape[1])
        else:
            n_bad += 1
            bad_samples.append(os.path.basename(p))
        px_total += int(a.shape[0]) * int(a.shape[1])
        del data
    iu.set_png_decoder("cv2", silence_noise=True)
    st = png_fast.stats()
    handled = st.get("ok", 0)
    print("   逐位一致 %d / 不一致 %d / 双边空 %d（样本 %d）"
          % (n_ok, n_bad, n_none, len(sample)))
    print("   新路径处理了 %d 张（%.1f%%），按像素计 %.1f%%；回退原因：%s"
          % (handled, 100.0 * handled / max(len(sample), 1),
             100.0 * px_ok / max(px_total, 1),
             ", ".join("%s=%d" % (k, v) for k, v in sorted(st.items())
                       if k.startswith("fallback_")) or "-"))
    print("   累计计时：cv2 %.2f s vs libdeflate %.2f s → %.2fx"
          % (t_cv2, t_ldf, (t_cv2 / t_ldf) if t_ldf else 0))
    if bad_samples:
        print("   不一致样例：", bad_samples[:8])

    verdict = "可安全切换（逐位一致；回退文件在 cv2 下正常）" if not bad_all and not n_bad \
        else "存在位差 —— 不可切换"
    print("\n结论：%s" % verdict)
    meta = {"tool": "verify-png-fast", "ts": time.strftime("%Y%m%d-%H%M%S"),
            "matrix": matrix, "matrix_bad": bad_all[:20],
            "big_sample": {"n": len(sample), "ok": n_ok, "bad": n_bad, "none": n_none,
                           "handled": handled,
                           "handled_pct": round(100.0 * handled / max(len(sample), 1), 2),
                           "pixel_pct": round(100.0 * px_ok / max(px_total, 1), 2),
                           "t_cv2": round(t_cv2, 3), "t_libdeflate": round(t_ldf, 3),
                           "speedup": round((t_cv2 / t_ldf) if t_ldf else 0, 2)},
            "fallback_reasons": {k: v for k, v in sorted(st.items())
                                 if k.startswith("fallback_")},
            "verdict": verdict, "deps": png_fast.describe()}
    os.makedirs(OUT_DIR, exist_ok=True)
    f = os.path.join(OUT_DIR, "verify_png_fast_%s.json" % meta["ts"])
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("JSON:", f)
    return 0 if (not bad_all and not n_bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
