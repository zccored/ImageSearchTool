# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 交叉引用方案决策实验
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""三条路径对照（决策实验）：
  A 现状：cv2.imdecode（libpng：inflate + SIMD 反滤波 + 色彩）
  B 自建：libdeflate inflate + 自写 SIMD 反滤波（hybrid_search/png_fast.decode_rgb）
  C 交叉：libdeflate inflate + stored(level 0) 重新封装 PNG + cv2.imdecode
          （反滤波交回 libpng；覆盖全部格式；与 A 天然逐位一致）

判据：**(1) 三条路径输出逐位一致；(2) C 是否比 A/B 快**。
用法: python -E devtools/probe_crossref.py [每档张数=6] [轮数=3] [单张上限MB=256]
"""
import json
import os
import random
import struct
import sys
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

N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_RAW_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 256
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}


def ihdr(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25], data[28]


def main() -> int:
    from hybrid_search import png_fast
    from hybrid_search.io_utils import collect_images
    paths = [p for p in collect_images(GALLERY_ROOT, [".png"]) if p.lower().endswith(".png")]
    random.seed(23)
    random.shuffle(paths)
    buckets = defaultdict(list)
    for p in paths:
        try:
            with open(p, "rb") as f:
                info = ihdr(f.read(33))
        except OSError:
            continue
        if not info:
            continue
        w, h, depth, ctype, inter = info
        key = "%s %dbit%s" % (CT.get(ctype, "ct%d" % ctype), depth, " 隔行" if inter else "")
        if len(buckets[key]) < N_EACH:
            buckets[key].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 10:
            break
    items = []
    for k in sorted(buckets):
        for p in buckets[k]:
            try:
                d = open(p, "rb").read()
            except OSError:
                continue
            info = ihdr(d)
            if not info:
                continue
            items.append({"path": p, "bucket": k, "data": d, "px": info[0] * info[1]})
    print("样本 %d 张；%s" % (len(items), png_fast.describe()))
    print("档位：", {k: len(v) for k, v in sorted(buckets.items()) if v})

    # 正确性：三条路径两两逐位比较（C 产出的 PNG 由 cv2 解）
    same = {"A=B": 0, "A=C": 0, "B=C": 0}
    cmp_n = {"A=B": 0, "A=C": 0, "B=C": 0}
    bad = []
    n_c = 0
    for it in items:
        d = it["data"]
        a = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR_RGB)
        b = png_fast.decode_rgb(d)
        synth = png_fast.compat_rgb(d)
        c = cv2.imdecode(np.frombuffer(synth, np.uint8), cv2.IMREAD_COLOR_RGB) \
            if synth else None
        if c is not None:
            n_c += 1
        if a is None or c is None:
            bad.append((os.path.basename(it["path"]), a is None, b is None, c is None))
        # 逐对比较：各自只要求"该对两边都成功"（B 只覆盖 RGB/RGBA，不能拖累 A=C 的判定）
        for pair, x, y in (("A=B", a, b), ("A=C", a, c), ("B=C", b, c)):
            if x is None or y is None:
                continue
            cmp_n[pair] += 1
            if x.shape == y.shape and np.array_equal(x, y):
                same[pair] += 1
    print("逐位一致：A=B %d/%d，A=C %d/%d，B=C %d/%d（C 成功产出 %d 张）"
          % (same["A=B"], cmp_n["A=B"], same["A=C"], cmp_n["A=C"],
             same["B=C"], cmp_n["B=C"], n_c))
    if bad:
        print("（A 或 C 失败的样例，前 5）：", bad[:5])

    # 计时：每条路径独立整轮遍历，轮间交替顺序
    t = defaultdict(float)
    n = defaultdict(int)
    for rnd in range(ROUNDS):
        order = ["A", "B", "C"] if rnd % 2 == 0 else ["C", "B", "A"]
        for name in order:
            for it in items:
                d = it["data"]
                if name == "A":
                    t0 = time.perf_counter()
                    r = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR_RGB)
                    dt = time.perf_counter() - t0
                elif name == "B":
                    t0 = time.perf_counter()
                    r = png_fast.decode_rgb(d)
                    dt = time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    s = png_fast.compat_rgb(d)
                    r = cv2.imdecode(np.frombuffer(s, np.uint8), cv2.IMREAD_COLOR_RGB) if s else None
                    dt = time.perf_counter() - t0
                if r is None:
                    continue
                t[name] += dt / ROUNDS
                n[name] += 1
    tot_px = sum(it["px"] for it in items)
    print("\n=== 计时（%d 轮，每轮全样本）" % ROUNDS)
    for name in ("A", "B", "C"):
        if not t[name]:
            continue
        print("  %s: %.2f s 合计 | %.2f ms/张 | %.1f MP/s"
              % ({"A": "A 现状 cv2", "B": "B 自建 libdeflate+自写反滤波",
                  "C": "C 交叉 libdeflate+stored+cv2"}[name],
                 t[name], t[name] / max(n[name], 1) * 1e3,
                 tot_px / t[name] / 1e6 if t[name] else 0))
    if t["A"] and t["B"] and t["C"]:
        print("  加速比：B/A = %.2fx ；C/A = %.2fx ；C/B = %.2fx"
              % (t["A"] / t["B"], t["A"] / t["C"], t["B"] / t["C"]))
    out = {"files": len(items), "rounds": ROUNDS, "same": same,
           "seconds": {k: round(v, 3) for k, v in t.items()},
           "counts": dict(n), "bad": bad[:10],
           "speedup": {"B_over_A": round(t["A"] / t["B"], 3) if t["B"] else 0,
                       "C_over_A": round(t["A"] / t["C"], 3) if t["C"] else 0,
                       "C_over_B": round(t["B"] / t["C"], 3) if t["C"] else 0}}
    f = os.path.join(_HERE, "perf_reports",
                     "crossref_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("JSON:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
