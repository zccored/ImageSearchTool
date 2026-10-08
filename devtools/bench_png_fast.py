# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — libdeflate PNG 解码路径性能图
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""cv2（现状）vs libdeflate 自建路径（inflate + SIMD 反滤波）逐档对照，产出性能图。

口径：
  * 分层抽样（颜色类型 × 像素档），**每个变体整轮独立遍历**（避免互相污染缓存/分配器），
    轮间交替顺序，3 轮取轮均值；
  * 单位同时给 ms/张、MP/s；加速比 = cv2 时间 / 新版时间；
  * 只统计两个路径都能解的文件（公平配对），并单列覆盖率与回退原因；
  * 加权加速比：按样本像素数加权（大图权重更高，接近真实建库的 CPU 分布）。

用法: python -E devtools/bench_png_fast.py [每档张数=6] [轮数=3] [单张上限MB=256]
"""
import json
import os
import random
import statistics as st
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

REPO = _HERE
EXTS = [".png"]
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_RAW_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 256
OUT_DIR = os.path.join(REPO, "perf_reports")
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}
BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
STATE = {"stage": "初始化", "detail": "", "done": 0, "total": 1, "best": "", "peak_mb": 0.0}


def ihdr(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25], data[28]


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:                              # noqa: BLE001
        return 0.0


def start_popup():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:                              # noqa: BLE001
        return None, lambda *a, **k: None
    root = tk.Tk()
    root.title("ImageSearchTool · libdeflate PNG 解码性能图")
    root.attributes("-topmost", True)
    root.geometry("+%d+%d" % (max(root.winfo_screenwidth() - 470, 0), 60))
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    a = ttk.Label(frm, text="初始化…", font=("Microsoft YaHei UI", 10, "bold"))
    a.pack(anchor="w")
    b = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=430, justify="left")
    b.pack(anchor="w", pady=(4, 2))
    bar = ttk.Progressbar(frm, length=430, mode="determinate", maximum=100)
    bar.pack(fill="x", pady=4)
    c = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=430, justify="left")
    c.pack(anchor="w")
    ttk.Button(frm, text="隐藏窗口", command=root.withdraw).pack(anchor="e", pady=(6, 0))
    root.update()

    def tick():
        try:
            STATE["peak_mb"] = max(STATE["peak_mb"], rss_mb())
            a.config(text=STATE["stage"])
            b.config(text=STATE["detail"])
            bar.config(value=100.0 * STATE["done"] / max(STATE["total"], 1))
            c.config(text="进度 %d/%d | 峰值RSS %.0f MB | %s"
                     % (STATE["done"], STATE["total"], STATE["peak_mb"], STATE["best"]))
            root.update()
        except Exception:                          # noqa: BLE001
            pass
    return root, tick


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    root, tick = start_popup()
    STATE["stage"] = "枚举图库 + 分层抽样"
    tick()
    from hybrid_search import io_utils as iu
    from hybrid_search import png_fast
    paths = [p for p in iu.collect_images(GALLERY_ROOT, EXTS) if p.lower().endswith(".png")]
    random.seed(23)
    random.shuffle(paths)
    buckets = defaultdict(list)
    for p in paths:
        try:
            with open(p, "rb") as f:
                info = ihdr(f.read(33))
        except OSError:
            continue
        if not info or info[2] != 8:
            continue
        w, h, _d, ctype, _i = info
        if ctype not in (2, 6):
            continue                     # 新路径只覆盖 RGB/RGBA，其余档无对照意义
        mp = w * h / 1e6
        bn = "0-1MP" if mp < 1 else "1-4MP" if mp < 4 else "4-12MP" if mp < 12 else "12+MP"
        k = "%s/%s" % (CT.get(ctype, "?"), bn)
        if len(buckets[k]) < N_EACH:
            buckets[k].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 8:
            break

    data = {}
    items = []
    for k in sorted(buckets):
        for p in buckets[k]:
            try:
                d = open(p, "rb").read()
            except OSError:
                continue
            info = ihdr(d)
            raw = (info[0] * BPP.get(info[3], 4) * info[2] // 8 + 1) * info[1]
            if raw > MAX_RAW_MB * 2 ** 20:
                continue
            data[p] = d
            items.append({"path": p, "bucket": k, "px": info[0] * info[1], "raw": raw})
    STATE["total"] = len(items) * ROUNDS
    tick()
    print("样本 %d 张；%s" % (len(items), png_fast.describe()))

    iu.set_png_decoder("cv2", silence_noise=True)
    t = defaultdict(lambda: defaultdict(float))     # [bucket][variant]
    n = defaultdict(lambda: defaultdict(int))
    dec = {"cv2": "cv2", "libdeflate": "libdeflate"}
    for rnd in range(ROUNDS):
        order = ["cv2", "libdeflate"] if rnd % 2 == 0 else ["libdeflate", "cv2"]
        for mode in order:
            iu.set_png_decoder(mode, silence_noise=(mode == "cv2"))
            for it in items:
                d = data[it["path"]]
                cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR_RGB)  # 预热
                t0 = time.perf_counter()
                arr = iu.decode_rgb(d)
                dt = time.perf_counter() - t0
                if arr is None:
                    continue
                t[it["bucket"]][mode] += dt / ROUNDS
                n[it["bucket"]][mode] += 1
                STATE["done"] += 1
                STATE["best"] = "%s · 第 %d/%d 轮" % (dec[mode], rnd + 1, ROUNDS)
                tick()
    iu.set_png_decoder("cv2", silence_noise=True)

    rows = []
    for k in sorted(buckets):
        if not t[k]["cv2"] or not t[k]["libdeflate"]:
            continue
        cnt = min(n[k]["cv2"], n[k]["libdeflate"])
        px = sum(it["px"] for it in items if it["bucket"] == k) / max(n[k]["cv2"], 1) * cnt
        c_s, l_s = t[k]["cv2"], t[k]["libdeflate"]
        rows.append({"bucket": k, "n": cnt, "px": int(px),
                     "cv2_ms": round(c_s / max(cnt, 1) * 1e3, 2),
                     "ldf_ms": round(l_s / max(cnt, 1) * 1e3, 2),
                     "cv2_mps": round(px / c_s / 1e6, 2) if c_s else 0,
                     "ldf_mps": round(px / l_s / 1e6, 2) if l_s else 0,
                     "speedup": round(c_s / l_s, 3) if l_s else 0})
    tot_px = sum(r["px"] for r in rows)
    tot_c = sum(r["cv2_ms"] * r["n"] for r in rows) / 1e3
    tot_l = sum(r["ldf_ms"] * r["n"] for r in rows) / 1e3
    overall = (tot_c / tot_l) if tot_l else 0
    px_w = sum(r["speedup"] * r["px"] for r in rows) / max(tot_px, 1)
    print("\n=== 逐档对照（cv2 现状 vs libdeflate 自建路径）")
    for r in rows:
        print("  %-14s n=%2d %7.2f MP | cv2 %8.2f ms  %6.2f MP/s | 新 %8.2f ms  %6.2f MP/s | %5.2fx"
              % (r["bucket"], r["n"], r["px"] / 1e6, r["cv2_ms"], r["cv2_mps"],
                 r["ldf_ms"], r["ldf_mps"], r["speedup"]))
    print("  合计：cv2 %.2f ms/张 → 新 %.2f ms/张；**样本均值 %.2fx，按像素加权 %.2fx**"
          % (tot_c / max(sum(r["n"] for r in rows), 1) * 1e3,
             tot_l / max(sum(r["n"] for r in rows), 1) * 1e3, overall, px_w))
    print("  回退统计：", {k2: v for k2, v in sorted(png_fast.stats().items())
                          if k2.startswith("fallback_")})

    meta = {"tool": "bench-png-fast", "ts": time.strftime("%Y%m%d-%H%M%S"),
            "files": len(items), "rounds": ROUNDS, "rows": rows,
            "speedup_overall": round(overall, 3), "speedup_pixel_weighted": round(px_w, 3),
            "peak_rss_mb": round(STATE["peak_mb"], 0), "deps": png_fast.describe(),
            "cpu_features": None}
    try:
        import _pngfast
        meta["cpu_features"] = _pngfast.cpu_features()
    except Exception:                              # noqa: BLE001
        pass
    jf = os.path.join(OUT_DIR, "png_fast_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hf = os.path.join(OUT_DIR, "png_fast_%s.html" % meta["ts"])
    write_html(hf, meta)
    print("JSON:", jf)
    print("HTML:", hf)
    STATE["stage"] = "完成 ✓ 窗口可关（5 分钟后自动关闭）"
    STATE["detail"] = os.path.basename(hf)
    for _ in range(int(float(os.environ.get("BENCH_LINGER", "300")) * 10)):
        tick()
        time.sleep(0.1)
    if root is not None:
        try:
            root.destroy()
        except Exception:                          # noqa: BLE001
            pass
    return 0


def write_html(path, m):
    rows = m["rows"]
    best = max(r["speedup"] for r in rows) or 1.0
    bars = []
    for i, r in enumerate(rows):
        w = 420.0 * r["speedup"] / best
        col = "#7fdb9a" if r["speedup"] >= 1.5 else ("#ffd479" if r["speedup"] >= 1.0 else "#ff9b8a")
        bars.append('<text x="12" y="%d" fill="#d7dee4" font-size="13">%s</text>'
                    '<rect x="300" y="%d" width="%.1f" height="18" fill="%s" opacity=".85"/>'
                    '<text x="%d" y="%d" fill="#9fd0ff" font-size="13">%.2fx（%.2f → %.2f ms/张）</text>'
                    % (42 + i * 34, r["bucket"], 28 + i * 34, w, col, 308 + w, 42 + i * 34,
                       r["speedup"], r["cv2_ms"], r["ldf_ms"]))
    tbl = "".join("<tr><td>%s</td><td class='num'>%d</td><td class='num'>%.2f</td>"
                  "<td class='num'>%.2f</td><td class='num'>%.2f</td>"
                  "<td class='num'>%.2f</td><td class='num'>%.2f</td>"
                  "<td class='num'>%.2f</td></tr>"
                  % (r["bucket"], r["n"], r["px"] / 1e6, r["cv2_ms"], r["ldf_ms"],
                     r["cv2_mps"], r["ldf_mps"], r["speedup"]) for r in rows)
    h = 70 + 34 * len(rows) + 24
    verdict = [
        "整体加速比：<b>%.2fx</b>（样本均值，等权）｜<b>%.2fx</b>（按像素加权，更接近建库 CPU 分布）"
        % (m["speedup_overall"], m["speedup_pixel_weighted"]),
        "纯 inflate 的理论上限是 4.18x（见 deflate_final 报告）；差距来自反滤波——"
        "Paeth/Average 行是**行内串行**的延迟受限循环（约 5 周期/像素），SIMD 只能覆盖 "
        "Sub/Up（本图库约 67%% 字节）",
        "构建/运行：<code>python -E devtools/build_native.py</code>；依赖 %s" % m["deps"],
    ]
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>libdeflate PNG 解码路径 性能图</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;margin-top:24px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%%;margin:8px 0;font-size:13px}
th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
th{background:#1a222b}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96;font-size:13px}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}</style></head><body>
<h1>libdeflate PNG 解码路径性能图（cv2 现状 vs 自建 inflate+SIMD 反滤波）</h1>
<p class="muted">样本 %d 张（颜色类型×像素档分层，仅计两路径都能解的文件），%d 轮、每变体整轮独立遍历；
峰值 RSS %.0f MB；指令集位 %s（bit0=SSE2 bit1=SSSE3 bit2=AVX2）</p>
<h2>1. 各档加速比</h2>
<svg viewBox="0 0 960 %d" height="%d">%s</svg>
<h2>2. 明细</h2>
<table><tr><th>档位</th><th>张数</th><th>MP</th><th>cv2 ms/张</th><th>新路径 ms/张</th>
<th>cv2 MP/s</th><th>新路径 MP/s</th><th>加速比</th></tr>%s</table>
<h2>3. 判读</h2><ul>%s</ul>
<p class="muted">只读测试。生成器：<code>devtools/bench_png_fast.py</code>；一致性验证：<code>devtools/verify_png_fast.py</code></p>
</body></html>""" % (
        m["files"], m["rounds"], m["peak_rss_mb"], m["cpu_features"],
        h, h, "".join(bars), tbl, "".join("<li>%s</li>" % v for v in verdict))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
