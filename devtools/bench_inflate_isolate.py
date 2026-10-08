# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 纯 inflate 隔离对比（口径校正）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""把"缓冲分配"从 zlib 计时里剥出去，只比纯 inflate 速度；并给出 cv2 全解码锚点。

⚠️ 作废提示（保留存档）：本脚本 inflate 的是 **Python zlib 重压后的流**，与图库真实 IDAT 流
的 Huffman/LZ77 结构不同（同实现速度可差 3×），故其绝对 MB/s 不可引用。
统一口径的权威版本是 `devtools/bench_deflate_final.py`（真实 IDAT + 每变体整轮遍历）。
保留本脚本仅为对照"输出缓冲分配"这一单一变量的影响。

用法: python -E devtools/bench_inflate_isolate.py [每档张数=6] [轮数=3] [单张上限MB=192]
"""
import ctypes
import hashlib
import json
import os
import random
import statistics as st
import struct
import sys
import time
import zlib
from collections import defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT, THIRD_PARTY_LIBDEFLATE  # noqa: E402

REPO = _HERE
EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_RAW_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 192
OUT_DIR = os.path.join(REPO, "perf_reports")
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}
BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
DEFLATE_SHARE = 0.84        # 实测：PNG 解码耗时里 DEFLATE 占 84%（另 16% 为反滤波/色彩/拷贝）
PNG_SHARE_OF_BUILD = 0.251  # 实测：PNG 解码占整库构建 CPU 的 25.1%
DLL_CANDIDATES = [
    os.environ.get("IMAGE_SEARCH_LIBDEFLATE_DLL", ""),
    THIRD_PARTY_LIBDEFLATE,
]
STATE = {"stage": "初始化", "detail": "", "done": 0, "total": 1, "best": "", "peak_mb": 0.0}


def ihdr(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25]


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


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:                              # noqa: BLE001
        return 0.0


def load_libdeflate():
    for p in DLL_CANDIDATES:
        if not p or not os.path.exists(p):
            continue
        lib = ctypes.CDLL(p)
        lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
        lib.libdeflate_zlib_decompress_ex.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        lib.libdeflate_zlib_decompress_ex.restype = ctypes.c_int
        h = lib.libdeflate_alloc_decompressor()
        scratch = [None]

        def scratch_call(b, n, _lib=lib, _h=h, _s=scratch):
            if _s[0] is None or _s[0].size < n:
                _s[0] = np.empty(max(n, 1 << 20), dtype=np.uint8)
            ao = ctypes.c_size_t()
            rc = _lib.libdeflate_zlib_decompress_ex(
                ctypes.c_void_p(_h), b, len(b), _s[0].ctypes.data_as(ctypes.c_void_p),
                n, None, ctypes.byref(ao))
            if rc != 0:
                raise RuntimeError("libdeflate rc=%d" % rc)
            return _s[0][:ao.value]

        def newbuf_call(b, n, _lib=lib, _h=h):
            buf = np.empty(n, dtype=np.uint8)
            ao = ctypes.c_size_t()
            rc = _lib.libdeflate_zlib_decompress_ex(
                ctypes.c_void_p(_h), b, len(b), buf.ctypes.data_as(ctypes.c_void_p),
                n, None, ctypes.byref(ao))
            if rc != 0:
                raise RuntimeError("libdeflate rc=%d" % rc)
            return buf[:ao.value]
        return scratch_call, newbuf_call, {"ok": True, "path": p, "api":
                                           "libdeflate_zlib_decompress_ex",
                                           "sha1": hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]}
    return None, None, {"ok": False, "reason": "未找到 libdeflate.dll"}


def start_popup():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:                              # noqa: BLE001
        return None, lambda *a, **k: None
    root = tk.Tk()
    root.title("ImageSearchTool · 纯 inflate 隔离对比")
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
    from hybrid_search.io_utils import collect_images
    pngs = [p for p in collect_images(GALLERY_ROOT, EXTS) if p.lower().endswith(".png")]
    random.seed(17)
    random.shuffle(pngs)
    buckets = defaultdict(list)
    for p in pngs:
        try:
            with open(p, "rb") as f:
                info = ihdr(f.read(33))
        except OSError:
            continue
        if not info or info[2] != 8:
            continue
        w, h, _d, ctype = info
        mp = w * h / 1e6
        bn = "0-1MP" if mp < 1 else "1-4MP" if mp < 4 else "4-12MP" if mp < 12 else "12+MP"
        k = "%s/%s" % (CT.get(ctype, "ct%d" % ctype), bn)
        if len(buckets[k]) < N_EACH:
            buckets[k].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 8:
            break

    items = []
    for k in sorted(buckets):
        for p in buckets[k]:
            data = open(p, "rb").read()
            info = ihdr(data)
            rawsz = (info[0] * BPP.get(info[3], 4) * info[2] // 8 + 1) * info[1]
            if rawsz > MAX_RAW_MB * 2 ** 20:
                continue
            items.append({"path": p, "name": os.path.basename(p), "bytes": data,
                          "idat": idat_of(data), "raw": rawsz})
    STATE["total"] = len(items) * ROUNDS
    tick()

    ldf_scratch, ldf_newbuf, ldf_meta = load_libdeflate()
    ic = None
    try:
        import imagecodecs as ic
    except Exception:                              # noqa: BLE001
        pass

    def v_pyzlib_auto(z, n, payload=None):
        return zlib.decompress(z)

    def v_pyzlib_pre(z, n, payload=None):
        return zlib.decompress(z, 15, n)

    def v_ic_zlib(z, n, payload=None):
        return ic.zlib_decode(z)

    def v_ic_zlibng(z, n, payload=None):
        return ic.zlibng_decode(z)

    variants = [("Python zlib 1.2.13（输出大小未知）", v_pyzlib_auto, "库自管/反复扩容"),
                ("Python zlib 1.2.13（bufsize=n 预分配）", v_pyzlib_pre, "预分配"),
                ("imagecodecs zlib " + (ic.zlib_version().split()[-1] if ic else "n/a"),
                 v_ic_zlib, "库自管/反复扩容"),
                ("imagecodecs zlib-ng " + (ic.zlibng_version().split()[-1] if ic else "n/a"),
                 v_ic_zlibng, "库自管/反复扩容")]
    if ldf_scratch:
        variants += [("libdeflate 1.25（scratch 复用）", ldf_scratch, "预分配复用"),
                     ("libdeflate 1.25（每次新建）", ldf_newbuf, "逐张分配")]

    print("样本 %d 张，解出合计 %.2f GB，轮数 %d" % (
        len(items), sum(i["raw"] for i in items) / 2 ** 30, ROUNDS))
    for nm, _f, bt in variants:
        print("   实现: %-40s %s" % (nm, bt))

    acc = defaultdict(list)
    bytesv = defaultdict(int)
    ok_cnt = defaultdict(lambda: [0, 0])
    cv2_t = []
    total_raw = sum(i["raw"] for i in items)
    STATE["stage"] = "配对计时（含 cv2 全解码锚点）"
    STATE["done"] = 0
    for it in items:
        payload = zlib.decompress(it["idat"])
        z = zlib.compress(payload, 6)
        for rnd in range(ROUNDS):
            for nm, fn, _bt in variants:
                t0 = time.perf_counter()
                got = fn(z, it["raw"])
                dt = time.perf_counter() - t0
                gb = got.tobytes() if isinstance(got, np.ndarray) else bytes(got)
                same = (gb == payload) if rnd == 0 else (len(gb) == it["raw"])
                if same:
                    acc[nm].append(dt)
                    bytesv[nm] += it["raw"]
                    ok_cnt[nm][0] += 1
                else:
                    ok_cnt[nm][1] += 1
                del got, gb
            if rnd == 0:
                img = cv2.imdecode(np.frombuffer(it["bytes"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                t0 = time.perf_counter()
                cv2.imdecode(np.frombuffer(it["bytes"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                cv2_t.append(time.perf_counter() - t0)
                del img
            STATE["done"] += 1
            tick()
        del z, payload
        STATE["best"] = "第 %d/%d 轮" % (rnd + 1, ROUNDS)

    def rate(key):
        return (bytesv[key] / sum(acc[key]) / 2 ** 20) if acc.get(key) else 0.0

    med = {k: (st.median(v) * 1e3 if v else 0.0) for k, v in acc.items()}
    rows = [{"name": nm, "buf_type": bt, "mb_s": round(rate(nm), 1),
             "median_ms": round(med[nm], 1), "n": ok_cnt[nm][0],
             "bad": ok_cnt[nm][1]} for nm, _f, bt in variants]
    base = next((r["mb_s"] for r in rows if r["name"].startswith("Python zlib")
                 and "未知" in r["name"]), 0) or 1.0
    pre = next((r["mb_s"] for r in rows if "bufsize=n" in r["name"]), 0) or 1.0
    ldf = next((r["mb_s"] for r in rows if r["name"].startswith("libdeflate")
                and "scratch" in r["name"]), 0) or 1.0
    cv2_ms = (st.median(cv2_t) * 1e3) if cv2_t else 0.0
    ratio_lib_vs_pre = ldf / pre
    proj_ms = cv2_ms * ((1 - DEFLATE_SHARE) + DEFLATE_SHARE / max(ratio_lib_vs_pre, 1e-9))
    build_cpu_cut = PNG_SHARE_OF_BUILD * (1 - 1.0 / max(ratio_lib_vs_pre, 1e-9))
    for r in rows:
        r["ratio_vs_auto"] = round(r["mb_s"] / base, 2)
        r["ratio_vs_pre"] = round(r["mb_s"] / pre, 2)

    print("\n=== 纯 inflate 吞吐（按解出字节）")
    for r in rows:
        print("  %-40s %8.1f MB/s  中位 %7.1f ms  相对未知大小口径 %.2fx  相对预分配 %.2fx"
              % (r["name"], r["mb_s"], r["median_ms"], r["ratio_vs_auto"], r["ratio_vs_pre"]))
    print("  cv2 全 PNG 解码（锚点）: 中位 %.1f ms" % cv2_ms)
    print("  预分配口径下 libdeflate/zlib = %.2fx；按 DEFLATE 占解码 %.0f%%、PNG 占构建CPU %.1f%% 推："
          "解码 %.1f→%.1f ms/张，构建 CPU 约 -%.1f%%"
          % (ratio_lib_vs_pre, DEFLATE_SHARE * 100, PNG_SHARE_OF_BUILD * 100,
             cv2_ms, proj_ms, build_cpu_cut * 100))

    meta = {"tool": "inflate-isolate", "ts": time.strftime("%Y%m%d-%H%M%S"),
            "files": len(items), "raw_gb": round(total_raw / 2 ** 30, 3), "rounds": ROUNDS,
            "rows": rows, "cv2_full_decode_median_ms": round(cv2_ms, 1),
            "deflate_share": DEFLATE_SHARE, "png_share_of_build_cpu": PNG_SHARE_OF_BUILD,
            "ratio_libdeflate_vs_pyzlib_prealloc": round(ratio_lib_vs_pre, 2),
            "projected_decode_ms": round(proj_ms, 1),
            "projected_build_cpu_cut": round(build_cpu_cut, 4),
            "libdeflate": ldf_meta, "peak_rss_mb": round(STATE["peak_mb"], 0),
            "buckets": {k: len(v) for k, v in sorted(buckets.items()) if v}}
    jf = os.path.join(OUT_DIR, "deflate_isolate_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hf = os.path.join(OUT_DIR, "deflate_isolate_%s.html" % meta["ts"])
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
    best = max(r["mb_s"] for r in rows) or 1.0
    bars = []
    for i, r in enumerate(rows):
        w = 460.0 * r["mb_s"] / best
        col = "#7fdb9a" if r["name"].startswith("libdeflate") else (
            "#8ab4ff" if "bufsize" in r["name"] else "#a3a3a3")
        bars.append('<text x="12" y="%d" fill="#d7dee4" font-size="13">%s</text>'
                    '<rect x="330" y="%d" width="%.1f" height="18" fill="%s" opacity=".85"/>'
                    '<text x="%d" y="%d" fill="#9fd0ff" font-size="13">%.0f MB/s（%.2fx 预分配口径）</text>'
                    % (42 + i * 34, r["name"], 28 + i * 34, w, col, 338 + w, 42 + i * 34,
                       r["mb_s"], r["ratio_vs_pre"]))
    tbl = "".join("<tr><td>%s</td><td class='num'>%.1f</td><td class='num'>%.1f</td>"
                  "<td class='num'>%.2f</td><td class='num'>%.2f</td><td>%s</td></tr>"
                  % (r["name"], r["mb_s"], r["median_ms"], r["ratio_vs_auto"],
                     r["ratio_vs_pre"], r["buf_type"]) for r in rows)
    h = 70 + 34 * len(rows) + 24
    verdict = [
        "未知大小口径（Python zlib 反复扩容）把 zlib 显得很慢，两个口径的比值相差 %.1f 倍——"
        "所以换 libdeflate 的真实收益要按<b>预分配口径</b>读。" % (
            rows[0]["mb_s"] / max(next((r["mb_s"] for r in rows if "bufsize" in r["name"]), 1), 1)),
        "预分配口径：libdeflate / Python zlib = <b>%.2fx</b>" % m["ratio_libdeflate_vs_pyzlib_prealloc"],
        "叠加实测占比（DEFLATE 占 PNG 解码 %.0f%%、PNG 解码占构建 CPU %.1f%%）："
        "同类 cv2 解码预计 %.1f → <b>%.1f ms/张</b>，构建 CPU 约 <b>-%.1f%%</b>" % (
            m["deflate_share"] * 100, m["png_share_of_build_cpu"] * 100,
            m["cv2_full_decode_median_ms"], m["projected_decode_ms"],
            m["projected_build_cpu_cut"] * 100),
        "cv2 全 PNG 解码中位 %.1f ms/张（锚点：含反滤波与色彩转换，不是同口径可比项）"
        % m["cv2_full_decode_median_ms"],
    ]
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>纯 inflate 隔离对比（口径校正）</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;margin-top:24px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%%;margin:8px 0;font-size:13px}
th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
th{background:#1a222b}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96;font-size:13px}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}</style></head><body>
<h1>纯 inflate 隔离对比 · 把"输出缓冲分配"从 zlib 计时里剥出去</h1>
<p class="muted">样本 %d 张图库 PNG，解出合计 %.2f GB，%d 轮配对取中位；
峰值 RSS %.0f MB；libdeflate：%s</p>
<h2>1. 纯 inflate 吞吐（MB/s，按解出字节）</h2>
<svg viewBox="0 0 960 %d" height="%d">%s</svg>
<h2>2. 明细</h2>
<table><tr><th>实现</th><th>MB/s</th><th>中位 ms</th><th>相对未知大小口径</th>
<th>相对预分配口径</th><th>输出缓冲策略</th></tr>%s</table>
<h2>3. 判读</h2><ul>%s</ul>
<p class="muted">只读测试。生成器：<code>devtools/bench_inflate_isolate.py</code></p>
</body></html>""" % (m["files"], m["raw_gb"], m["rounds"], m["peak_rss_mb"],
                    (m["libdeflate"].get("path") or m["libdeflate"].get("reason", ""))[:70],
                    h, h, "".join(bars), tbl,
                    "".join("<li>%s</li>" % v for v in verdict))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
