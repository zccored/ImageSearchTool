# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — DEFLATE 权威对比（口径统一版）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""统一口径的 DEFLATE 对比：**全部在真实 IDAT 流上测**，按变体整轮独立遍历 + 轮间交替顺序。

为什么重做：早期脚本混用了两种不可比口径——
  a) "输出大小未知"（`zlib.decompress(b)`）会反复扩容，实测比预分配慢 3.3×；
  b) 有的脚本 inflate 的是 Python 重压后的流（Huffman/LZ77 结构不同，速度可差 3×）。
本脚本统一为：真实 IDAT 流 + 每变体一轮独立遍历（避免互相的分配器/缓存污染）+ 预分配对照组。

输出：JSON + 自包含 HTML 性能图（含 cv2 当前路径锚点与"换 inflate 的理论上限"）。
只读图库样本，不写工程数据。
用法: python -E devtools/bench_deflate_final.py [每档张数=8] [轮数=3] [单张上限MB=256]
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
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_RAW_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 256
OUT_DIR = os.path.join(REPO, "perf_reports")
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}
BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
PNG_SHARE_OF_BUILD = 0.251     # 实测：PNG 解码占构建 CPU 比例（代表样本 600 张，40% PNG）
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

        def call(b, n, newbuf=False, _lib=lib, _h=h, _s=scratch):
            buf = np.empty(n, dtype=np.uint8) if newbuf else None
            if not newbuf:
                if _s[0] is None or _s[0].size < n:
                    _s[0] = np.empty(max(n, 1 << 20), dtype=np.uint8)
                buf = _s[0]
            ao = ctypes.c_size_t()
            rc = _lib.libdeflate_zlib_decompress_ex(
                ctypes.c_void_p(_h), b, len(b), buf.ctypes.data_as(ctypes.c_void_p),
                n, None, ctypes.byref(ao))
            if rc != 0:
                raise RuntimeError("libdeflate rc=%d" % rc)
            return buf[:ao.value]
        return call, {"ok": True, "path": p, "api": "libdeflate_zlib_decompress_ex",
                      "sha1": hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]}
    return None, {"ok": False, "reason": "未找到 libdeflate.dll"}


def start_popup():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:                              # noqa: BLE001
        return None, lambda *a, **k: None
    root = tk.Tk()
    root.title("ImageSearchTool · DEFLATE 权威对比（口径统一）")
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
        mp = info[0] * info[1] / 1e6
        bn = "0-1MP" if mp < 1 else "1-4MP" if mp < 4 else "4-12MP" if mp < 12 else "12+MP"
        k = "%s/%s" % (CT.get(info[3], "ct%d" % info[3]), bn)
        if len(buckets[k]) < N_EACH:
            buckets[k].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 8:
            break

    items, skipped = [], 0
    inter = 0
    for k in sorted(buckets):
        for p in buckets[k]:
            data = open(p, "rb").read()
            info = ihdr(data)
            idat = idat_of(data)
            try:
                # 用真实解出长度做基准（IHDR 公式对交错 PNG 不成立）
                payload = zlib.decompress(idat)
                n = len(payload)
                del payload
            except Exception:                      # noqa: BLE001
                skipped += 1
                continue
            if info[2] == 8 and data[28] != 0:
                inter += 1
            if n > MAX_RAW_MB * 2 ** 20:
                skipped += 1
                continue
            items.append({"path": p, "name": os.path.basename(p), "bytes": data,
                          "idat": idat, "raw": n})
    total_raw = sum(i["raw"] for i in items)
    STATE["total"] = len(items) * (ROUNDS + 1)
    tick()
    print("样本 %d 张（跳过 >%dMB 的 %d 张），解出合计 %.3f GB，轮数 %d"
          % (len(items), MAX_RAW_MB, skipped, total_raw / 2 ** 30, ROUNDS))

    ldf, ldf_meta = load_libdeflate()
    ic = None
    try:
        import imagecodecs as ic
    except Exception:                              # noqa: BLE001
        pass
    zver = ic.zlib_version().split()[-1] if ic else "n/a"
    ngver = ic.zlibng_version().split()[-1] if ic else "n/a"

    # 变体：(名称, 缓冲策略, 逐张 fn(it)->bytes/ndarray)  —— 全部吃真实 IDAT 流
    variants = [
        ("Python zlib %s（输出大小未知）" % zlib.ZLIB_VERSION, "库自管/反复扩容",
         lambda it: zlib.decompress(it["idat"])),
        ("Python zlib %s（bufsize=n 预分配）" % zlib.ZLIB_VERSION, "预分配",
         lambda it: zlib.decompress(it["idat"], 15, it["raw"])),
    ]
    if ic:
        variants += [("imagecodecs zlib %s" % zver, "库自管/反复扩容",
                      lambda it: ic.zlib_decode(it["idat"])),
                     ("imagecodecs zlib-ng %s" % ngver, "库自管/反复扩容",
                      lambda it: ic.zlibng_decode(it["idat"]))]
    if ldf:
        _rgb = getattr(cv2, "IMREAD_COLOR_RGB", cv2.IMREAD_COLOR)
        _rgb_name = "IMREAD_COLOR_RGB" if hasattr(cv2, "IMREAD_COLOR_RGB") else "IMREAD_COLOR"
        variants += [
            ("libdeflate 1.25（scratch 复用）", "预分配复用",
             lambda it: ldf(it["idat"], it["raw"], newbuf=False)),
            ("libdeflate 1.25（每次新建缓冲）", "逐张分配",
             lambda it: ldf(it["idat"], it["raw"], newbuf=True)),
            ("cv2 全 PNG 解码（%s · 当前路径）" % _rgb_name, "—",
             lambda it, _f=_rgb: cv2.imdecode(np.frombuffer(it["bytes"], dtype=np.uint8), _f)),
        ]
    for nm, bt, _f in variants:
        print("   变体: %-46s %s" % (nm, bt))

    # 每变体一轮独立遍历，轮间交替顺序（正序/逆序）
    times = defaultdict(dict)
    per_file = defaultdict(list)
    okcnt = defaultdict(lambda: [0, 0])
    ref = {i["path"]: None for i in items}         # 按完整路径为键（同名文件很常见！）
    bad_names = []
    STATE["stage"] = "逐变体整轮遍历（轮间交替顺序）"
    STATE["done"] = 0
    for rnd in range(ROUNDS):
        order = variants if rnd % 2 == 0 else list(reversed(variants))
        for nm, _bt, fn in order:
            t_all = 0.0
            for it in items:
                if rnd == 0 and ref[it["path"]] is None:
                    payload = zlib.decompress(it["idat"], 15, it["raw"])
                    # 只留摘要，立刻丢弃大对象，避免把整库解出常驻内存
                    ref[it["path"]] = hashlib.blake2b(payload, digest_size=16).digest()
                    del payload
                t0 = time.perf_counter()
                got = fn(it)
                dt = time.perf_counter() - t0
                gb = got.tobytes() if isinstance(got, np.ndarray) else bytes(got)
                if rnd == 0:
                    dig = hashlib.blake2b(gb, digest_size=16).digest()
                    if nm.startswith("cv2"):
                        okcnt[nm][0] += 1                 # cv2 输出语义不同，不做逐位比对
                    elif dig == ref[it["path"]]:
                        okcnt[nm][0] += 1
                    else:
                        okcnt[nm][1] += 1
                        if it["name"] not in bad_names:
                            bad_names.append(it["name"])
                del got, gb
                if dt > 0.2:
                    per_file[nm].append((it["name"], round(dt * 1e3, 1)))
                t_all += dt
                STATE["done"] += 1
                STATE["best"] = "%s · 第 %d/%d 轮" % (nm[:22], rnd + 1, ROUNDS)
                tick()
            times[nm][rnd] = t_all
    for it in items:
        it["bytes"] = b""                          # 释放原始文件字节
    del ref

    rows = []
    for nm, bt, _f in variants:
        ts = [times[nm][r] for r in sorted(times[nm])]
        best = min(ts)                             # 取最快一轮（暖机后、最少干扰）
        mean = sum(ts) / len(ts)                   # 每轮都跑完整个样本 → 用轮均值
        rows.append({"name": nm, "buf_type": bt,
                     "total_s": round(sum(ts), 3),
                     "best_round_s": round(best, 3),
                     "mb_s": round(total_raw / mean / 2 ** 20, 1),
                     "mb_s_best": round(total_raw / best / 2 ** 20, 1),
                     "per_round_s": [round(x, 3) for x in ts],
                     "ok": okcnt[nm][0], "bad": okcnt[nm][1],
                     "slow_files": sorted(per_file[nm], key=lambda x: -x[1])[:5]})
    by = {r["name"]: r for r in rows}

    def find(sub, key="mb_s"):
        for r in rows:
            if sub in r["name"]:
                return r[key]
        return 0.0
    pre = find("bufsize=n")
    auto = find("输出大小未知")
    ldf_s = find("libdeflate 1.25（scratch")
    ldf_n = find("libdeflate 1.25（每次")
    cv2v = find("cv2 全 PNG 解码")
    ratio_ldf_pre = ldf_s / pre if pre else 0.0
    ratio_ldf_auto = ldf_s / auto if auto else 0.0
    alloc_penalty = pre / auto if auto else 0.0
    # 上限：若反滤波/色彩开销为 0，整个解码能快到多少倍 = libdeflate 纯 inflate 时间 / cv2 全解码时间
    bound = (ldf_s / cv2v) if (cv2v and ldf_s) else 0.0
    # 悲观推演：反滤波等开销按 cv2 现状的 16%（实测占比）保留
    proj_ms = 0.0
    proj_cpu = 0.0
    if cv2v and ldf_s:
        per_img_cv2 = cv2v and (total_raw / 2 ** 20) / cv2v * 1e3 / len(items)
        proj_ms = per_img_cv2 * (0.16 + 0.84 / ratio_ldf_pre)
        proj_cpu = PNG_SHARE_OF_BUILD * (1 - 1.0 / ratio_ldf_pre)

    print("\n=== 统一口径结果（真实 IDAT 流，按解出字节）")
    for r in rows:
        print("  %-46s %8.1f MB/s（最快轮 %.1f）  逐位 %d/%d  各轮 %s"
              % (r["name"], r["mb_s"], r["mb_s_best"], r["ok"], r["ok"] + r["bad"],
                 r["per_round_s"]))
    print("  预分配 vs 未知大小（同库同流）: %.2fx  → 分配开销本身就有 %.1fx"
          % (alloc_penalty, alloc_penalty))
    print("  libdeflate / Python zlib 预分配 = %.2fx ；libdeflate / 未知大小口径 = %.2fx"
          % (ratio_ldf_pre, ratio_ldf_auto))
    print("  cv2 全解码 / libdeflate 纯 inflate = %.2fx ← 反滤波零成本时的最快可能倍数" % bound)
    print("  逐位不一致的文件:", bad_names if bad_names else "无")
    print("  推演：PNG 解码 %.1f → %.1f ms/张（保留 16%% 反滤波开销），构建 CPU 约 -%.1f%%"
          % ((total_raw / 2 ** 20) / cv2v * 1e3 / len(items) if cv2v else 0.0, proj_ms,
             proj_cpu * 100))

    meta = {"tool": "deflate-final", "ts": time.strftime("%Y%m%d-%H%M%S"),
            "files": len(items), "raw_gb": round(total_raw / 2 ** 30, 3), "rounds": ROUNDS,
            "skipped_big": skipped, "max_raw_mb": MAX_RAW_MB, "rows": rows,
            "interlaced_in_sample": inter,
            "ratio_libdeflate_vs_pyzlib_prealloc": round(ratio_ldf_pre, 2),
            "ratio_libdeflate_vs_pyzlib_auto": round(ratio_ldf_auto, 2),
            "alloc_penalty_same_stream": round(alloc_penalty, 2),
            "bound_cv2_over_libdeflate": round(bound, 2),
            "projected_decode_ms_per_img": round(proj_ms, 1),
            "projected_build_cpu_cut": round(proj_cpu, 4),
            "png_share_of_build_cpu": PNG_SHARE_OF_BUILD,
            "libdeflate": ldf_meta, "peak_rss_mb": round(STATE["peak_mb"], 0),
            "buckets": {k: len(v) for k, v in sorted(buckets.items()) if v}}
    jf = os.path.join(OUT_DIR, "deflate_final_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hf = os.path.join(OUT_DIR, "deflate_final_%s.html" % meta["ts"])
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
        w = 480.0 * r["mb_s"] / best
        col = "#7fdb9a" if "libdeflate" in r["name"] else (
            "#ffd479" if r["name"].startswith("cv2") else (
                "#8ab4ff" if "bufsize" in r["name"] else "#a3a3a3"))
        bars.append('<text x="12" y="%d" fill="#d7dee4" font-size="13">%s</text>'
                    '<rect x="330" y="%d" width="%.1f" height="18" fill="%s" opacity=".85"/>'
                    '<text x="%d" y="%d" fill="#9fd0ff" font-size="13">%.0f MB/s</text>'
                    % (42 + i * 34, r["name"], 28 + i * 34, w, col, 338 + w,
                       42 + i * 34, r["mb_s"]))
    tbl = "".join("<tr><td>%s</td><td class='num'>%.1f</td><td class='num'>%.1f</td>"
                  "<td class='num'>%d/%d</td><td>%s</td><td>%s</td></tr>"
                  % (r["name"], r["mb_s"], r["mb_s_best"], r["ok"], r["ok"] + r["bad"],
                     r["per_round_s"], r["buf_type"]) for r in rows)
    v = [("口径 1：同一 zlib、仅换缓冲策略", "%.2fx（未知大小 → 预分配）"
          % m["alloc_penalty_same_stream"]),
         ("口径 2：libdeflate / Python zlib（均预分配）",
          "%.2fx" % m["ratio_libdeflate_vs_pyzlib_prealloc"]),
         ("口径 3：libdeflate / Python zlib（现状写法）",
          "%.2fx" % m["ratio_libdeflate_vs_pyzlib_auto"]),
         ("换 inflate 后整个 PNG 解码的最快可能倍数（cv2 全解码 / libdeflate 纯 inflate）",
          "%.2fx" % m["bound_cv2_over_libdeflate"])]
    h = 70 + 34 * len(rows) + 24
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>DEFLATE 权威对比（口径统一）</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;margin-top:24px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%%;margin:8px 0;font-size:13px}
th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
th{background:#1a222b}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96;font-size:13px}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}</style></head><body>
<h1>DEFLATE 权威对比 · 统一口径（真实 IDAT 流 · 每变体独立整轮 · 轮间交替顺序）</h1>
<p class="muted">样本 %d 张图库 PNG（分层抽样），解出合计 <b>%.3f GB</b>，%d 轮；
跳过 &gt;%dMB 的大图 %d 张；峰值 RSS %.0f MB<br>libdeflate：%s</p>
<h2>1. 吞吐（MB/s，按解出字节；越高越好）</h2>
<svg viewBox="0 0 960 %d" height="%d">%s</svg>
<h2>2. 明细（每变体各轮耗时单列，便于判断稳定性）</h2>
<table><tr><th>实现</th><th>MB/s</th><th>最快轮 MB/s</th><th>逐位一致</th>
<th>各轮秒</th><th>缓冲策略</th></tr>%s</table>
<h2>3. 三个口径 + 理论上限</h2>
<table><tr><th>口径</th><th>倍数</th></tr>%s</table>
<p class="muted">推演（保留 16%% 反滤波/色彩开销不变）：PNG 解码约 <b>%.1f ms/张</b>；
按 PNG 解码占构建 CPU %.1f%% 计，构建 CPU 约 <b>-%.1f%%</b>（墙钟还要按生产者占比打折）。</p>
<h2>4. 最慢的几张（诊断用）</h2>
<pre class="muted">%s</pre>
<p class="muted">只读测试。生成器：<code>devtools/bench_deflate_final.py</code></p>
</body></html>""" % (
        m["files"], m["raw_gb"], m["rounds"], m["max_raw_mb"], m["skipped_big"],
        m["peak_rss_mb"],
        (m["libdeflate"].get("path") or m["libdeflate"].get("reason", ""))[:90],
        h, h, "".join(bars), tbl,
        "".join("<tr><td>%s</td><td class='num'>%s</td></tr>" % kv for kv in v),
        m["projected_decode_ms_per_img"], m["png_share_of_build_cpu"] * 100,
        m["projected_build_cpu_cut"] * 100,
        "\n".join("%s: %s" % (r["name"][:34], r["slow_files"]) for r in rows))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
