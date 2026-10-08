# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — DEFLATE 解码方案对比（第一组：libdeflate）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""libdeflate vs zlib vs zlib-ng：PNG 的 IDAT（zlib/DEFLATE 流）解压对比。

设计（内存友好 + 可判别）：
  * 逐张处理：一次只读一个文件、只留一个输出缓冲；单张解出 > 上限（默认 256MB）跳过；
    参考输出用完即弃，不留全样本副本（否则 64 张 × 100MB = 内存炸弹）；
  * 各实现跑**同一批文件、同一轮内交替顺序**，多轮取中位，消时钟漂移；
  * 逐位校验：每个实现与 Python zlib 的输出必须逐字节相同（zlib 流含 adler32 校验）；
  * libdeflate 走 ctypes（其 API 必须**预先知道输出大小**并给足缓冲）——
    因此额外对比两种缓冲策略：每次新建 / 复用 scratch（真实接入时的省内存做法）；
  * 记录每个实现的峰值 RSS、单张最大输出缓冲占用；
  * 结束时写 JSON + 自包含 HTML 性能图（深色，风格同项目性能报告）；
  * 顶部置顶小窗实时显示进度（隐藏窗口不影响跑）。

只读图库，不写索引、不删文件。
用法: python -E devtools/bench_deflate_group.py [每档张数=8] [轮数=3] [解出上限MB=256]
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
DLL_CANDIDATES = [
    os.environ.get("IMAGE_SEARCH_LIBDEFLATE_DLL", ""),
    THIRD_PARTY_LIBDEFLATE,
]

STATE = {"stage": "初始化", "detail": "", "done": 0, "total": 1,
         "best": "", "peak_mb": 0.0}


# ------------------------------------------------------------------ 工具函数
def ihdr(data: bytes):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25]


def idat_of(data: bytes) -> bytes:
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


def raw_size(info) -> int:
    w, h, depth, ctype = info
    return (w * BPP.get(ctype, 4) * depth // 8 + 1) * h


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:                              # noqa: BLE001
        return 0.0


def load_libdeflate():
    """返回 (lib, 路径, sha1) 或 (None, 原因, '')。"""
    for p in DLL_CANDIDATES:
        if not p or not os.path.exists(p):
            continue
        try:
            lib = ctypes.CDLL(p)
            lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
            lib.libdeflate_free_decompressor.argtypes = [ctypes.c_void_p]
            lib.libdeflate_zlib_decompress_ex.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            lib.libdeflate_zlib_decompress_ex.restype = ctypes.c_int
            h = hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
            return lib, p, h
        except Exception as e:                     # noqa: BLE001
            return None, "绑定失败 %s: %s" % (p, e), ""
    return None, "未找到 libdeflate.dll（可用环境变量 IMAGE_SEARCH_LIBDEFLATE_DLL 指定）", ""


# ------------------------------------------------------------------ 被测实现
def build_variants():
    """[(名称, fn(idat, raw_size) -> bytes|ndarray, 备注, 该实现自己的缓冲描述)]"""
    out = [("Python zlib " + zlib.ZLIB_VERSION,
            lambda b, n: zlib.decompress(b), "", "库自管")]
    try:
        import imagecodecs as ic
        out.append(("imagecodecs zlib " + ic.zlib_version().split()[-1],
                    lambda b, n: ic.zlib_decode(b), "", "库自管"))
        out.append(("imagecodecs zlib-ng " + ic.zlibng_version().split()[-1],
                    lambda b, n: ic.zlibng_decode(b), "", "库自管"))
    except Exception as e:                         # noqa: BLE001
        print("imagecodecs 不可用:", e)

    lib, path, sha1 = load_libdeflate()
    if lib is None:
        print("libdeflate 不可用:", path)
        return out, {"ok": False, "reason": path}
    handle = lib.libdeflate_alloc_decompressor()
    if not handle:
        return out, {"ok": False, "reason": "alloc_decompressor 返回 NULL"}

    def ldf_newbuf(b, n, _lib=lib, _h=handle):
        buf = np.empty(n, dtype=np.uint8)
        ao = ctypes.c_size_t()
        rc = _lib.libdeflate_zlib_decompress_ex(
            ctypes.c_void_p(_h), b, len(b),
            buf.ctypes.data_as(ctypes.c_void_p), n, None, ctypes.byref(ao))
        if rc != 0:
            raise RuntimeError("libdeflate rc=%d" % rc)
        return buf[:ao.value]

    scratch = {"buf": None}

    def ldf_scratch(b, n, _lib=lib, _h=handle, _s=scratch):
        if _s["buf"] is None or _s["buf"].size < n:
            _s["buf"] = np.empty(max(n, 1 << 20), dtype=np.uint8)   # 只增不减，复用
        buf = _s["buf"]
        ao = ctypes.c_size_t()
        rc = _lib.libdeflate_zlib_decompress_ex(
            ctypes.c_void_p(_h), b, len(b),
            buf.ctypes.data_as(ctypes.c_void_p), n, None, ctypes.byref(ao))
        if rc != 0:
            raise RuntimeError("libdeflate rc=%d" % rc)
        return buf[:ao.value]

    ver = "libdeflate(DLL)"
    out.append((ver + " 每次新建缓冲", ldf_newbuf, "ctypes; out 缓冲逐张新建", "numpy 逐张"))
    out.append((ver + " 复用 scratch", ldf_scratch,
                "ctypes; scratch 只增不减复用", "numpy 复用"))
    return out, {"ok": True, "path": path, "sha1": sha1, "api": "libdeflate_zlib_decompress_ex"}


# ------------------------------------------------------------------ 弹窗
def start_popup():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:                              # noqa: BLE001
        return None, lambda *a, **k: None
    root = tk.Tk()
    root.title("ImageSearchTool · DEFLATE 基准（libdeflate 组）")
    root.attributes("-topmost", True)
    root.geometry("+%d+%d" % (max(root.winfo_screenwidth() - 470, 0), 60))
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    lab_stage = ttk.Label(frm, text="初始化…", font=("Microsoft YaHei UI", 10, "bold"))
    lab_stage.pack(anchor="w")
    lab_detail = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=430,
                           justify="left")
    lab_detail.pack(anchor="w", pady=(4, 2))
    bar = ttk.Progressbar(frm, length=430, mode="determinate", maximum=100)
    bar.pack(fill="x", pady=4)
    lab_stat = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=430,
                         justify="left")
    lab_stat.pack(anchor="w")
    ttk.Button(frm, text="隐藏窗口（不影响跑测）",
               command=root.withdraw).pack(anchor="e", pady=(6, 0))
    root.update()

    def tick():
        try:
            STATE["peak_mb"] = max(STATE["peak_mb"], rss_mb())
            lab_stage.config(text=STATE["stage"])
            lab_detail.config(text=STATE["detail"])
            bar.config(value=100.0 * STATE["done"] / max(STATE["total"], 1))
            lab_stat.config(text="进度 %d/%d | 本进程峰值RSS %.0f MB | %s"
                            % (STATE["done"], STATE["total"], STATE["peak_mb"],
                               STATE["best"]))
            root.update()
        except Exception:                          # noqa: BLE001
            pass
    return root, tick


# ------------------------------------------------------------------ 主流程
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
                head = f.read(33)
        except OSError:
            continue
        info = ihdr(head)
        if not info or info[2] != 8:               # 只看 8bit（图库主体）
            continue
        w, h, depth, ctype = info
        mp = w * h / 1e6
        binname = "0-1MP" if mp < 1 else "1-4MP" if mp < 4 else "4-12MP" if mp < 12 else "12+MP"
        key = "%s/%s" % (CT.get(ctype, "ct%d" % ctype), binname)
        if len(buckets[key]) < N_EACH:
            buckets[key].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 8:
            break

    files, skipped, big = [], 0, 0
    for key in sorted(buckets):
        for p in buckets[key]:
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            info = ihdr(data)
            if not info:
                continue
            rs = raw_size(info)
            if rs > MAX_RAW_MB * 2 ** 20:
                skipped += 1
                continue
            big = max(big, rs)
            files.append((p, info, idat_of(data), rs))
    total_raw = sum(f[3] for f in files)
    STATE["stage"] = "样本就绪：%d 张 / 解出 %.2f GB" % (len(files), total_raw / 2 ** 30)
    STATE["detail"] = "抽样档: " + ", ".join("%s=%d" % (k, len(v))
                                            for k, v in sorted(buckets.items()) if v)
    STATE["total"] = len(files) * max(ROUNDS, 1)
    tick()

    variants, ldf_meta = build_variants()
    print("样本 %d 张（跳过 >%dMB 的 %d 张）；解出合计 %.2f GB；单张最大解出 %.0f MB；轮数 %d"
          % (len(files), MAX_RAW_MB, skipped, total_raw / 2 ** 30, big / 2 ** 20, ROUNDS))
    print("libdeflate:", ("%s  sha1=%s  api=%s" % (ldf_meta["path"], ldf_meta["sha1"],
                                                  ldf_meta["api"])) if ldf_meta.get("ok")
          else ldf_meta.get("reason"))
    for name, _, note, _b in variants:
        print("   实现: %-34s %s" % (name, note))

    # ---- 正确性：以 Python zlib 为基准（用完即弃，不留全样本副本）
    STATE["stage"] = "逐位正确性校验"
    STATE["detail"] = "每个实现与 zlib 逐字节比对（含 adler32）"
    STATE["done"] = 0
    tick()
    correct = {name: [0, 0, ""] for name, _, _, _ in variants}
    for i, (p, info, idat, rs) in enumerate(files):
        try:
            r = zlib.decompress(idat)
        except Exception as e:                     # noqa: BLE001
            STATE["detail"] = "zlib 失败 %s: %s" % (os.path.basename(p), e)
            continue
        for name, fn, _n, _b in variants:
            try:
                got = fn(idat, rs)
                gb = got.tobytes() if isinstance(got, np.ndarray) else bytes(got)
                if gb == r:
                    correct[name][0] += 1
                else:
                    correct[name][1] += 1
                    correct[name][2] = "与 zlib 不一致: " + os.path.basename(p)
                del got, gb
            except Exception as e:                 # noqa: BLE001
                correct[name][1] += 1
                if not correct[name][2]:
                    correct[name][2] = "%s: %s" % (type(e).__name__, e)
        del r
        STATE["done"] = i + 1
        tick()

    # ---- 计时
    STATE["stage"] = "计时中（同轮交替顺序，%d 轮取中位）" % ROUNDS
    STATE["done"] = 0
    acc = {name: [] for name, _, _, _ in variants}
    bytev = {name: 0 for name, _, _, _ in variants}
    peakv = {name: 0.0 for name, _, _, _ in variants}
    bufm = {}
    for name, fn, _n, _b in variants:
        bufm[name] = 0.0
    rss0 = rss_mb()
    for rnd in range(ROUNDS):
        order = variants if rnd % 2 == 0 else list(reversed(variants))
        for i, (p, info, idat, rs) in enumerate(files):
            for name, fn, _n, _b in order:
                t0 = time.perf_counter()
                try:
                    fn(idat, rs)
                except Exception:                  # noqa: BLE001
                    continue
                dt = time.perf_counter() - t0
                acc[name].append(dt)
                bytev[name] += rs
                peakv[name] = max(peakv[name], rss_mb())
                bufm[name] = max(bufm[name], rs / 2 ** 20)
            STATE["done"] += 1
            STATE["best"] = "当前轮 %d/%d" % (rnd + 1, ROUNDS)
            if i % 2 == 0:
                tick()

    results = []
    for name, _, note, buftype in variants:
        ts = acc[name]
        if not ts:
            results.append({"name": name, "ok": False,
                            "error": correct[name][2] or "无计时数据"})
            continue
        results.append({
            "name": name, "note": note, "buf_type": buftype,
            "runs": len(ts), "total_s": round(sum(ts), 3),
            "median_ms": round(st.median(ts) * 1000, 1),
            "p90_ms": round(sorted(ts)[int(len(ts) * 0.9)] * 1000, 1),
            "mb_s": round(bytev[name] / sum(ts) / 2 ** 20, 1),
            "correct_ok": correct[name][0], "correct_bad": correct[name][1],
            "correct_note": correct[name][2],
            "peak_rss_mb": round(peakv[name], 0),
            "max_buf_mb": round(bufm[name], 1), "ok": True})
    oks = [r for r in results if r.get("ok")]
    STATE["stage"] = "完成 ✓ 峰值RSS %.0f MB（起始 %.0f MB）" % (STATE["peak_mb"], rss0)
    STATE["detail"] = ""
    STATE["best"] = ("最快: " + max(oks, key=lambda r: r["mb_s"])["name"]) if oks else ""
    tick()

    ts = time.strftime("%Y%m%d-%H%M%S")
    jf = os.path.join(OUT_DIR, "deflate_libdeflate_%s.json" % ts)
    meta = {"tool": "libdeflate-group", "ts": ts, "files": len(files),
            "raw_gb": round(total_raw / 2 ** 30, 3), "max_raw_mb": MAX_RAW_MB,
            "rounds": ROUNDS, "skipped_big": skipped,
            "max_single_raw_mb": round(big / 2 ** 20, 1),
            "peak_rss_mb": round(STATE["peak_mb"], 0),
            "start_rss_mb": round(rss0, 0), "libdeflate": ldf_meta,
            "results": results,
            "buckets": {k: len(v) for k, v in sorted(buckets.items()) if v}}
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hf = os.path.join(OUT_DIR, "deflate_libdeflate_%s.html" % ts)
    write_html(hf, meta)
    print("JSON:", jf)
    print("HTML:", hf)
    for r in oks:
        print("  %-36s %7.1f MB/s  中位 %6.1f ms  逐位 %d/%d  峰值RSS %.0f MB"
              % (r["name"], r["mb_s"], r["median_ms"], r["correct_ok"],
                 r["correct_ok"] + r["correct_bad"], r["peak_rss_mb"]))
    STATE["stage"] = "报告已生成 ✓ 窗口可关（5 分钟后自动关闭）"
    STATE["detail"] = os.path.basename(hf)
    STATE["done"] = STATE["total"]
    for _ in range(3000):
        tick()
        time.sleep(0.1)
    if root is not None:
        try:
            root.destroy()
        except Exception:                          # noqa: BLE001
            pass
    return 0


def write_html(path, meta):
    ok = [r for r in meta["results"] if r.get("ok")]
    best = max(r["mb_s"] for r in ok) if ok else 1.0
    base = next((r["mb_s"] for r in ok if r["name"].startswith("Python zlib")), None)
    bars, bullets = [], []
    for i, r in enumerate(ok):
        w = 560.0 * r["mb_s"] / best
        color = "#7fdb9a" if r["mb_s"] == best else "#8ab4ff"
        rel = ("（相对 zlib %.2fx）" % (r["mb_s"] / base)) if base else ""
        bars.append(
            '<text x="12" y="%d" fill="#d7dee4" font-size="13">%s</text>'
            '<rect x="300" y="%d" width="%.1f" height="18" fill="%s" opacity=".85"/>'
            '<text x="%d" y="%d" fill="#9fd0ff" font-size="13">%.0f MB/s %s</text>'
            % (42 + i * 34, r["name"], 28 + i * 34, w, color,
               308 + w, 42 + i * 34, r["mb_s"], rel))
        bullets.append(
            "<li><b>%s</b>：%.0f MB/s，单流中位 %.1f ms，逐位一致 %d/%d，"
            "单张最大输出缓冲 %.0f MB%s</li>"
            % (r["name"], r["mb_s"], r["median_ms"], r["correct_ok"],
               r["correct_ok"] + r["correct_bad"], r["max_buf_mb"],
               ("，<span style='color:#ffb4a2'>%s</span>" % r["correct_note"]) if r["correct_bad"] else ""))
    rows = "".join(
        "<tr><td>%s</td><td class='num'>%.1f</td><td class='num'>%.1f</td>"
        "<td class='num'>%.0f</td><td class='num'>%d/%d</td>"
        "<td class='num'>%.0f</td><td>%s</td></tr>"
        % (r["name"], r["median_ms"], r["p90_ms"], r["mb_s"], r["correct_ok"],
           r["correct_ok"] + r["correct_bad"], r["peak_rss_mb"], r["buf_type"])
        for r in ok)
    ldf = meta.get("libdeflate", {})
    ldf_line = ("libdeflate 来源：%s（sha1 %s，API %s）"
                % (ldf.get("path"), ldf.get("sha1"), ldf.get("api"))
                if ldf.get("ok") else "libdeflate 未启用：%s" % ldf.get("reason"))
    h = 70 + 34 * max(len(ok), 1) + 30
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>DEFLATE 解码方案对比（libdeflate 组）</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;margin-top:24px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%%;margin:8px 0;font-size:13px}
th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
th{background:#1a222b}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96;font-size:13px}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}</style></head><body>
<h1>DEFLATE（zlib 流）解码方案对比 · 第一组：libdeflate</h1>
<p class="muted">样本：图库分层抽样 <b>%d</b> 张 PNG 的 IDAT 流（%d 轮，同轮交替顺序取中位），
解出合计 <b>%.2f GB</b>；跳过 &gt;%dMB/张的大图 %d 张；单张最大解出 <b>%.0f MB</b>；
进程峰值 RSS <b>%.0f MB</b>（起始 %.0f MB）。<br>%s</p>
<h2>1. 吞吐（MB/s，按解出字节计；越高越好）</h2>
<svg viewBox="0 0 960 %d" height="%d">%s</svg>
<h2>2. 明细</h2>
<table><tr><th>实现</th><th>单流中位 ms</th><th>P90 ms</th><th>MB/s</th>
<th>逐位一致</th><th>该实现峰值RSS MB</th><th>输出缓冲策略</th></tr>%s</table>
<h2>3. 判读</h2>
<ul>%s</ul>
<p class="muted">只读测试：未写索引、未改图库。生成器：<code>devtools/bench_deflate_group.py</code>；
样本档：%s</p>
</body></html>""" % (
        meta["files"], meta["rounds"], meta["raw_gb"], meta["max_raw_mb"],
        meta["skipped_big"], meta["max_single_raw_mb"], meta["peak_rss_mb"],
        meta["start_rss_mb"], ldf_line, h, h, "".join(bars), rows,
        "".join(bullets),
        ", ".join("%s=%d" % kv for kv in meta["buckets"].items()))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
