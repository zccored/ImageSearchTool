# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — DEFLATE 解码方案对比（第二组：GDeflate / nvCOMP）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""GDeflate（NVIDIA nvCOMP）对图库 PNG 场景的可用性与吞吐判别。

判别三问：
  Q1 能否直接吃图库里的 PNG IDAT（外来 zlib/deflate 流）？—— 决定"能不能替换解码器"
  Q2 GDeflate 的输出是不是标准 DEFLATE 兼容位流？—— 决定"CPU 侧能不能读 GPU 写的缓存"
  Q3 若只用于自建缓存格式，GPU 解压 + H2D/D2H 的端到端吞吐是多少？—— 决定"值不值得改架构"

测量：同一样本上 GPU(GDeflate) 与 CPU(zlib / libdeflate) 交替配对，取中位；
逐位校验；GPU 侧分别记录 纯解码 / 含 H2D / 含 D2H 三种口径；另测 pinned 传输带宽做盈亏平衡。
只读图库样本，不改工程数据。
用法: python -E devtools/bench_gdeflate.py [每档张数=6] [轮数=3] [单张上限MB=192]
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
import torch
# CUDA 守卫：本脚本基准依赖 NVIDIA CUDA；AMD / 无卡环境下友好退出而不是抛栈。
if not torch.cuda.is_available():
    print("需要 NVIDIA CUDA 设备（AMD 显卡或无卡环境无法运行本基准）；已跳过。")
    raise SystemExit(0)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO = _HERE
GALLERY_ROOT = r"F:\视频"
EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_RAW_MB = int(sys.argv[3]) if len(sys.argv) > 3 else 192
OUT_DIR = os.path.join(REPO, "perf_reports")
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}
BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
DLL_CANDIDATES = [
    os.environ.get("IMAGE_SEARCH_LIBDEFLATE_DLL", ""),
    r"D:\code\剩余存储\baidudownload\BaiduNetdisk\module\ImageViewer\libdeflate.dll",
]

STATE = {"stage": "初始化", "detail": "", "done": 0, "total": 1, "best": "", "peak_mb": 0.0}


# ------------------------------------------------------------------ 工具
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

        def call(b, n, _lib=lib, _h=h, _s=scratch):
            if _s[0] is None or _s[0].size < n:
                _s[0] = np.empty(max(n, 1 << 20), dtype=np.uint8)
            ao = ctypes.c_size_t()
            rc = _lib.libdeflate_zlib_decompress_ex(
                ctypes.c_void_p(_h), b, len(b), _s[0].ctypes.data_as(ctypes.c_void_p),
                n, None, ctypes.byref(ao))
            if rc != 0:
                raise RuntimeError("libdeflate rc=%d" % rc)
            return _s[0][:ao.value]
        return call, {"ok": True, "path": p,
                      "sha1": hashlib.sha1(open(p, "rb").read()).hexdigest()[:12],
                      "api": "libdeflate_zlib_decompress_ex"}
    return None, {"ok": False, "reason": "未找到 libdeflate.dll"}


def start_popup():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:                              # noqa: BLE001
        return None, lambda *a, **k: None
    root = tk.Tk()
    root.title("ImageSearchTool · DEFLATE 基准（GDeflate 组）")
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
    ttk.Button(frm, text="隐藏窗口（不影响跑测）",
               command=root.withdraw).pack(anchor="e", pady=(6, 0))
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


def comp_bytes(arr):
    """nvcomp.Array(压缩输出) -> uint8 CUDA tensor（去掉容量尾巴）。"""
    t = torch.from_dlpack(arr)
    n = int(getattr(arr, "buffer_size", 0) or getattr(arr, "size", 0))
    flat = t.view(torch.uint8).reshape(-1)
    return flat[:n] if n and n <= flat.numel() else flat


# ------------------------------------------------------------------ 主流程
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    root, tick = start_popup()
    STATE["stage"] = "枚举图库 + 分层抽样"
    tick()

    from nvidia import nvcomp
    gpu = torch.cuda.get_device_name(0)
    print("nvcomp", nvcomp.__version__, "cuda", nvcomp.__cuda_version__, "| GPU", gpu)

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

    items, skipped = [], 0
    for k in sorted(buckets):
        for p in buckets[k]:
            data = open(p, "rb").read()
            info = ihdr(data)
            rawsz = (info[0] * BPP.get(info[3], 4) * info[2] // 8 + 1) * info[1]
            if rawsz > MAX_RAW_MB * 2 ** 20:
                skipped += 1
                continue
            items.append({"path": p, "idat": idat_of(data), "raw": rawsz,
                          "name": os.path.basename(p)})
    total_raw = sum(i["raw"] for i in items)
    STATE["stage"] = "样本 %d 张 / 解出 %.2f GB" % (len(items), total_raw / 2 ** 30)
    STATE["total"] = max(len(items) * ROUNDS, 1)
    tick()
    print("样本 %d 张（跳过 >%dMB 的 %d 张），解出合计 %.2f GB，轮数 %d"
          % (len(items), MAX_RAW_MB, skipped, total_raw / 2 ** 30, ROUNDS))

    codec = nvcomp.Codec(algorithm="GDeflate", device_id=0)
    ldf_call, ldf_meta = load_libdeflate()

    # ---------------- Q1: 外来 DEFLATE 流能否被 GDeflate 解 ----------------
    STATE["stage"] = "Q1 外来 zlib/deflate 流 → GDeflate 解码"
    STATE["detail"] = ""
    tick()
    q1 = {"zlib_try": 0, "zlib_ok": 0, "raw_try": 0, "raw_ok": 0, "err": ""}
    for it in items:
        out = torch.empty(it["raw"], dtype=torch.uint8, device="cuda")
        for key, blob in (("zlib", it["idat"]), ("raw", it["idat"][2:-4])):
            t = torch.frombuffer(bytearray(blob), dtype=torch.uint8).cuda()
            q1[key + "_try"] += 1
            try:
                codec.decode(nvcomp.as_array(t), out=nvcomp.as_array(out))
                q1[key + "_ok"] += 1
            except Exception as e:                 # noqa: BLE001
                if not q1["err"]:
                    q1["err"] = "%s: %s" % (type(e).__name__, str(e).strip()[:160])
        STATE["done"] += 1
        tick()
    print("Q1 外来流：zlib %d/%d 成功，raw deflate %d/%d 成功；首个报错：%s"
          % (q1["zlib_ok"], q1["zlib_try"], q1["raw_ok"], q1["raw_try"], q1["err"]))

    # ---------------- Q2: GDeflate 输出的互操作性 ----------------
    STATE["stage"] = "Q2 GDeflate 输出位流的互操作性"
    STATE["detail"] = ""
    tick()
    q2 = {}
    probe = torch.arange(1 << 20, dtype=torch.uint8, device="cuda")
    kinds = [("NVCOMP_NATIVE（默认）", None), ("RAW", nvcomp.BitstreamKind.RAW),
             ("WITH_UNCOMPRESSED_SIZE", nvcomp.BitstreamKind.WITH_UNCOMPRESSED_SIZE)]

    def try_encode(label, kind):
        rec = {"encode": None, "zlib15": None, "raw_inflate": None, "len": None,
               "api_error": ""}
        try:
            c = codec if kind is None else nvcomp.Codec(algorithm="GDeflate",
                                                        bitstream_kind=kind, device_id=0)
            e = c.encode(nvcomp.as_array(probe))
            cb = comp_bytes(e).cpu().numpy().tobytes()
            rec["encode"], rec["len"] = "ok", len(cb)
            try:
                zlib.decompress(cb)
                rec["zlib15"] = "ok"
            except Exception as ex:                # noqa: BLE001
                rec["zlib15"] = type(ex).__name__ + ": " + str(ex)[:70]
            try:
                zlib.decompressobj(-15).decompress(cb)
                rec["raw_inflate"] = "ok"
            except Exception as ex:                # noqa: BLE001
                rec["raw_inflate"] = type(ex).__name__ + ": " + str(ex)[:70]
        except Exception as ex:                    # noqa: BLE001
            rec["encode"] = type(ex).__name__
            rec["api_error"] = str(ex).replace("\n", " ")[:400]
        q2[label] = rec
        print("Q2 %-24s encode=%-14s len=%-8s zlib=%-32s raw=%s"
              % (label, rec["encode"], rec["len"], rec["zlib15"], rec["raw_inflate"]))
        if rec["api_error"]:
            print("    api:", rec["api_error"][:200])
    for label, kind in kinds:
        try_encode(label, kind)
    tick()

    # ---------------- 传输带宽（pinned） ----------------
    STATE["stage"] = "测 PCIe 传输带宽（pinned H2D / D2H）"
    tick()
    xf = {}
    for size_mb in (64, 256):
        n = size_mb << 20
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        for _ in range(2):
            dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        h2d = n * 5 / (time.perf_counter() - t0) / 2 ** 30
        for _ in range(2):
            host.copy_(dev, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            host.copy_(dev, non_blocking=True)
        torch.cuda.synchronize()
        d2h = n * 5 / (time.perf_counter() - t0) / 2 ** 30
        xf["%dMB" % size_mb] = {"h2d_gb_s": round(h2d, 2), "d2h_gb_s": round(d2h, 2)}
        print("传输 %4dMB: H2D %.2f GB/s  D2H %.2f GB/s" % (size_mb, h2d, d2h))
        del host, dev
    torch.cuda.empty_cache()
    tick()

    # ---------------- Q3: 端到端吞吐 ----------------
    STATE["stage"] = "Q3 吞吐配对测量（GPU GDeflate vs CPU zlib/libdeflate）"
    STATE["done"] = 0
    acc = defaultdict(list)
    bytev = defaultdict(int)
    pk = defaultdict(float)
    bufm = defaultdict(float)

    def rec(key, dt, nbytes):
        acc[key].append(dt)
        bytev[key] += nbytes

    correct = {"gpu_gdeflate": [0, 0, ""], "cpu_zlib": [0, 0, ""], "cpu_libdeflate": [0, 0, ""]}
    sizes = {"payload": 0, "zlib": 0, "gdeflate": 0}
    torch.cuda.synchronize()
    vram0 = torch.cuda.memory_allocated()
    done_raw = 0
    # 逐张处理：一次只留一张的 payload 缓冲，轮内交替顺序
    for i, it in enumerate(items):
        try:
            payload = zlib.decompress(it["idat"])          # 真实解出字节（scanline）
        except Exception as e:                             # noqa: BLE001
            print("跳过（解压失败）", it["name"], e)
            continue
        n = len(payload)
        if n > MAX_RAW_MB * 2 ** 20:
            skipped += 1
            del payload
            continue
        done_raw += n
        zl = zlib.compress(payload, 6)
        t0 = time.perf_counter()
        zlib.compress(payload, 6)
        rec("cpu_zlib_compress", time.perf_counter() - t0, n)
        host_src = torch.frombuffer(bytearray(payload), dtype=torch.uint8).pin_memory()
        out = torch.empty(n, dtype=torch.uint8, device="cuda")
        host_out = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        # 写入侧：GPU 压缩（含 H2D）
        t0 = time.perf_counter()
        dev_src = host_src.cuda(non_blocking=True)
        e = codec.encode(nvcomp.as_array(dev_src))
        torch.cuda.synchronize()
        rec("gpu_gdeflate_compress", time.perf_counter() - t0, n)
        enc = comp_bytes(e).clone()
        sizes["payload"] += n
        sizes["zlib"] += len(zl)
        sizes["gdeflate"] += int(enc.numel())
        # 预置：pinned 压缩流（模拟从磁盘读入主机缓冲）+ 两个显存目标缓冲，避免把分配算进计时
        host_g = torch.empty(enc.numel(), dtype=torch.uint8, pin_memory=True)
        host_g.copy_(enc, non_blocking=True)
        dev_enc = torch.empty(enc.numel(), dtype=torch.uint8, device="cuda")
        dev_enc2 = torch.empty(enc.numel(), dtype=torch.uint8, device="cuda")
        dev_src2 = torch.empty(n, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        for rnd in range(ROUNDS):
            # ---- CPU: zlib 解（缓存格式为 zlib）
            t0 = time.perf_counter()
            got = zlib.decompress(zl)
            rec("cpu_zlib", time.perf_counter() - t0, n)
            ok = got == payload
            correct["cpu_zlib"][0 if ok else 1] += 1
            del got
            # ---- CPU: libdeflate 解（同一 zlib 流，复用 scratch）
            if ldf_call is not None:
                t0 = time.perf_counter()
                got = ldf_call(zl, n)
                rec("cpu_libdeflate", time.perf_counter() - t0, n)
                ok = got.tobytes() == payload
                correct["cpu_libdeflate"][0 if ok else 1] += 1
                if not ok and not correct["cpu_libdeflate"][2]:
                    correct["cpu_libdeflate"][2] = "与 zlib 不一致 " + it["name"]
                del got
            bufm["cpu"] = max(bufm["cpu"], n / 2 ** 20)
            # ---- GPU: 纯解码（数据已在显存）
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            codec.decode(nvcomp.as_array(e), out=nvcomp.as_array(out))
            torch.cuda.synchronize()
            rec("gpu_gdeflate_decode", time.perf_counter() - t0, n)
            # ---- GPU: H2D(压缩流) + 解码
            t0 = time.perf_counter()
            dev_enc.copy_(host_g, non_blocking=True)
            codec.decode(nvcomp.as_array(dev_enc), out=nvcomp.as_array(out))
            torch.cuda.synchronize()
            rec("gpu_gdeflate_h2d+decode", time.perf_counter() - t0, n)
            # ---- GPU: H2D + 解码 + D2H（CPU 也要像素时的完整代价）
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dev_enc2.copy_(host_g, non_blocking=True)
            codec.decode(nvcomp.as_array(dev_enc2), out=nvcomp.as_array(out))
            host_out.copy_(out, non_blocking=True)
            torch.cuda.synchronize()
            rec("gpu_gdeflate_h2d+decode+d2h", time.perf_counter() - t0, n)
            # ---- GPU: 纯压缩（源数据已在显存）
            dev_src2.copy_(host_src, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            codec.encode(nvcomp.as_array(dev_src2))
            torch.cuda.synchronize()
            rec("gpu_gdeflate_compress_only", time.perf_counter() - t0, n)
            if rnd == 0:
                gpu_ok = host_out.numpy().tobytes() == payload
                correct["gpu_gdeflate"][0 if gpu_ok else 1] += 1
                if not gpu_ok and not correct["gpu_gdeflate"][2]:
                    correct["gpu_gdeflate"][2] = "往返不一致 " + it["name"]
            STATE["done"] += 1
            STATE["best"] = "第 %d/%d 轮" % (rnd + 1, ROUNDS)
            tick()
        pk["gpu"] = max(pk["gpu"], torch.cuda.max_memory_allocated() / 2 ** 20)
        pk["cpu"] = max(pk["cpu"], rss_mb())
        del payload, zl, host_g, host_src, out, host_out, enc, e, dev_src
        del dev_enc, dev_enc2, dev_src2
    total_raw = done_raw
    STATE["done"] = STATE["total"]
    tick()

    def rate(key):                                 # MB/s（按各变体实际处理的解出字节）
        ts = acc[key]
        return (bytev[key] / sum(ts) / 2 ** 20) if ts else 0.0

    med = {k: (st.median(v) * 1e3 if v else 0.0) for k, v in acc.items()}
    res = {
        "gpu_gdeflate_decode": {"gb_s": rate("gpu_gdeflate_decode") / 1024, "median_ms": med["gpu_gdeflate_decode"]},
        "gpu_gdeflate_h2d+decode": {"gb_s": rate("gpu_gdeflate_h2d+decode") / 1024,
                                    "median_ms": med["gpu_gdeflate_h2d+decode"]},
        "gpu_gdeflate_h2d+decode+d2h": {"gb_s": rate("gpu_gdeflate_h2d+decode+d2h") / 1024,
                                        "median_ms": med["gpu_gdeflate_h2d+decode+d2h"]},
        "gpu_gdeflate_compress": {"gb_s": rate("gpu_gdeflate_compress") / 1024,
                                  "median_ms": med["gpu_gdeflate_compress"]},
        "gpu_gdeflate_compress_only": {"gb_s": rate("gpu_gdeflate_compress_only") / 1024,
                                       "median_ms": med["gpu_gdeflate_compress_only"]},
        "cpu_zlib": {"gb_s": rate("cpu_zlib") / 1024, "median_ms": med["cpu_zlib"]},
        "cpu_libdeflate": {"gb_s": rate("cpu_libdeflate") / 1024, "median_ms": med["cpu_libdeflate"]},
        "cpu_zlib_compress": {"gb_s": rate("cpu_zlib_compress") / 1024,
                              "median_ms": med["cpu_zlib_compress"]},
    }
    print("\n=== Q3 结果（按解出字节计）")
    for k, v in res.items():
        print("  %-30s %8.2f GB/s  中位 %8.1f ms" % (k, v["gb_s"], v["median_ms"]))
    for k in correct:
        print("  逐位校验 %-16s %d/%d %s" % (k, correct[k][0], correct[k][0] + correct[k][1],
                                            correct[k][2]))

    meta = {"tool": "gdeflate-group", "ts": time.strftime("%Y%m%d-%H%M%S"), "gpu": gpu,
            "nvcomp": nvcomp.__version__, "files": len(items),
            "raw_gb": round(total_raw / 2 ** 30, 3), "rounds": ROUNDS,
            "skipped_big": skipped, "max_raw_mb": MAX_RAW_MB,
            "footprint_gb": {k: round(v / 2 ** 30, 3) for k, v in sizes.items()},
            "footprint_ratio": {"zlib": round(sizes["payload"] / max(sizes["zlib"], 1), 2),
                                "gdeflate": round(sizes["payload"] / max(sizes["gdeflate"], 1), 2)},
            "q1_foreign": q1, "q2_interop": q2, "transfer": xf,
            "results": res, "correct": {k: v[:2] for k, v in correct.items()},
            "correct_note": {k: v[2] for k, v in correct.items() if v[2]},
            "peak_gpu_mb": round(pk["gpu"], 0), "peak_rss_mb": round(pk["cpu"], 0),
            "vram_start_mb": round(vram0 / 2 ** 20, 0),
            "max_buf_mb": round(bufm["cpu"], 1), "libdeflate": ldf_meta,
            "buckets": {k: len(v) for k, v in sorted(buckets.items()) if v}}
    jf = os.path.join(OUT_DIR, "deflate_gdeflate_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hf = os.path.join(OUT_DIR, "deflate_gdeflate_%s.html" % meta["ts"])
    write_html(hf, meta)
    print("JSON:", jf)
    print("HTML:", hf)
    STATE["stage"] = "报告已生成 ✓ 窗口可关（5 分钟后自动关闭）"
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
    order = [("GDeflate GPU 纯解码", "gpu_gdeflate_decode", "#7fdb9a"),
             ("GDeflate + H2D", "gpu_gdeflate_h2d+decode", "#8fe3c0"),
             ("GDeflate + H2D + D2H", "gpu_gdeflate_h2d+decode+d2h", "#ffd479"),
             ("CPU libdeflate", "cpu_libdeflate", "#8ab4ff"),
             ("CPU zlib", "cpu_zlib", "#a3a3a3"),
             ("GDeflate GPU 压缩", "gpu_gdeflate_compress", "#c58cff")]
    vals = [(lab, m["results"][k]["gb_s"], col, m["results"][k]["median_ms"]) for lab, k, col in order]
    best = max(v[1] for v in vals) or 1
    bars = []
    for i, (lab, v, col, ms) in enumerate(vals):
        w = 520.0 * v / best
        bars.append('<text x="12" y="%d" fill="#d7dee4" font-size="13">%s</text>'
                    '<rect x="290" y="%d" width="%.1f" height="18" fill="%s" opacity=".85"/>'
                    '<text x="%d" y="%d" fill="#9fd0ff" font-size="13">%.2f GB/s（中位 %.1f ms）</text>'
                    % (42 + i * 34, lab, 28 + i * 34, w, col, 298 + w, 42 + i * 34, v, ms))
    q1 = m["q1_foreign"]
    q2rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                     % (k, str(v["encode"])[:70], str(v["zlib15"])[:70], str(v["raw_inflate"])[:70])
                     for k, v in m["q2_interop"].items())
    xf = "；".join("%s: H2D %.2f GB/s, D2H %.2f GB/s" % (k, v["h2d_gb_s"], v["d2h_gb_s"])
                   for k, v in m["transfer"].items())
    h = 70 + 34 * len(vals) + 20
    cores = [(lab, m["results"][k]["gb_s"], m["results"][k]["median_ms"]) for lab, k, _c in order]
    verdict = []
    verdict.append("Q1 外来 PNG IDAT：GDeflate 解码成功 <b>%d/%d</b>（zlib 流）与 <b>%d/%d</b>（raw deflate）"
                   "→ %s；首个报错 <code>%s</code>"
                   % (q1["zlib_ok"], q1["zlib_try"], q1["raw_ok"], q1["raw_try"],
                      "不能替换现有解码器" if q1["zlib_ok"] == 0 else "需逐档核查", q1["err"]))
    g = m["results"]["gpu_gdeflate_decode"]["gb_s"]
    hd = m["results"]["gpu_gdeflate_h2d+decode+d2h"]["gb_s"]
    c = m["results"]["cpu_libdeflate"]["gb_s"] or 1e-9
    verdict.append("Q3 自建格式口径：GPU 纯解码 %.2f GB/s（CPU libdeflate %.2f GB/s，约 %.1fx）；"
                   "含 H2D+D2H 后 %.2f GB/s（约 %.1fx CPU）"
                   % (g, c, g / c, hd, hd / c))
    fp = m.get("footprint_ratio", {})
    verdict.append("缓存体积（同批像素）：zlib 压缩比 %.2f:1，GDeflate 压缩比 %.2f:1 → "
                   "若换 GDeflate，缓存盘占用变为 zlib 的 %.0f%%"
                   % (fp.get("zlib", 0), fp.get("gdeflate", 0),
                      100.0 * fp.get("zlib", 0) / max(fp.get("gdeflate", 1e-9), 1e-9)))
    verdict.append("传输带宽：%s" % xf)
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>DEFLATE 解码方案对比（GDeflate 组）</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;margin-top:24px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%%;margin:8px 0;font-size:13px}
th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
th{background:#1a222b}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96;font-size:13px}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}.warn{color:#ffb4a2}</style></head><body>
<h1>DEFLATE 解码方案对比 · 第二组：GDeflate（NVIDIA nvCOMP %s）</h1>
<p class="muted">GPU %s；样本 %d 张图库 PNG（分层抽样），解出合计 <b>%.2f GB</b>，%d 轮配对取中位；
缓存体积（同批像素）：zlib %.3f GB（%.2f:1），GDeflate %.3f GB（%.2f:1）；
峰值显存 %.0f MB，峰值 RSS %.0f MB<br>libdeflate 来源：%s</p>
<h2>1. 吞吐（GB/s，按解出字节计）</h2>
<svg viewBox="0 0 960 %d" height="%d">%s</svg>
<h2>2. Q1 可用性：GDeflate 能否吃图库里的 DEFLATE 流</h2>
<p class="warn">zlib 流（IDAT 原样）成功 %d/%d；raw deflate（剥头）成功 %d/%d。<br>
首个报错：<code>%s</code></p>
<h2>3. Q2 互操作性：GDeflate 输出的位流</h2>
<table><tr><th>位流模式</th><th>encode</th><th>zlib(wbits=15) 能否解</th><th>raw inflate 能否解</th></tr>%s</table>
<h2>4. Q3 端到端口径对比</h2>
<table><tr><th>口径</th><th>GB/s</th><th>单张中位 ms</th></tr>%s</table>
<p class="muted">逐位校验：GDeflate 往返 %s，CPU zlib %s，CPU libdeflate %s</p>
<h2>5. 判读</h2><ul>%s</ul>
<p class="muted">只读测试：未写索引、未改图库。生成器：<code>devtools/bench_gdeflate.py</code>。
libdeflate 来源：%s</p></body></html>""" % (
        m["nvcomp"], m["gpu"], m["files"], m["raw_gb"], m["rounds"],
        m["footprint_gb"]["zlib"], m["footprint_ratio"]["zlib"],
        m["footprint_gb"]["gdeflate"], m["footprint_ratio"]["gdeflate"],
        m["peak_gpu_mb"], m["peak_rss_mb"],
        (m["libdeflate"].get("path") or m["libdeflate"].get("reason", ""))[:80],
        h, h, "".join(bars),
        q1["zlib_ok"], q1["zlib_try"], q1["raw_ok"], q1["raw_try"], q1["err"],
        q2rows,
        "".join("<tr><td>%s</td><td class='num'>%.2f</td><td class='num'>%.1f</td></tr>"
                % (k, v, ms) for k, v, ms in cores),
        "%d/%d" % tuple(m["correct"]["gpu_gdeflate"]), "%d/%d" % tuple(m["correct"]["cpu_zlib"]),
        "%d/%d" % tuple(m["correct"]["cpu_libdeflate"]),
        "".join("<li>%s</li>" % v for v in verdict),
        (m["libdeflate"].get("path") or m["libdeflate"].get("reason", "")))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
