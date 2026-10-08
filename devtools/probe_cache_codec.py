# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 缓存条目编码方案对照（写侧/读侧）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""prep 缓存条目（pre_side×pre_side RGB uint8）用哪种编码更快/更省盘。

动机：工程现状是「cv2 解码 → 预处理 → 缓存写 PNG」。**写侧**要走 zlib 压缩
（实测 level6 只有 ~43 MB/s），是缓存写盘的大头；**读侧**每次命中都要 inflate+反滤波。
libdeflate 只在**解压**上有快路径，压缩要显式绑它的压缩 API。

对照（同一批真实图缩放出的载荷）：
  写：cv2 PNG / Pillow PNG / zlib(level1,6) / libdeflate(level1,6,12)
  读：cv2 PNG 解码 / 我们 libdeflate inflate
指标：每张 ms、压缩后字节、相对 PNG 的体积比；并逐位校验往返一致。

用法: python -E devtools/probe_cache_codec.py [样本张数=120] [边长=256]
"""
import ctypes
import io
import os
import random
import statistics as st
import sys
import time
import zlib

import cv2
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 256


def bind_ldf():
    from hybrid_search import png_fast
    pair = png_fast._load_libdeflate()
    if not pair:
        return None, None
    lib = pair[0]
    lib.libdeflate_alloc_compressor.restype = ctypes.c_void_p
    lib.libdeflate_alloc_compressor.argtypes = [ctypes.c_int]
    lib.libdeflate_zlib_compress_bound.restype = ctypes.c_size_t
    lib.libdeflate_zlib_compress_bound.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.libdeflate_zlib_compress.restype = ctypes.c_size_t
    lib.libdeflate_zlib_compress.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t]
    lib.libdeflate_zlib_decompress_ex.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    lib.libdeflate_zlib_decompress_ex.restype = ctypes.c_int
    lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
    return lib, pair[1]


def main() -> int:
    from hybrid_search.io_utils import collect_images
    lib, dll = bind_ldf()
    print("libdeflate:", dll)
    paths = [p for p in collect_images(GALLERY_ROOT, [".png", ".jpg", ".jpeg"])
             if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.seed(11)
    random.shuffle(paths)
    payloads = []
    for p in paths:
        if len(payloads) >= N:
            break
        try:
            d = open(p, "rb").read()
        except OSError:
            continue
        img = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR_RGB)
        if img is None or img.shape[0] < SIDE or img.shape[1] < SIDE:
            continue
        payloads.append(np.ascontiguousarray(cv2.resize(img, (SIDE, SIDE),
                                                        interpolation=cv2.INTER_AREA)))
    print("载荷 %d 张 %dx%d（%.1f MB 原始）"
          % (len(payloads), SIDE, SIDE, sum(p.nbytes for p in payloads) / 2 ** 20))

    comps = {}
    if lib:
        for lvl in (1, 6, 12):
            comps["libdeflate-%d" % lvl] = lib.libdeflate_alloc_compressor(lvl)
    encoders = {
        "cv2 PNG": lambda a: cv2.imencode(".png", a)[1].tobytes(),
        "Pillow PNG": lambda a: _pil_png(a),
        "zlib-1": lambda a: zlib.compress(a.tobytes(), 1),
        "zlib-6": lambda a: zlib.compress(a.tobytes(), 6),
    }
    for name, c in comps.items():
        encoders[name] = (lambda a, _c=c: _ldf_compress(lib, _c, a))

    res = {}
    for name, fn in encoders.items():
        ts, sizes, ok = [], 0, 0
        for a in payloads:
            t0 = time.perf_counter()
            blob = fn(a)
            ts.append(time.perf_counter() - t0)
            sizes += len(blob)
            # 往返校验
            back = _decode_any(name, blob, a.shape, lib)
            if back is not None and back.tobytes() == a.tobytes():
                ok += 1
        res[name] = {"enc_ms": st.median(ts) * 1e3, "total_s": sum(ts),
                     "kb": sizes / len(payloads) / 1024, "ok": ok,
                     "payload_mb": sum(a.nbytes for a in payloads) / 2 ** 20}
    # 读侧
    dec = {}
    for name, fn in encoders.items():
        blobs = [fn(a) for a in payloads]
        ts = []
        for a, blob in zip(payloads, blobs):
            t0 = time.perf_counter()
            back = _decode_any(name, blob, a.shape, lib)
            ts.append(time.perf_counter() - t0)
            assert back is not None
        dec[name] = {"dec_ms": st.median(ts) * 1e3, "total_s": sum(ts)}
        del blobs

    tot = sum(a.nbytes for a in payloads) / 2 ** 20
    print("\n=== 写侧（每张 %dx%d）" % (SIDE, SIDE))
    base_kb = res["cv2 PNG"]["kb"]
    for name in encoders:
        r = res[name]
        print("  %-14s %6.3f ms/张  合计 %6.2f s  %7.1f KB/张（体积 %.2fx PNG）  往返一致 %d/%d"
              % (name, r["enc_ms"], r["total_s"], r["kb"], r["kb"] / base_kb, r["ok"], len(payloads)))
    print("=== 读侧")
    base_d = dec["cv2 PNG"]["dec_ms"]
    for name in encoders:
        print("  %-14s %6.3f ms/张  合计 %6.2f s  （相对 cv2 PNG 解码 %.2fx）"
              % (name, dec[name]["dec_ms"], dec[name]["total_s"], base_d / dec[name]["dec_ms"]))
    if "libdeflate-6" in res and "cv2 PNG" in res:
        w = res["cv2 PNG"]["enc_ms"] / res["libdeflate-6"]["enc_ms"]
        r = dec["cv2 PNG"]["dec_ms"] / dec["libdeflate-6"]["dec_ms"]
        print("\n判读：换 libdeflate-6 → 写侧 %.2fx、读侧 %.2fx、体积 %.2fx"
              % (w, r, res["libdeflate-6"]["kb"] / base_kb))
    return 0


def _pil_png(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


def _ldf_compress(lib, comp, a: np.ndarray) -> bytes:
    src = np.ascontiguousarray(a)
    n = src.nbytes
    bound = lib.libdeflate_zlib_compress_bound(ctypes.c_void_p(comp), n)
    buf = np.empty(bound, np.uint8)
    got = lib.libdeflate_zlib_compress(
        ctypes.c_void_p(comp), src.ctypes.data_as(ctypes.c_void_p), n,
        buf.ctypes.data_as(ctypes.c_void_p), bound)
    if not got:
        raise RuntimeError("compress failed")
    return buf[:got].tobytes()


def _decode_any(name: str, blob: bytes, shape, lib):
    if name in ("cv2 PNG", "Pillow PNG"):
        arr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR_RGB)
        return arr
    n = int(np.prod(shape))
    out = np.empty(n, np.uint8)
    if name.startswith("libdeflate"):
        h = lib.libdeflate_alloc_decompressor()
        ao = ctypes.c_size_t()
        rc = lib.libdeflate_zlib_decompress_ex(
            ctypes.c_void_p(h), blob, len(blob), out.ctypes.data_as(ctypes.c_void_p),
            n, None, ctypes.byref(ao))
        return out[:ao.value].reshape(shape) if rc == 0 and ao.value == n else None
    back = zlib.decompress(blob)
    return np.frombuffer(back, np.uint8).reshape(shape)


if __name__ == "__main__":
    raise SystemExit(main())
