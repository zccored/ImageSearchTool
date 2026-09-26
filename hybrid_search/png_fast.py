# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 快速 PNG 解码（libdeflate + 原生反滤波）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""PNG 快速解码路径：libdeflate(ctypes) inflate + 原生反滤波，直出紧凑 RGB。

为什么能快：实测（`docs/perf-plan.md` 第五节）图库 PNG 的 IDAT 解压里，
libdeflate 纯 inflate 1526 MB/s，而现状 cv2 的**整个**解码（inflate+反滤波+色彩）只有
365 MB/s —— inflate 换 libdeflate 后整个解码的最快可能倍数是 4.18×。

覆盖范围（原型）：**8bit 非交错的 RGB(ctype=2) / RGBA(ctype=6)**。
其余情况（调色板、16bit、灰度、灰度+alpha、隔行、带 tRNS 的 RGB、非法滤波类型、
inflate 长度不符、缺少原生模块或 libdeflate DLL）一律返回 None，由调用方回退 cv2，
并计入 `stats()` 便于判断覆盖率。

线程安全：每个线程各自持有一个 libdeflate 解压器句柄与一个 scratch 缓冲
（复用可省掉大图反复 malloc 的 12%~14% 开销）；超过 `scratch_cap_mb` 的图按需新建。
只读调用，不改任何工程数据。
"""
import os
import struct
import threading
import zlib
from typing import Dict, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------ 常量
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_CT_RGB = 2
_CT_RGBA = 6
# 旁路（libdeflate）解哪些颜色类型 —— 实测（bench_png_fast，逐档加速比）：
#   RGBA：1-4MP 1.43× / 4-12MP 1.25× / 0-1MP 1.27× / 12+MP 1.16×  → 划算
#   RGB ：1.07~1.23×，且 ch=3 没有 SIMD 反滤波内核（走标量），
#         却同样要占"整幅 inflate 缓冲 + 2×wb 行缓冲" → 不划算，直接交 cv2
# 即"不是每张图都值得开旁路"，这条判断直接写进主程序，不做成 UI 开关。
_BYPASS_CT = (_CT_RGBA,)
_SUPPORTED_CT = (_CT_RGB, _CT_RGBA)
# 颜色类型 → 每像素通道数（含 8bit 位深的调色板/灰度）
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DLL_ENV = "IMAGE_SEARCH_LIBDEFLATE_DLL"
_DLL_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native",
                          "libdeflate.dll")
_NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
# 兜底候选：默认留空。加载顺序 = 环境变量 → hybrid_search/native/libdeflate.dll → 本列表。
# （这里曾写死过一条第三方随包 DLL 的本机绝对路径，公开仓库前已移除；
#   自编 DLL 见 devtools/build_libdeflate.py，发布版请放到 native/ 或设环境变量。）
_DLL_FALLBACK: list = []

_LOCK = threading.Lock()
_LDF = None                      # ctypes 句柄/库
_LDF_REASON = ""
_NATIVE = None                   # _pngfast 模块
_NATIVE_REASON = ""
_TL = threading.local()
_SCRATCH_CAP = 32 << 20          # 单线程 scratch 上限 32MB，超过则逐张分配
# 全局 scratch 预算：18 个解码线程各自长到最大图后不释放会把峰值抬高（实测 +520MB）。
# 预算内允许线程缓存复用（省 12~14% 的反复分配），超预算的大图改为"逐张分配、用完即弃"。
_SCRATCH_TOTAL = 256 << 20
_scratch_used = 0
_budget_lock = threading.Lock()
_STATS: Dict[str, int] = {}
_STATS_LOCK = threading.Lock()


def _bump(key: str, n: int = 1) -> None:
    with _STATS_LOCK:
        _STATS[key] = _STATS.get(key, 0) + n


def reset_stats() -> None:
    with _STATS_LOCK:
        _STATS.clear()


def stats() -> Dict[str, int]:
    with _STATS_LOCK:
        return dict(_STATS)


def set_scratch_cap_mb(mb: float) -> None:
    """设置每线程 scratch 复用的上限（MB）。设 0 表示永不复用（每张新建缓冲）。"""
    global _SCRATCH_CAP
    _SCRATCH_CAP = max(0, int(float(mb) * (1 << 20)))


def set_scratch_budget_mb(mb: float) -> None:
    """设置**全局** scratch 缓存预算（MB）。0 表示任何缓冲都不缓存（每张新建）。"""
    global _SCRATCH_TOTAL
    _SCRATCH_TOTAL = max(0, int(float(mb) * (1 << 20)))


def scratch_mb() -> float:
    """当前被各线程缓存占用的 scratch 总量（MB）。"""
    return _scratch_used / 2 ** 20


# ------------------------------------------------------------------ 依赖加载
def _load_native():
    global _NATIVE, _NATIVE_REASON
    if _NATIVE is not None:
        return _NATIVE
    with _LOCK:
        if _NATIVE is not None:
            return _NATIVE
        try:
            import importlib
            import sys
            if _NATIVE_DIR not in sys.path:
                sys.path.insert(0, _NATIVE_DIR)
            _NATIVE = importlib.import_module("_pngfast")
        except Exception as e:                     # noqa: BLE001
            _NATIVE_REASON = "%s: %s" % (type(e).__name__, e)
            _NATIVE = False
        return _NATIVE


def _load_libdeflate():
    global _LDF, _LDF_REASON
    if _LDF is not None:
        return _LDF
    with _LOCK:
        if _LDF is not None:
            return _LDF
        import ctypes
        cands = [os.environ.get(_DLL_ENV, ""), _DLL_LOCAL] + _DLL_FALLBACK
        for p in cands:
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
                _LDF = (lib, p)
                return _LDF
            except Exception as e:                 # noqa: BLE001
                _LDF_REASON = "%s: %s" % (p, e)
        _LDF = False
        if not _LDF_REASON:
            _LDF_REASON = "未找到 libdeflate.dll（可用环境变量 %s 指定）" % _DLL_ENV
        return _LDF


def _ensure_handle():
    """线程本地 (lib, 解压器句柄, ao 结构体)。"""
    h = getattr(_TL, "handle", None)
    if h is not None:
        return h
    pair = _load_libdeflate()
    if not pair:
        return None
    import ctypes
    lib, _path = pair
    handle = lib.libdeflate_alloc_decompressor()
    if not handle:
        return None
    h = (lib, handle, ctypes.c_size_t())
    _TL.handle = h
    return h


def available() -> bool:
    """原生模块与 libdeflate 是否都就绪。"""
    return bool(_load_native()) and bool(_load_libdeflate())


def describe() -> str:
    """人类可读的依赖状态（日志/报告用）。"""
    nat = _load_native()
    ldf = _load_libdeflate()
    return ("原生反滤波 %s；libdeflate %s"
            % (getattr(nat, "__file__", None) or ("不可用(%s)" % _NATIVE_REASON),
               ldf[1] if ldf else ("不可用(%s)" % _LDF_REASON)))


# ------------------------------------------------------------------ PNG 解析
def _parse_ihdr(data: bytes):
    """返回 (w, h, depth, ctype, interlace) 或 None。"""
    if len(data) < 33 or data[:8] != _PNG_SIG or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    depth, ctype, comp, filt, interlace = data[24], data[25], data[26], data[27], data[28]
    if w == 0 or h == 0 or comp != 0 or filt != 0:
        return None
    return w, h, depth, ctype, interlace


def _idat_and_flags(data: bytes) -> Tuple[bytes, bool]:
    """返回 (IDAT 拼接, 是否出现 tRNS)。"""
    i = 8
    parts = []
    has_trns = False
    n = len(data)
    while i + 8 <= n:
        (ln,) = struct.unpack(">I", data[i:i + 4])
        typ = data[i + 4:i + 8]
        if typ == b"IDAT":
            parts.append(data[i + 8:i + 8 + ln])
        elif typ == b"tRNS":
            has_trns = True
        elif typ == b"IEND":
            break
        i += 12 + ln
    return (b"".join(parts), has_trns)


# ------------------------------------------------------------------ 解码
def compat_rgb(data: bytes) -> Optional[np.ndarray]:
    """交叉引用路径：libdeflate 只做 inflate，反滤波/色彩交回 cv2(libpng)。

    做法：把 libdeflate 解出的扫描线用 **stored（level 0）deflate 块**重新封装成一个 PNG
    （除 IDAT 外所有块原样保留、CRC 重算），再交给 cv2.imdecode。libpng 此时只需
    以 memcpy 级速度"解"stored 块，然后跑它自己的 SIMD 反滤波与色彩转换。

    优点：① 反滤波用回 libpng 的优化实现（不必自写 SIMD Paeth）；
          ② 覆盖**全部** PNG 格式（调色板/16 位/灰度/隔行都靠原样 IHDR 交给 libpng）；
          ③ 输出与 cv2 现状天然逐位一致（只有一片无改动的中间缓冲差异）。
    代价：多一次"解出字节 → 重压 stored"的拷贝与 CRC/adler 计算。
    失败/不符合条件时返回 None（调用方回退 cv2）。
    """
    info = _parse_ihdr(data)
    if info is None:
        _bump("cx_fallback_badheader")
        return None
    if not _load_libdeflate():
        _bump("cx_fallback_nodeps")
        return None
    w, h, depth, ctype, interlace = info
    ch = _CHANNELS.get(ctype, 0)
    if ch == 0:
        _bump("cx_fallback_ctype%d" % ctype)
        return None
    bits = w * ch * depth
    stride = (bits + 7) // 8
    expect = h * (stride + 1)
    if expect <= 0 or expect > (1 << 31):
        _bump("cx_fallback_toobig")
        return None
    idat, _trns = _idat_and_flags(data)
    if not idat:
        _bump("cx_fallback_noidat")
        return None
    ent = _ensure_handle()
    if ent is None:
        _bump("cx_fallback_nodeps")
        return None
    import ctypes
    lib, handle, actual = ent
    buf = _scratch_acquire(expect, "cxbuf")
    rc = lib.libdeflate_zlib_decompress_ex(
        ctypes.c_void_p(handle), idat, len(idat),
        buf.ctypes.data_as(ctypes.c_void_p), expect, None, ctypes.byref(actual))
    if rc != 0 or actual.value != expect:
        _bump("cx_fallback_inflate_rc%d" % rc)
        return None
    # 重新封装：stored deflate 流（level 0）承载同一批扫描线
    payload = buf[:expect]
    stream = zlib.compress(payload, 0)
    out = bytearray(_PNG_SIG)
    i, n = 8, len(data)
    while i + 8 <= n:
        (ln,) = struct.unpack(">I", data[i:i + 4])
        typ = data[i + 4:i + 8]
        if typ == b"IDAT":
            i += 12 + ln
            continue
        if typ == b"IEND":
            break
        out += data[i:i + 12 + ln]                  # 其余块（IHDR/PLTE/tRNS/gAMA…）原样
        i += 12 + ln
    out += struct.pack(">I", len(stream)) + b"IDAT" + stream
    out += struct.pack(">I", zlib.crc32(b"IDAT" + stream) & 0xFFFFFFFF)
    out += b"\x00\x00\x00\x00IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    _bump("cx_pack", len(stream))
    _bump("cx_ok")
    _bump("cx_pixels", w * h)
    return bytes(out)


def _scratch_acquire(need: int, slot: str = "scratch"):
    """取一个 ≥need 的缓冲：预算允许则缓存到本线程复用，否则逐张分配。

    工程特征：解码池有 N 个常驻线程（tile_decode_slots/decode_workers），线程会把
    scratch 长到"自己见过的最大图"。若无限增长，一个 12MP RGBA 就是 48MB × N 线程，
    峰值 RSS 随进度单调爬升（实测 libdeflate 路径峰值 +520MB、出现在 95% 进度）。
    这里用全局预算把总量钳住：预算内复用（省反复分配的 12~14%），超预算的图用完即弃。
    """
    global _scratch_used
    buf = getattr(_TL, slot, None)
    if buf is not None and buf.size >= need:
        return buf
    old = buf.size if buf is not None else 0
    keep = False
    if _SCRATCH_CAP and need <= _SCRATCH_CAP:
        with _budget_lock:
            if _scratch_used - old + need <= _SCRATCH_TOTAL:
                _scratch_used += need - old
                keep = True
    new = np.empty(need, dtype=np.uint8)
    if keep:
        setattr(_TL, slot, new)
    elif old:                                      # 原来缓存的还在用，别丢
        return buf if buf.size >= need else new
    return new


def has_libdeflate() -> bool:
    """libdeflate 是否可用（供缓存等调用方决定编码格式）。"""
    return bool(_load_libdeflate())


def ldf_compress(data: bytes, level: int = 6):
    """用 libdeflate 压缩（zlib 封装）。不可用/失败返回 None。

    实测（devtools/probe_cache_codec.py，256×256 缓存载荷）：level6 写侧与 PNG 打平、
    读侧 3.0~3.2×、体积仅 +8%；level1 写侧 1.7×、体积 +15%。
    """
    global _LDF_COMP
    pair = _load_libdeflate()
    if not pair:
        return None
    lib = pair[0]
    if getattr(lib, "libdeflate_zlib_compress", None) is None:
        try:
            import ctypes
            lib.libdeflate_alloc_compressor.restype = ctypes.c_void_p
            lib.libdeflate_alloc_compressor.argtypes = [ctypes.c_int]
            lib.libdeflate_zlib_compress_bound.restype = ctypes.c_size_t
            lib.libdeflate_zlib_compress_bound.argtypes = [ctypes.c_void_p,
                                                           ctypes.c_size_t]
            lib.libdeflate_zlib_compress.restype = ctypes.c_size_t
            lib.libdeflate_zlib_compress.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_void_p, ctypes.c_size_t]
        except Exception:                          # noqa: BLE001
            return None
    try:
        import ctypes
        comp = getattr(_TL, "comp", None)
        if comp is None:
            comp = lib.libdeflate_alloc_compressor(int(level))
            if not comp:
                return None
            _TL.comp = comp
        n = len(data)
        bound = lib.libdeflate_zlib_compress_bound(ctypes.c_void_p(comp), n)
        buf = np.empty(bound, dtype=np.uint8)
        got = lib.libdeflate_zlib_compress(
            ctypes.c_void_p(comp), data, n,
            buf.ctypes.data_as(ctypes.c_void_p), bound)
        if not got:
            return None
        return buf[:got].tobytes()
    except Exception:                              # noqa: BLE001
        return None


def ldf_decompress_into(blob: bytes, out: np.ndarray) -> bool:
    """把 zlib 流解到 out（长度必须恰好等于 data.nbytes）。失败返回 False。"""
    ent = _ensure_handle()
    if ent is None:
        return False
    import ctypes
    lib, handle, actual = ent
    expect = int(out.nbytes)
    rc = lib.libdeflate_zlib_decompress_ex(
        ctypes.c_void_p(handle), blob, len(blob),
        out.ctypes.data_as(ctypes.c_void_p), expect, None, ctypes.byref(actual))
    return rc == 0 and actual.value == expect


def decode_rgb(data: bytes) -> Optional[np.ndarray]:
    """PNG 字节 → (H, W, 3) uint8 RGB（与 cv2 IMREAD_COLOR_RGB 语义对齐）。

    不支持/异常一律返回 None（调用方回退 cv2）；`stats()` 里有原因计数。
    """
    info = _parse_ihdr(data)
    if info is None:
        _bump("fallback_badheader")
        return None
    w, h, depth, ctype, interlace = info
    if depth != 8:
        _bump("fallback_depth%d" % depth)
        return None
    if ctype not in _SUPPORTED_CT:
        _bump("fallback_ctype%d" % ctype)
        return None
    if ctype not in _BYPASS_CT:                    # 见 _BYPASS_CT：不值得开旁路的档
        _bump("skip_ctype%d" % ctype)
        return None
    if interlace != 0:
        _bump("fallback_interlaced")
        return None
    if not _load_native() or not _load_libdeflate():
        _bump("fallback_nodeps")
        return None
    idat, has_trns = _idat_and_flags(data)
    if not idat:
        _bump("fallback_noidat")
        return None
    if has_trns:
        # RGB + tRNS：libpng/OpenCV 会做 tRNS→alpha 展开，为稳妥直接回退
        _bump("fallback_trns")
        return None

    channels = 4 if ctype == _CT_RGBA else 3
    expect = h * (w * channels + 1)
    ent = _ensure_handle()
    if ent is None:
        _bump("fallback_nodeps")
        return None
    import ctypes
    lib, handle, actual = ent

    # 输出缓冲（反滤波目标）：原生层要求 (h, w, 3) 连续
    e = _scratch_acquire(expect, "scratch")
    rc = lib.libdeflate_zlib_decompress_ex(
        ctypes.c_void_p(handle), idat, len(idat),
        e.ctypes.data_as(ctypes.c_void_p), expect, None, ctypes.byref(actual))
    if rc != 0 or actual.value != expect:
        _bump("fallback_inflate_rc%d" % rc)
        _bump("bytes_in", len(idat))
        return None
    _bump("bytes_in", len(idat))

    out = np.empty((h, w, 3), dtype=np.uint8)
    try:
        u = _NATIVE.unfilter_to_rgb(e[:expect], out)
    except Exception as ex:                        # noqa: BLE001
        _bump("fallback_unfilter_exc")
        _bump("unfilter_exc_%s" % type(ex).__name__)
        return None
    if u != 0:
        _bump("fallback_unfilter_%d" % u)
        return None
    _bump("ok")
    _bump("pixels", w * h)
    _bump("bytes_out", expect)
    return out
