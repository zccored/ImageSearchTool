# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PNG 反滤波原生核心（Cython）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""PNG 逐行反滤波（filter 0..4）原生实现，并把结果写成紧凑 RGB 输出。

为什么用原生代码：filter 3(Average)/4(Paeth) 在**行内是串行的**（每个像素依赖左邻），
numpy 无法向量化；而 filter 1(Sub)/2(Up) 虽然可以 cumsum，但行数多时仍不如 C 循环。
一行一个 C 循环 + 只保留 RGB（顺手丢掉 alpha）= 一次遍历同时完成反滤波与通道裁剪。

输入：libdeflate 解出的扫描线缓冲（每行 = 1 字节滤波类型 + width*channels 数据）
输出：(height, width, 3) uint8 连续数组，语义与 cv2.imdecode(IMREAD_COLOR_RGB) 对齐

本模块只做"反滤波 + 裁剪"，不做 inflate / 不做调色板 / 不做 16bit —— 那些情况由
Python 侧回退 cv2（见 hybrid_search/png_fast.py）。
"""
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# 原生 SIMD 内核（png_filter_simd.c，与本源文件一起编译）
cdef extern from "png_filter_simd.h":
    int pf_unfilter(const unsigned char* src, unsigned char* out,
                    int width, int height, int channels) noexcept nogil
    int pf_cpu_features() noexcept nogil


cdef inline unsigned char _paeth(int a, int b, int c) noexcept nogil:
    """PNG 规范里的 Paeth 预测器（取 a/b/c 中与 p=a+b-c 最接近者）。"""
    cdef int p = a + b - c
    cdef int pa = p - a
    if pa < 0:
        pa = -pa
    cdef int pb = p - b
    if pb < 0:
        pb = -pb
    cdef int pc = p - c
    if pc < 0:
        pc = -pc
    if pa <= pb and pa <= pc:
        return <unsigned char>a
    if pb <= pc:
        return <unsigned char>b
    return <unsigned char>c


cdef int _unfilter_rgba(const unsigned char* src, unsigned char* out,
                        int width, int height) noexcept nogil:
    """RGBA8 → RGB8：反滤波 4 通道后只写前 3 通道。返回 0 成功，负数为错误码。"""
    cdef int wb = width * 4
    cdef unsigned char* prev = <unsigned char*>malloc(wb)
    cdef unsigned char* cur = <unsigned char*>malloc(wb)
    cdef const unsigned char* p = src
    cdef unsigned char* tmp
    cdef unsigned char ft, x
    cdef int y, i, a, b, c, pv
    if prev == NULL or cur == NULL:
        if prev != NULL:
            free(prev)
        if cur != NULL:
            free(cur)
        return -1
    for i in range(wb):
        prev[i] = 0
    for y in range(height):
        ft = p[0]
        p += 1
        if ft == 0:                                  # None：直接拷贝
            memcpy(cur, p, wb)
        elif ft == 2:                                # Up：只加上一行
            for i in range(wb):
                cur[i] = <unsigned char>(p[i] + prev[i])
        elif ft == 1:                                # Sub：行内前缀和（步长 4）
            for i in range(4):
                cur[i] = p[i]
            for i in range(4, wb):
                cur[i] = <unsigned char>(p[i] + cur[i - 4])
        elif ft == 3:                                # Average
            for i in range(4):
                cur[i] = <unsigned char>(p[i] + (prev[i] >> 1))
            for i in range(4, wb):
                cur[i] = <unsigned char>(p[i] + ((cur[i - 4] + prev[i]) >> 1))
        elif ft == 4:                                # Paeth
            for i in range(4):
                cur[i] = <unsigned char>(p[i] + _paeth(0, prev[i], 0))
            for i in range(4, wb):
                cur[i] = <unsigned char>(p[i] + _paeth(cur[i - 4], prev[i], prev[i - 4]))
        else:
            free(prev)
            free(cur)
            return -2
        p += wb
        for i in range(width):                       # 顺手丢 alpha
            out[0] = cur[0 + i * 4]
            out[1] = cur[1 + i * 4]
            out[2] = cur[2 + i * 4]
            out += 3
        tmp = prev
        prev = cur
        cur = tmp
    free(prev)
    free(cur)
    return 0


cdef int _unfilter_rgb(const unsigned char* src, unsigned char* out,
                       int width, int height) noexcept nogil:
    """RGB8 → RGB8。返回 0 成功，负数为错误码。"""
    cdef int wb = width * 3
    cdef unsigned char* prev = <unsigned char*>malloc(wb)
    cdef unsigned char* cur = <unsigned char*>malloc(wb)
    cdef const unsigned char* p = src
    cdef unsigned char* tmp
    cdef unsigned char ft
    cdef int y, i
    if prev == NULL or cur == NULL:
        if prev != NULL:
            free(prev)
        if cur != NULL:
            free(cur)
        return -1
    for i in range(wb):
        prev[i] = 0
    for y in range(height):
        ft = p[0]
        p += 1
        if ft == 0:
            memcpy(cur, p, wb)
        elif ft == 2:
            for i in range(wb):
                cur[i] = <unsigned char>(p[i] + prev[i])
        elif ft == 1:
            for i in range(3):
                cur[i] = p[i]
            for i in range(3, wb):
                cur[i] = <unsigned char>(p[i] + cur[i - 3])
        elif ft == 3:
            for i in range(3):
                cur[i] = <unsigned char>(p[i] + (prev[i] >> 1))
            for i in range(3, wb):
                cur[i] = <unsigned char>(p[i] + ((cur[i - 3] + prev[i]) >> 1))
        elif ft == 4:
            for i in range(3):
                cur[i] = <unsigned char>(p[i] + _paeth(0, prev[i], 0))
            for i in range(3, wb):
                cur[i] = <unsigned char>(p[i] + _paeth(cur[i - 3], prev[i], prev[i - 3]))
        else:
            free(prev)
            free(cur)
            return -2
        p += wb
        memcpy(out, cur, wb)                         # RGB 直通
        out += wb
        tmp = prev
        prev = cur
        cur = tmp
    free(prev)
    free(cur)
    return 0


def unfilter_to_rgb(const unsigned char[::1] src, unsigned char[:, :, ::1] out):
    """反滤波 + 转 RGB（**SIMD 内核**，见 png_filter_simd.c）。

    src: 解出的扫描线（长度必须 = height*(width*channels+1)）
    out: (height, width, 3) uint8 连续数组（channels 由两者推出：3 或 4）
    返回 0 成功；-1 内存不足；-2 非法滤波类型；-3 参数不合法（调用方应回退 cv2）
    """
    cdef int height = out.shape[0]
    cdef int width = out.shape[1]
    cdef int channels = (src.shape[0] // height - 1) // width if height > 0 else 0
    cdef int rc
    if height <= 0 or width <= 0 or out.shape[2] != 3:
        return -3
    if channels not in (3, 4):
        return -3
    if src.shape[0] != height * (width * channels + 1):
        return -3
    with nogil:
        rc = pf_unfilter(&src[0], &out[0, 0, 0], width, height, channels)
    return rc


def unfilter_to_rgb_scalar(const unsigned char[::1] src, unsigned char[:, :, ::1] out):
    """纯标量实现（与 SIMD 内核交叉校验 / SIMD 不可用时的对照）。参数同 unfilter_to_rgb。"""
    cdef int height = out.shape[0]
    cdef int width = out.shape[1]
    cdef int channels = (src.shape[0] // height - 1) // width if height > 0 else 0
    cdef int rc
    if height <= 0 or width <= 0 or out.shape[2] != 3:
        return -3
    if src.shape[0] != height * (width * channels + 1):
        return -3
    if channels == 4:
        with nogil:
            rc = _unfilter_rgba(&src[0], &out[0, 0, 0], width, height)
        return rc
    if channels == 3:
        with nogil:
            rc = _unfilter_rgb(&src[0], &out[0, 0, 0], width, height)
        return rc
    return -3


def cpu_features():
    """运行时指令集：bit0=SSE2, bit1=SSSE3, bit2=AVX2。"""
    cdef int f
    with nogil:
        f = pf_cpu_features()
    return f
