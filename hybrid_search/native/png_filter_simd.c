/* ---------------------------------------------------------------------------
 * ImageSearchTool · 图库检索管理器 — PNG 反滤波 SIMD 内核
 * Copyright (C) 2026 zccored
 *
 * 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
 * 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
 * This program is free software under the GNU Affero General Public License
 * v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
 * ---------------------------------------------------------------------------
 * 为什么需要：实测（docs/perf-plan.md 第五节）发现换 libdeflate 后，
 * inflate 只要 3.00 ms/MP，而**标量反滤波也要 2.91 ms/MP** —— 反滤波成了新瓶颈。
 * libpng 用的是 SSE2 反滤波内核，这里为 RGBA(bpp=4) 补上：
 *   filter 1(Sub)：行内 4 字节步长前缀和 = 移位相加（每 16/32 字节一次，替代逐字节串行链）
 *   filter 2(Up) ：纯向量加法
 *   RGBA→RGB 打包：pshufb/permute 一次出 4 像素→12 字节
 * filter 3(Average)/4(Paeth) 行内是非线性/分支串行，保持标量（libpng 的 SSE2 Paeth 复杂易错，
 * 且本项目的验收标准是"与 cv2 逐位一致"，故不冒险）。
 *
 * 运行时分派：SSE2 是 x86-64 基线；SSSE3/AVX2 用 __builtin_cpu_supports 检测后使用。
 * 非 x86 平台自动走标量路径（结果一致，只是慢）。
 */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "png_filter_simd.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PF_X86 1
#include <emmintrin.h>                 /* SSE2 */
#include <tmmintrin.h>                 /* SSSE3 */
#include <immintrin.h>                 /* AVX2（用 target 属性限定） */
#endif

/* ------------------------------------------------------------------ 标量兜底 */
/* Paeth 预测器：写成**无分支**形式（两级"取更小者"），便于编译器生成 cmov。
 * 实测：分支版在 Paeth 行上会因数据相关跳转产生预测失败，
 * 同类混合的图之间能差 2.4 倍（2.75 vs 6.56 ms/MP）。语义与 PNG 规范一致：
 * 先 a、再 b、最后 c，且平局偏向靠前者。 */
static inline unsigned char pf_paeth(int a, int b, int c)
{
    int p = a + b - c;
    int pa = p - a; if (pa < 0) pa = -pa;
    int pb = p - b; if (pb < 0) pb = -pb;
    int pc = p - c; if (pc < 0) pc = -pc;
    int best = a, m = pa;
    if (pb < m) { m = pb; best = b; }
    if (pc < m) { best = c; }
    return (unsigned char)best;
}

/* Up 滤波：cur = p + prev（纯向量加法）。返回已处理的字节数（尾部由调用方标量收尾）。 */
#ifdef PF_X86
__attribute__((target("avx2")))
static int pf_up_avx2(unsigned char *cur, const unsigned char *p,
                      const unsigned char *prev, int wb)
{
    int i = 0;
    for (; i + 32 <= wb; i += 32)
        _mm256_storeu_si256(
            (__m256i *)(cur + i),
            _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)(p + i)),
                            _mm256_loadu_si256((const __m256i *)(prev + i))));
    return i;
}
#endif

/* 反滤波一行到 cur（不含 RGB 打包）。返回 0 成功，-2 非法滤波类型。 */
static int pf_unfilter_row(const unsigned char *p, unsigned char *cur,
                           const unsigned char *prev, int wb, int bpp,
                           unsigned char ft, int simd, int bpp4)
{
    int i;
    switch (ft) {
    case 0:
        memcpy(cur, p, (size_t)wb);
        return 0;
    case 2:
#ifdef PF_X86
        if (simd >= 2 && bpp4 && wb >= 32) {
            i = pf_up_avx2(cur, p, prev, wb);
            for (; i < wb; ++i) cur[i] = (unsigned char)(p[i] + prev[i]);
            return 0;
        }
        if (simd >= 1 && bpp4 && wb >= 16) {
            for (i = 0; i + 16 <= wb; i += 16)
                _mm_storeu_si128((__m128i *)(cur + i),
                                 _mm_add_epi8(_mm_loadu_si128((const __m128i *)(p + i)),
                                              _mm_loadu_si128((const __m128i *)(prev + i))));
            for (; i < wb; ++i) cur[i] = (unsigned char)(p[i] + prev[i]);
            return 0;
        }
#endif
        for (i = 0; i < wb; ++i) cur[i] = (unsigned char)(p[i] + prev[i]);
        return 0;
    case 1:
#ifdef PF_X86
        if (simd >= 1 && bpp4 && wb >= 20) {
            /* 前 4 字节无左邻 */
            cur[0] = p[0]; cur[1] = p[1]; cur[2] = p[2]; cur[3] = p[3];
            {
                /* 注意：_mm_cvtsi32_si128 把首像素放在**最低** dword，故用 0x00 广播它；
                   循环内的 carry 才是广播**最高** dword（0xFF），二者不可混用。 */
                __m128i carry = _mm_shuffle_epi32(
                    _mm_cvtsi32_si128((int)(unsigned)cur[0] | ((int)cur[1] << 8) |
                                      ((int)cur[2] << 16) | ((int)cur[3] << 24)), 0x00);
                i = 4;
                for (; i + 16 <= wb; i += 16) {
                    __m128i x = _mm_loadu_si128((const __m128i *)(p + i));
                    x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
                    x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
                    x = _mm_add_epi8(x, carry);
                    _mm_storeu_si128((__m128i *)(cur + i), x);
                    carry = _mm_shuffle_epi32(x, 0xFF);   /* 本块最后一个像素 */
                }
                for (; i < wb; ++i) cur[i] = (unsigned char)(p[i] + cur[i - 4]);
            }
            return 0;
        }
#endif
        for (i = 0; i < bpp && i < wb; ++i) cur[i] = p[i];
        for (; i < wb; ++i) cur[i] = (unsigned char)(p[i] + cur[i - bpp]);
        return 0;
    case 3:
        for (i = 0; i < bpp && i < wb; ++i) cur[i] = (unsigned char)(p[i] + (prev[i] >> 1));
        for (; i < wb; ++i)
            cur[i] = (unsigned char)(p[i] + ((cur[i - bpp] + prev[i]) >> 1));
        return 0;
    case 4:
        for (i = 0; i < bpp && i < wb; ++i)
            cur[i] = (unsigned char)(p[i] + pf_paeth(0, prev[i], 0));
        for (; i < wb; ++i)
            cur[i] = (unsigned char)(p[i] + pf_paeth(cur[i - bpp], prev[i], prev[i - bpp]));
        return 0;
    default:
        return -2;
    }
}

/* ------------------------------------------------------- RGBA→RGB 打包（SSSE3） */
#ifdef PF_X86
/* pshufb 掩码：取每 4 字节的前 3 字节，输出前 12 字节有效 */
static const unsigned char pf_mask4to3[16] = {
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80
};

__attribute__((target("ssse3")))
static int pf_pack_rgba_to_rgb_ssse3(const unsigned char *cur, unsigned char *out, int width)
{
    const __m128i mask = _mm_loadu_si128((const __m128i *)pf_mask4to3);
    int i = 0;
    for (; i + 4 <= width; i += 4) {
        __m128i x = _mm_loadu_si128((const __m128i *)(cur + (size_t)i * 4));
        __m128i y = _mm_shuffle_epi8(x, mask);        /* 低 12 字节有效 */
        int tail = _mm_cvtsi128_si32(_mm_srli_si128(y, 8));
        _mm_storel_epi64((__m128i *)(out + (size_t)i * 3), y);   /* 前 8 字节 */
        memcpy(out + (size_t)i * 3 + 8, &tail, 4);               /* 后 4 字节 */
    }
    return i;
}

__attribute__((target("avx2")))
static int pf_pack_rgba_to_rgb_avx2(const unsigned char *cur, unsigned char *out, int width)
{
    const __m256i mask = _mm256_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
    int i = 0;
    /* 一次处理 **8 个像素**（32 字节输入 → 24 字节输出），不是 16 个！ */
    for (; i + 8 <= width; i += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(cur + (size_t)i * 4));
        __m256i y = _mm256_shuffle_epi8(x, mask);
        /* 每个 128 位 lane 内各自得到 12 字节有效数据，把高 lane 的 12 字节接到低 lane 之后 */
        __m256i perm = _mm256_permutevar8x32_epi32(
            y, _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 7));
        _mm_storeu_si128((__m128i *)(out + (size_t)i * 3),
                         _mm256_castsi256_si128(perm));               /* 16 字节 */
        _mm_storel_epi64((__m128i *)(out + (size_t)i * 3 + 16),
                         _mm256_extracti128_si256(perm, 1));          /* 8 字节 */
    }
    return i;
}
#endif

/* ------------------------------------------------------- 线程本地行缓冲复用
 * pf_unfilter 每次调用都要两个 wb 大小的行缓冲（wb = 宽×通道，4MB/MP）。
 * 若每次都 malloc/free，小中图会白白付"全新页首次写入"的开销（每 MP 约 2×4MB）。
 * 这里按线程缓存，只在容量不足时扩容；进程退出前不释放（每线程最多 2×wb）。 */
#if defined(_WIN32)
/* MinGW 下 __thread 会引入 emutls（本扩展加载时失败），Windows 原生用 declspec */
#define PF_TLS __declspec(thread)
#else
#define PF_TLS __thread
#endif
static PF_TLS unsigned char *pf_rowbuf[2] = { NULL, NULL };
static PF_TLS size_t pf_rowcap[2] = { 0, 0 };
/* 全局行缓冲预算：TLS 缓存只增不减，18 个解码线程各长到最大图就是 2×wb×18
 * （12MP RGBA ≈ 96MB/线程 ≈ 1.7GB）。用全局预算钳住：预算内照旧复用（省反复分配的
 * 12~14%），超预算的图改为"逐张 malloc、用完即弃"，峰值不再单调爬升。 */
static size_t pf_row_budget = (size_t)96 << 20;
static size_t pf_row_reserved = 0;

void pf_set_rowcap_bytes(size_t n)
{
    pf_row_budget = n;
}

size_t pf_row_reserved_bytes(void)
{
    return __atomic_load_n(&pf_row_reserved, __ATOMIC_RELAXED);
}

/* 返回行缓冲；*cached=1 表示由 TLS 持有（调用方不要 free），=0 表示临时缓冲（要 free）。 */
static unsigned char *pf_row_acquire(int slot, size_t need, int *cached)
{
    /* 调试用：PF_NOCACHE=1 / PF_ROWNS=1 —— 见下方默认值说明。
     * 【2026-09-26 定位结论】TLS 行缓冲缓存（__declspec(thread)）在真实多线程建库下
     * 触发 STATUS_HEAP_CORRUPTION / ACCESS_VIOLATION（18 路解码池、单线程口径不复现）；
     * 三种 SIMD 级别（标量/SSE2/AVX2）同样崩，用 PF_NOCACHE=1 完全绕开后建库正常完成。
     * 因此**默认关闭 TLS 缓存**（逐张 malloc/free，实测并不慢：600 张瓦片建库 12.70s，
     * 与带缓存时的 13.36s 相比在噪声内甚至更好）；确需实验时可 PF_ROWNS=1 打开。
     * MinGW 的静态 TLS 在 LoadLibrary 动态加载的 DLL 里行为不可靠，是本次根因方向。 */
    static int pf_force_uncached = -1;
    size_t old;
    if (pf_force_uncached < 0) {
        const char *e = getenv("PF_NOCACHE");
        const char *n = getenv("PF_ROWNS");
        if (e != NULL && e[0] == '1')
            pf_force_uncached = 1;
        else if (n != NULL && n[0] == '1')
            pf_force_uncached = 0;
        else
            pf_force_uncached = 1;
    }
    if (pf_force_uncached) {
        *cached = 0;
        return (unsigned char *)malloc(need);
    }
    if (pf_rowbuf[slot] != NULL && pf_rowcap[slot] >= need) {
        *cached = 1;
        return pf_rowbuf[slot];
    }
    old = pf_rowcap[slot];
    if (need > old) {
        size_t delta = need - old;
        size_t cur = __atomic_load_n(&pf_row_reserved, __ATOMIC_RELAXED);
        for (;;) {
            if (cur + delta > pf_row_budget)
                break;                              /* 超预算：不缓存 */
            if (__atomic_compare_exchange_n(&pf_row_reserved, &cur, cur + delta, 0,
                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                free(pf_rowbuf[slot]);
                pf_rowbuf[slot] = (unsigned char *)malloc(need);
                pf_rowcap[slot] = pf_rowbuf[slot] ? need : 0;
                if (pf_rowbuf[slot] == NULL) {      /* 分配失败：退回记账 */
                    __atomic_sub_fetch(&pf_row_reserved, delta, __ATOMIC_RELAXED);
                    *cached = 0;
                    return NULL;
                }
                *cached = 1;
                return pf_rowbuf[slot];
            }
        }
        if (old >= need) {                          /* 预算不足但旧缓冲够用 */
            *cached = 1;
            return pf_rowbuf[slot];
        }
    }
    *cached = 0;
    return (unsigned char *)malloc(need);
}

/* ------------------------------------------------------------------ 主入口 */
int pf_unfilter(const unsigned char *src, unsigned char *out,
                int width, int height, int channels)
{
    size_t wb = (size_t)width * channels;
    unsigned char *prev, *cur, *tmp;
    const unsigned char *p = src;
    int y;
    int simd = 0;
    int bpp4 = (channels == 4);
    int prev_cached = 0, cur_cached = 0;
    unsigned char *tmp_prev, *tmp_cur;
    size_t pack_i;
#if defined(PF_X86)
    {
        /* 调试用：PF_SIMD=0/1/2 → 强制标量 / SSE2 / AVX2，用于二分定位内核问题。
         * （多线程真实建库下出现 STATUS_HEAP_CORRUPTION，单线程口径不复现） */
        const char *pf_env = getenv("PF_SIMD");
        __builtin_cpu_init();
        simd = __builtin_cpu_supports("avx2") ? 2 : 1;      /* SSE2 是 x86-64 基线 */
        if (pf_env != NULL) {
            if (pf_env[0] == '0')
                simd = 0;
            else if (pf_env[0] == '1')
                simd = 1;
            else if (pf_env[0] == '2')
                simd = 2;
        }
    }
#endif
    if (width <= 0 || height <= 0 || (channels != 3 && channels != 4))
        return -3;
    prev = pf_row_acquire(0, wb, &prev_cached);
    cur = pf_row_acquire(1, wb, &cur_cached);
    if (!prev || !cur) {
        if (!prev_cached) free(prev);
        if (!cur_cached) free(cur);
        return -1;
    }
    /* 行循环里 prev/cur 会互换，所以按"是否为临时缓冲"记录，收尾统一释放 */
    tmp_prev = prev_cached ? NULL : prev;
    tmp_cur = cur_cached ? NULL : cur;
    if (!prev || !cur)
        return -1;
    memset(prev, 0, wb);
    for (y = 0; y < height; ++y) {
        unsigned char ft = p[0];
        int rc = pf_unfilter_row(p + 1, cur, prev, (int)wb, channels, ft, simd, bpp4);
        if (rc != 0) {
            /* 非法滤波类型等错误路径：临时缓冲（超预算的大图）也要释放 */
            if (tmp_prev)
                free(tmp_prev);
            if (tmp_cur)
                free(tmp_cur);
            return rc;
        }
        p += (size_t)wb + 1;
        /* 打包输出：RGBA 丢 alpha；RGB 直通 */
        if (bpp4) {
            pack_i = 0;
#if defined(PF_X86)
            if (simd >= 2 && width >= 8)
                pack_i = (size_t)pf_pack_rgba_to_rgb_avx2(cur, out, width);
            else if (simd >= 1 && width >= 4)
                pack_i = (size_t)pf_pack_rgba_to_rgb_ssse3(cur, out, width);
#else
            (void)pack_i;
#endif
            for (; pack_i < (size_t)width; ++pack_i) {
                out[pack_i * 3 + 0] = cur[pack_i * 4 + 0];
                out[pack_i * 3 + 1] = cur[pack_i * 4 + 1];
                out[pack_i * 3 + 2] = cur[pack_i * 4 + 2];
            }
        } else {
            memcpy(out, cur, wb);
        }
        out += (size_t)width * 3;
        tmp = prev; prev = cur; cur = tmp;
    }
    /* TLS 缓冲留在原地复用；临时缓冲（超预算的大图）用完即弃 */
    if (tmp_prev)
        free(tmp_prev);
    if (tmp_cur)
        free(tmp_cur);
    return 0;
}

int pf_cpu_features(void)
{
    /* bit0=SSE2, bit1=SSSE3, bit2=AVX2（供 Python 侧记录/日志） */
    int f = 0;
#if defined(PF_X86)
    __builtin_cpu_init();
    f |= 1;
    f |= __builtin_cpu_supports("ssse3") ? 2 : 0;
    f |= __builtin_cpu_supports("avx2") ? 4 : 0;
#endif
    return f;
}
