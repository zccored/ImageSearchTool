/* ImageSearchTool · PNG 反滤波 SIMD 内核接口（见 png_filter_simd.c 顶部说明） */
#ifndef PF_PNG_FILTER_SIMD_H
#define PF_PNG_FILTER_SIMD_H

#ifdef __cplusplus
extern "C" {
#endif

/* 反滤波 + 打包成 RGB。src 为 inflate 输出（每行 1B 滤波类型 + width*channels），
 * out 为 (height, width, 3) uint8 连续缓冲。
 * channels ∈ {3,4}。返回 0 成功；-1 内存不足；-2 非法滤波类型；-3 参数非法。 */
int pf_unfilter(const unsigned char *src, unsigned char *out,
                int width, int height, int channels);

/* 行缓冲全局预算(字节)与当前占用量（Python 侧可调，用于钳住峰值 RSS） */
void pf_set_rowcap_bytes(size_t n);
size_t pf_row_reserved_bytes(void);

/* 运行时可用的指令集：bit0=SSE2, bit1=SSSE3, bit2=AVX2 */
int pf_cpu_features(void);

#ifdef __cplusplus
}
#endif
#endif
