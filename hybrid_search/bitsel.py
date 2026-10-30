# -*- coding: utf-8 -*-
"""二进制指纹的逐位统计与选位（方差/熵 + 相关性去冗余）。

背景
----
粗筛用 64×64 OTSU 二值图（4096 bit，已按字节打包为 512 B/行）。
实测：裁切类变换的漏筛率高达 51%（见 docs/paper/results/regression_summary_variant.md）。
原因之一是**大量位在整库内几乎恒定**（对区分无贡献），而少数高度相关的位
互相重复计票。本模块给出：

1. 逐位统计：置位概率 p、方差 p(1-p)、熵 H(p)；
2. 相关性：在样本上估计位间相关系数矩阵；
3. 选位：按熵排序 + 贪心去相关（阈值可调），得到子集；
4. 掩码汉明：只对选中的位做 popcount（保持 SWAR 快路径）。

所有操作只读，不修改索引。
"""

from __future__ import annotations

import numpy as np

# np.packbits 默认按 MSB-first 打包：位 i 位于字节 i//8 的 (7 - i%8) 位
_BITS_PER_BYTE = 8


def unpack(packed: np.ndarray) -> np.ndarray:
    """(N, B) uint8 打包 -> (N, 8B) uint8 位矩阵（0/1）。"""
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1)


def bit_stats(packed: np.ndarray, *, batch: int = 8192) -> dict:
    """逐位置位概率/方差/熵。分批解包，避免整库解包的内存峰值。"""
    n, nbytes = packed.shape
    nbits = nbytes * _BITS_PER_BYTE
    counts = np.zeros(nbits, dtype=np.int64)
    for s in range(0, n, batch):
        # 直接按 uint8 位计数比 unpackbits 省一半内存
        block = packed[s:s + batch]
        counts += np.unpackbits(block, axis=1).sum(axis=0, dtype=np.int64)
    p = counts / max(n, 1)
    eps = 1e-12
    h = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    var = p * (1.0 - p)
    return {"n": n, "nbits": nbits, "counts": counts, "p": p, "var": var,
            "entropy": h}


def correlation(packed: np.ndarray, *, rows: int = 2000, seed: int = 20261023,
                batch: int = 1024) -> np.ndarray:
    """位间相关系数矩阵（在 rows 个样本上估计，float32，(nbits, nbits)）。

    4096²×4 B ≈ 67 MB；rows=2000 时计算量约 33 GFLOP（数十秒）。
    """
    n = packed.shape[0]
    idx = np.random.default_rng(seed).choice(n, size=min(rows, n), replace=False)
    x = unpack(packed[idx]).astype(np.float32)          # (rows, 4096)
    x -= x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x /= std
    corr = (x.T @ x) / max(1, x.shape[0] - 1)
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def select_bits(stats: dict, k: int, *, corr: np.ndarray | None = None,
                max_abs_corr: float = 0.9, mode: str = "entropy") -> np.ndarray:
    """按熵（或方差）排序，贪心去相关，返回选中的位下标（升序）。

    corr=None 时只按单变量信息量排序（不做去相关）。
    """
    key = stats["entropy"] if mode == "entropy" else stats["var"]
    order = np.argsort(key)[::-1]
    chosen: list[int] = []
    if corr is None:
        return np.sort(order[:max(1, min(k, order.size))])
    for i in order:
        if len(chosen) >= k:
            break
        if chosen:
            # 只与已选位比较（向量化）
            if np.max(np.abs(corr[i, chosen])) > max_abs_corr:
                continue
        chosen.append(int(i))
    if len(chosen) < k:                     # 相关性太强时放宽，补足 k
        for i in order:
            if len(chosen) >= k:
                break
            if i not in chosen:
                chosen.append(int(i))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def byte_mask(bits: np.ndarray, nbytes: int) -> np.ndarray:
    """把位下标集合转成按字节的掩码（(nbytes,) uint8）。"""
    mask = np.zeros(nbytes, dtype=np.uint8)
    for b in np.asarray(bits, dtype=np.int64):
        byte = int(b) // _BITS_PER_BYTE
        pos = 7 - (int(b) % _BITS_PER_BYTE)
        mask[byte] |= np.uint8(1 << pos)
    return mask


def apply_mask(packed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按字节掩码清零未选中的位（返回新数组）。"""
    return (np.asarray(packed, dtype=np.uint8) & mask[None, :]).copy()


def popcount_bytes(xor: np.ndarray) -> np.ndarray:
    """按字节 LUT 的 popcount（用于验证与掩码路径的一致性）。"""
    lut = _POPCOUNT_LUT()
    return lut[xor].sum(axis=1, dtype=np.uint32)


_LUT = None


def _POPCOUNT_LUT() -> np.ndarray:
    global _LUT
    if _LUT is None:
        _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return _LUT


def hamming_masked(q_packed: np.ndarray, db_packed: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
    """只统计掩码内的位：先与掩码，再 popcount。"""
    xor = np.bitwise_xor(np.asarray(db_packed, dtype=np.uint8),
                         np.asarray(q_packed, dtype=np.uint8)[None, :])
    xor &= mask[None, :]
    return popcount_bytes(xor)
