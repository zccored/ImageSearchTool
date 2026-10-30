# -*- coding: utf-8 -*-
"""精排白化（ZCA / PCA）—— 侧车实现，免重建、可回退。

背景
----
ImageNet 预训练 ResNet 的特征存在强各向异性：无关图对的余弦中位数约 0.70，
导致绝对阈值（0.85 / 0.89）落在噪声带内。ZCA 白化把协方差拉回单位阵，
可显著收窄噪声带（实测随机对 std 0.080 → 0.044）。

设计
----
* 只对**精排特征**做白化，不动粗筛索引（Hu/指纹）；
* 白化矩阵与白化后的特征都写成侧车：
      <prefix>.whiten.npz       mu(D,) + W(D,K) + meta
      <prefix>.fine_white.npy   已白化的特征 (N,K) float32，mmap 可读
* 检索时只把**查询向量**过一遍 (D×K)，成本可忽略；
  全库特征在建 sidecar 时一次性变换，检索端零额外计算；
* 不启用时行为完全不变；删除侧车即回退。

用法
----
    python main.py whiten --prefix <prefix> [--mode zca|pca] [--dims N]
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Tuple

import numpy as np

WHITEN_SCHEMA = 1


def transform_path(prefix: str) -> str:
    return prefix + ".whiten.npz"


def whitened_path(prefix: str) -> str:
    return prefix + ".fine_white.npy"


def exists(prefix: str) -> bool:
    return os.path.exists(transform_path(prefix)) and os.path.exists(whitened_path(prefix))


# --------------------------------------------------------------------------- 拟合
def fit(x: np.ndarray, *, mode: str = "zca", dims: int = 0,
        eps: float = 1e-6) -> dict[str, Any]:
    """在 (N,D) 特征上拟合白化参数。

    mode="zca"：完整 ZCA，W = V Λ^{-1/2} Vᵀ，输出维度 D；
    mode="pca"：PCA 白化，只保留前 `dims` 个主方向，输出维度 dims。
    协方差用 float64 累加（float32 会在大库上丢精度）。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("特征必须是 (N,D) 二维数组")
    n, d = x.shape
    if n < 2:
        raise ValueError("样本数不足，无法拟合白化")
    mu = x.mean(axis=0, keepdims=True).astype(np.float64)
    xc = x.astype(np.float64) - mu
    cov = (xc.T @ xc) / max(1, n - 1)
    evals, evecs = np.linalg.eigh(cov)          # 升序
    evals = np.maximum(evals, eps)
    if mode == "pca":
        k = int(dims) if int(dims) > 0 else d
        k = max(1, min(k, d))
        order = np.argsort(evals)[::-1][:k]
        vals = evals[order]
        vecs = evecs[:, order]
        w = vecs @ np.diag(1.0 / np.sqrt(vals))
        out_dim = k
    elif mode == "zca":
        w = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        out_dim = d
    else:
        raise ValueError(f"未知白化模式: {mode}")
    return {
        "mu": mu.astype(np.float32),
        "w": w.astype(np.float32),
        "meta": {
            "schema": WHITEN_SCHEMA,
            "mode": mode,
            "dims_in": d,
            "dims_out": out_dim,
            "n_fit": n,
            "eps": float(eps),
        },
    }


# --------------------------------------------------------------------------- 应用
def apply(x: np.ndarray, mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(x - mu) @ W；x 可为 (D,) 或 (M,D)。"""
    x = np.asarray(x, dtype=np.float32)
    return (x - mu.astype(np.float32)) @ w.astype(np.float32)


def l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), 1e-8)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


# --------------------------------------------------------------------------- 落盘
def _atomic_save_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr, allow_pickle=False)
    os.replace(tmp, path)


def save(prefix: str, fitted: dict[str, Any], whitened: np.ndarray) -> Tuple[str, str]:
    """写 <prefix>.whiten.npz 与 <prefix>.fine_white.npy（原子替换）。"""
    tp = transform_path(prefix)
    tmp = tp + ".tmp.npz"
    np.savez(tmp,
             mu=fitted["mu"],
             w=fitted["w"],
             meta=np.array(json.dumps(fitted["meta"], ensure_ascii=False)))
    os.replace(tmp, tp)
    wp = whitened_path(prefix)
    _atomic_save_npy(wp, whitened.astype(np.float32))
    return tp, wp


def remove(prefix: str) -> int:
    """删除白化侧车（npz + 白化特征）；返回实际删除的文件数。"""
    import os as _os
    n = 0
    for p in (transform_path(prefix), whitened_path(prefix)):
        try:
            _os.remove(p)
            n += 1
        except FileNotFoundError:
            pass
        except OSError:
            # 仍被 mmap 占用等情况：留给调用方的行数校验去回退，不在这里中断任务
            pass
    return n


def load(prefix: str) -> Optional[dict[str, Any]]:
    tp = transform_path(prefix)
    if not os.path.exists(tp):
        return None
    try:
        with np.load(tp, allow_pickle=False) as data:
            mu = data["mu"]
            w = data["w"]
            meta_raw = data["meta"]
            meta = json.loads(str(meta_raw)) if meta_raw is not None else {}
    except Exception:
        return None
    return {"mu": mu, "w": w, "meta": meta}


def fit_and_write(prefix: str, feats: np.ndarray, *, mode: str = "zca",
                  dims: int = 0, eps: float = 1e-6) -> dict[str, Any]:
    """拟合并落盘，返回 {mu, w, meta, whitened}。"""
    fitted = fit(feats, mode=mode, dims=dims, eps=eps)
    whitened = apply(feats, fitted["mu"], fitted["w"])
    save(prefix, fitted, whitened)
    fitted["whitened"] = whitened
    return fitted
