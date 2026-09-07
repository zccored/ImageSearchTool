# -*- coding: utf-8 -*-
"""
索引持久化层。

磁盘上每个图库索引由同一前缀的多个文件组成：
  <prefix>.meta.json       元数据（版本、参数、条数、库根目录等）
  <prefix>.coarse.npz      粗筛索引：paths / md5s / hu / fp / hu_mean / hu_std
  <prefix>.fine.npz        精排索引（可选）：features（L2 归一化后）

约定：paths 顺序在三处完全一致（coarse / fine 按同一顺序追加），
加载时若发现 fine 存在则做长度校验。索引按“追加式”更新：add 只写增量。
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import numpy as np

from .config import Config


class IndexFiles:
    """按前缀管理三个文件的读写。"""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.meta_path = prefix + ".meta.json"
        self.coarse_path = prefix + ".coarse.npz"
        self.fine_path = prefix + ".fine.npz"

    def meta_exists(self) -> bool:
        return os.path.exists(self.meta_path)

    # --------------------------------------------------------------
    # meta
    # --------------------------------------------------------------
    def save_meta(self, meta: dict) -> None:
        meta.setdefault("version", 1)
        meta.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        _ensure_parent(self.meta_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load_meta(self) -> dict:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------------------
    # coarse
    # --------------------------------------------------------------
    def save_coarse(self, paths: List[str], md5s: List[str],
                    hu: Optional[np.ndarray], fp: Optional[np.ndarray],
                    hu_mean: np.ndarray, hu_std: np.ndarray) -> None:
        _ensure_parent(self.coarse_path)
        np.savez_compressed(
            self.coarse_path,
            paths=np.array(paths, dtype=object),
            md5s=np.array(md5s, dtype=object),
            hu=hu if hu is not None else np.zeros((0, 7), dtype=np.float32),
            fp=fp if fp is not None else np.zeros((0, 1), dtype=np.uint8),
            hu_mean=hu_mean,
            hu_std=hu_std,
        )

    def load_coarse(self) -> dict:
        data = np.load(self.coarse_path, allow_pickle=True)
        return {
            "paths": list(data["paths"]),
            "md5s": list(data["md5s"]),
            "hu": data["hu"] if data["hu"].size > 0 else None,
            "fp": data["fp"] if data["fp"].size > 0 else None,
            "hu_mean": data["hu_mean"],
            "hu_std": data["hu_std"],
        }

    # --------------------------------------------------------------
    # fine（不压缩：fp32 矩阵压缩收益小且耗时）
    # --------------------------------------------------------------
    def save_fine(self, paths: List[str], features: np.ndarray) -> None:
        _ensure_parent(self.fine_path)
        np.savez(self.fine_path, paths=np.array(paths, dtype=object),
                 features=features)

    def fine_exists(self) -> bool:
        return os.path.exists(self.fine_path)

    def load_fine(self, mmap: bool = True) -> dict:
        data = np.load(self.fine_path, allow_pickle=True, mmap_mode="r" if mmap else None)
        paths = list(data["paths"])
        feats = data["features"]
        return {"paths": paths, "features": feats}

    def fine_size(self) -> Optional[int]:
        try:
            return os.path.getsize(self.fine_path)
        except OSError:
            return None

    def coarse_size(self) -> Optional[int]:
        try:
            return os.path.getsize(self.coarse_path)
        except OSError:
            return None


def _ensure_parent(path: str) -> None:
    """索引常被保存到尚不存在的自定义目录（如 GUI 填的新前缀），自动建目录。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def meta_of(cfg: Config, n_coarse: int, db_root: Optional[str],
            has_fine: bool, feature_dim: Optional[int]) -> dict:
    """把可复现参数写入 meta，供 stats / 未来版本迁移判断使用。"""
    return {
        "db_root": db_root or "",
        "n": n_coarse,
        "cfg": {
            "coarse_size": cfg.coarse_size,
            "coarse_blur": cfg.coarse_blur,
            "use_hu": cfg.use_hu,
            "use_fp": cfg.use_fp,
            "hu_weight": cfg.hu_weight,
            "fp_weight": cfg.fp_weight,
            "invert_binary": cfg.invert_binary,
            "png_decoder": getattr(cfg, "png_decoder", "cv2"),
            "model": cfg.model,
            "store_fine": cfg.store_fine,
            "dedup": cfg.dedup,
            "extensions": sorted(cfg.extensions),
        },
        "fine": {
            "exists": has_fine,
            "feature_dim": feature_dim,
        },
    }
