# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器（二值法粗筛 + ResNet 精排）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
索引持久化层。

磁盘上每个图库索引由同一前缀的多个文件组成：
  <prefix>.meta.json       元数据（版本、参数、条数、库根目录、存储格式等）
  <prefix>.coarse.npz      粗筛索引：paths / md5s / hu / fp / hu_mean / hu_std
  <prefix>.fine.npz        精排索引（可选）：features（L2 归一化后）

另有一代“侧车（sidecar）”存储，把大数组拆成可 mmap 的 .npy，避免 npz 必须
整体读入内存的问题（npz 成员无法 mmap，实测 444k 瓦片库加载 3.6s / 常驻 1.3GB）：
  <prefix>.paths.npy / .md5s.npy / .hu.npy / .fp.npy / .boxes.npy / .hu_stats.npy
  <prefix>.fine.npy / <prefix>.fine_paths.npy
meta 里以 "storage": "sidecar" 标记；只有标记存在且文件齐全才走侧车，
否则自动回退 npz —— 旧索引无需转换即可继续使用（见 compact()）。

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

# 侧车文件名（不含 .npy）
SIDE_COARSE = ("paths", "md5s", "hu", "fp", "hu_stats", "boxes")
SIDE_FINE = ("fine", "fine_paths")


def _atomic_npy(path: str, arr: np.ndarray) -> None:
    """先写临时文件再原子替换，避免中途失败留下半截索引。"""
    tmp = path + ".tmp.npy"
    np.save(tmp, arr, allow_pickle=True)
    os.replace(tmp, path)


class IndexFiles:
    """按前缀管理索引文件的读写（npz 旧格式 + npy 侧车新格式）。"""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.meta_path = prefix + ".meta.json"
        self.coarse_path = prefix + ".coarse.npz"
        self.fine_path = prefix + ".fine.npz"
        self._storage_cache: Optional[str] = None

    def meta_exists(self) -> bool:
        return os.path.exists(self.meta_path)

    # --------------------------------------------------------------
    # 侧车（可 mmap 的 .npy）
    # --------------------------------------------------------------
    def side(self, name: str) -> str:
        return f"{self.prefix}.{name}.npy"

    def sidecar_exists(self, need_fine: bool = False) -> bool:
        need = ["paths", "md5s", "hu_stats"]
        if need_fine:
            need += ["fine", "fine_paths"]
        return all(os.path.exists(self.side(n)) for n in need)

    def storage_of(self, meta: Optional[dict] = None) -> str:
        """返回 "sidecar" 或 "npz"；结果缓存（同一前缀只判一次）。

        额外一致性校验：meta 里的行数必须与侧车文件头一致，否则视为陈旧
        侧车（例如增量写盘只落了 npz），回退 npz 以免读到旧数据。
        """
        if self._storage_cache is not None:
            return self._storage_cache
        if meta is None:
            try:
                meta = self.load_meta()
            except Exception:       # noqa: BLE001 —— meta 缺失/损坏时按旧格式
                meta = {}
        need_fine = bool((meta or {}).get("fine", {}).get("exists"))
        mode = "npz"
        if ((meta or {}).get("storage") == "sidecar"
                and self.sidecar_exists(need_fine)):
            n_meta = int((meta or {}).get("n") or 0)
            n_side = _npy_rows(self.side("paths"))
            if n_meta and n_side is not None and n_meta != n_side:
                print(f"[索引] 侧车行数({n_side})与 meta({n_meta})不符，"
                      f"回退旧 npz 读取：{self.prefix}")
            else:
                mode = "sidecar"
        self._storage_cache = mode
        return mode

    def forget_storage(self) -> None:
        self._storage_cache = None

    # --------------------------------------------------------------
    # meta
    # --------------------------------------------------------------
    def save_meta(self, meta: dict) -> None:
        meta.setdefault("version", 1)
        meta.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        _ensure_parent(self.meta_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.forget_storage()

    def load_meta(self) -> dict:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------------------
    # coarse
    # --------------------------------------------------------------
    def save_coarse(self, paths: List[str], md5s: List[str],
                    hu: Optional[np.ndarray], fp: Optional[np.ndarray],
                    hu_mean: np.ndarray, hu_std: np.ndarray,
                    boxes: Optional[np.ndarray] = None,
                    sidecar: bool = False) -> None:
        _ensure_parent(self.coarse_path)
        if sidecar:
            self._save_coarse_sidecar(paths, md5s, hu, fp, hu_mean, hu_std,
                                      boxes)
            return
        kw = dict(
            paths=np.array(paths, dtype=object),
            md5s=np.array(md5s, dtype=object),
            hu=hu if hu is not None else np.zeros((0, 7), dtype=np.float32),
            fp=fp if fp is not None else np.zeros((0, 1), dtype=np.uint8),
            hu_mean=hu_mean,
            hu_std=hu_std,
        )
        if boxes is not None:
            kw["boxes"] = np.asarray(boxes, dtype=np.float32).reshape(
                len(paths), 4)
        np.savez_compressed(self.coarse_path, **kw)

    def _save_coarse_sidecar(self, paths, md5s, hu, fp, hu_mean, hu_std,
                             boxes) -> None:
        _atomic_npy(self.side("paths"), np.array(paths, dtype=object))
        _atomic_npy(self.side("md5s"), np.array(md5s, dtype=object))
        for name, arr, dt in (("hu", hu, np.float32), ("fp", fp, np.uint8)):
            p = self.side(name)
            if arr is None:
                # 避免上一轮遗留的陈旧文件（长度不符会被读取端丢弃）
                if os.path.exists(p):
                    os.remove(p)
            else:
                _atomic_npy(p, np.ascontiguousarray(arr, dtype=dt))
        hm = (np.zeros(7, dtype=np.float32) if hu_mean is None
              else np.asarray(hu_mean, dtype=np.float32))
        hs = (np.zeros(7, dtype=np.float32) if hu_std is None
              else np.asarray(hu_std, dtype=np.float32))
        _atomic_npy(self.side("hu_stats"), np.stack([hm, hs]))
        if boxes is not None:
            _atomic_npy(self.side("boxes"),
                        np.asarray(boxes, dtype=np.float32).reshape(
                            len(paths), 4))
        elif os.path.exists(self.side("boxes")):
            os.remove(self.side("boxes"))

    def _load_coarse_sidecar(self) -> dict:
        paths = list(np.load(self.side("paths"), allow_pickle=True))
        md5s = list(np.load(self.side("md5s"), allow_pickle=True))
        hu = (np.load(self.side("hu"), mmap_mode="r")
              if os.path.exists(self.side("hu")) else None)
        fp = (np.load(self.side("fp"), mmap_mode="r")
              if os.path.exists(self.side("fp")) else None)
        stats = np.load(self.side("hu_stats"))
        out = {"paths": paths, "md5s": md5s, "hu": hu, "fp": fp,
               "hu_mean": stats[0], "hu_std": stats[1]}
        if os.path.exists(self.side("boxes")):
            b = np.load(self.side("boxes"), mmap_mode="r")
            if len(b) == len(paths):
                out["boxes"] = b
        return out

    def load_coarse(self, sidecar: Optional[bool] = None) -> dict:
        if sidecar is None:
            sidecar = self.storage_of() == "sidecar"
        if sidecar:
            try:
                return self._load_coarse_sidecar()
            except Exception as e:      # noqa: BLE001 —— 侧车异常自动回退 npz
                print(f"[索引] 侧车读取失败（{type(e).__name__}: {e}），"
                      f"回退旧 npz：{self.prefix}")
                self._storage_cache = "npz"
        data = np.load(self.coarse_path, allow_pickle=True)
        out = {
            "paths": list(data["paths"]),
            "md5s": list(data["md5s"]),
            "hu": data["hu"] if data["hu"].size > 0 else None,
            "fp": data["fp"] if data["fp"].size > 0 else None,
            "hu_mean": data["hu_mean"],
            "hu_std": data["hu_std"],
        }
        if "boxes" in data:
            out["boxes"] = np.asarray(data["boxes"], dtype=np.float32)
        return out

    # --------------------------------------------------------------
    # fine（npz 不压缩：fp32 矩阵压缩收益小且耗时；侧车为可 mmap 的 .npy）
    # --------------------------------------------------------------
    def save_fine(self, paths: List[str], features: np.ndarray,
                  sidecar: bool = False) -> None:
        _ensure_parent(self.fine_path)
        if sidecar:
            _atomic_npy(self.side("fine"), np.ascontiguousarray(features))
            _atomic_npy(self.side("fine_paths"),
                        np.array(paths, dtype=object))
            return
        np.savez(self.fine_path, paths=np.array(paths, dtype=object),
                 features=features)

    def fine_exists(self, sidecar: Optional[bool] = None) -> bool:
        if sidecar is None:
            sidecar = self.storage_of() == "sidecar"
        return (os.path.exists(self.side("fine")) if sidecar
                else os.path.exists(self.fine_path))

    def load_fine(self, mmap: bool = True,
                  sidecar: Optional[bool] = None) -> dict:
        if sidecar is None:
            sidecar = self.storage_of() == "sidecar"
        if sidecar:
            try:
                feats = np.load(self.side("fine"),
                                mmap_mode="r" if mmap else None)
                paths = list(np.load(self.side("fine_paths"),
                                     allow_pickle=True))
                return {"paths": paths, "features": feats}
            except Exception as e:      # noqa: BLE001 —— 侧车异常自动回退 npz
                print(f"[索引] 精排侧车读取失败（{type(e).__name__}: {e}），"
                      f"回退旧 npz：{self.prefix}")
        data = np.load(self.fine_path, allow_pickle=True,
                       mmap_mode="r" if mmap else None)
        return {"paths": list(data["paths"]), "features": data["features"]}

    # --------------------------------------------------------------
    # 体积统计（两种格式合计，便于日志/报告）
    # --------------------------------------------------------------
    def fine_size(self) -> Optional[int]:
        return _size_of(self.fine_path, self.side("fine"))

    def coarse_size(self) -> Optional[int]:
        return _size_of(self.coarse_path, *(self.side(n)
                                            for n in SIDE_COARSE))


def prune(prefix: str, drop_paths, progress=None) -> dict:
    """从索引里剔除指定路径的所有条目（删除/转移重复图后保持索引一致）。

    * 整图索引：按路径剔除该行；瓦片索引：剔除该原图的所有瓦片（paths 存原图路径）；
    * 同时重写 coarse 与 fine，保持两者行序一致；
    * 先把 mmap 数组拷成内存数组再落盘（Windows 下被映射的文件无法替换）。
    """
    files = IndexFiles(prefix)
    if not files.meta_exists():
        raise FileNotFoundError(f"索引 {prefix}.* 不存在")
    meta = files.load_meta()
    side = files.storage_of(meta) == "sidecar"
    st = files.load_coarse(sidecar=side)
    paths = list(st["paths"])
    n0 = len(paths)
    drop = {os.path.normcase(os.path.abspath(p)) for p in drop_paths}
    keep = np.fromiter((os.path.normcase(os.path.abspath(p)) not in drop
                        for p in paths), dtype=bool, count=n0)
    n1 = int(keep.sum())
    if n1 == n0:
        return {"prefix": prefix, "removed": 0, "kept": n0}
    if n1 == 0:
        raise RuntimeError(f"{prefix}.* 剔除后为空，已中止（请改用重建索引）")
    if progress:
        progress(0, 3, "读取索引")
    new_paths = [p for p, k in zip(paths, keep) if k]
    new_md5s = [m for m, k in zip(st["md5s"], keep) if k]
    hu_new = None if st["hu"] is None else np.array(st["hu"])[keep]
    fp_new = None if st["fp"] is None else np.array(st["fp"])[keep]
    boxes_new = (None if st.get("boxes") is None
                 else np.array(st["boxes"])[keep])
    hu_mean = np.asarray(st["hu_mean"])
    hu_std = np.asarray(st["hu_std"])
    del st                          # 释放 mmap，否则 Windows 无法替换文件
    has_fine = files.fine_exists(sidecar=side)
    feats_new = None
    if has_fine:
        if progress:
            progress(1, 3, "读取精排索引")
        fine = files.load_fine(mmap=True, sidecar=side)
        feats_new = np.array(fine["features"])[keep]
        del fine
    if progress:
        progress(2, 3, "写回索引")
    files.save_coarse(new_paths, new_md5s, hu_new, fp_new, hu_mean, hu_std,
                      boxes=boxes_new, sidecar=side)
    if has_fine:
        files.save_fine(new_paths, feats_new, sidecar=side)
        meta.setdefault("fine", {})["feature_dim"] = int(feats_new.shape[1])
    meta["n"] = n1
    files.save_meta(meta)
    if progress:
        progress(3, 3, "完成")
    return {"prefix": prefix, "removed": n0 - n1, "kept": n1,
            "has_fine": has_fine}


def _npy_rows(path: str) -> Optional[int]:
    """只读 .npy 文件头拿行数（不加载数据），用于侧车一致性校验。"""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, _fortran, _dtype = np.lib.format._read_array_header(
                f, version)
        return int(shape[0])
    except Exception:               # noqa: BLE001
        return None


def _size_of(*paths: str) -> Optional[int]:
    total, found = 0, False
    for p in paths:
        if os.path.exists(p):
            total += os.path.getsize(p)
            found = True
    return total if found else None


def compact(prefix: str, delete_legacy: bool = False,
            progress=None) -> dict:
    """把旧 npz 索引转换成侧车（.npy）快载格式；原 npz 默认保留。

    * 数值大数组改为 mmap 懒加载：444k 瓦片库实测加载 3.6s -> <1s、
      常驻内存 1.3GB -> 约 0.3GB；
    * 幂等：已是侧车格式时直接返回；转换失败不会破坏原文件（先写 .tmp.npy
      再原子替换，meta 标记最后写）。
    """
    files = IndexFiles(prefix)
    if not files.meta_exists():
        raise FileNotFoundError(f"索引 {prefix}.* 不存在")
    meta = files.load_meta()
    if files.storage_of(meta) == "sidecar":
        return {"prefix": prefix, "already": True, "n": int(meta.get("n") or 0)}

    t0 = time.time()
    st = files.load_coarse(sidecar=False)
    n = len(st["paths"])
    if progress:
        progress(0, 3, "读取旧索引")
    has_fine = bool(meta.get("fine", {}).get("exists")) \
        and os.path.exists(files.fine_path)
    fine = files.load_fine(mmap=False, sidecar=False) if has_fine else None
    if progress:
        progress(1, 3, "写入侧车 .npy")
    files.save_coarse(st["paths"], st["md5s"], st["hu"], st["fp"],
                      st["hu_mean"], st["hu_std"], boxes=st.get("boxes"),
                      sidecar=True)
    if fine is not None:
        files.save_fine(fine["paths"], fine["features"], sidecar=True)
    if progress:
        progress(2, 3, "写入 meta 标记")
    meta["storage"] = "sidecar"
    meta["compacted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    files.save_meta(meta)
    freed = 0
    if delete_legacy:
        for p in (files.coarse_path, files.fine_path):
            if os.path.exists(p):
                freed += os.path.getsize(p)
                os.remove(p)
    if progress:
        progress(3, 3, "完成")
    return {"prefix": prefix, "already": False, "n": n, "has_fine": has_fine,
            "sec": time.time() - t0, "deleted_bytes": freed,
            "fine_rows": int(fine["features"].shape[0]) if fine else 0}


def _ensure_parent(path: str) -> None:
    """索引常被保存到尚不存在的自定义目录（如 GUI 填的新前缀），自动建目录。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def meta_of(cfg: Config, n_coarse: int, db_root: Optional[str],
            has_fine: bool, feature_dim: Optional[int],
            storage: str = "npz") -> dict:
    """把可复现参数写入 meta，供 stats / 未来版本迁移判断使用。"""
    return {
        "db_root": db_root or "",
        "n": n_coarse,
        "storage": storage,
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
