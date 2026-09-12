# -*- coding: utf-8 -*-
"""
混合检索引擎：把“二值法粗筛”与“ResNet 精排”串成一条漏斗式流水线。

数据流：
  查询图
    ├─ [阶段1] 二值法粗筛（Hu 矩 + 二值指纹，纯 CPU 毫秒级）
    │        在全集上算距离 -> 取 top coarse_k 张候选（记录库内行号）
    ├─ [阶段2] ResNet 精排
    │        方式A（有全库索引）：按行号取特征，一次矩阵乘法出余弦分
    │        方式B（无全库索引）：只对候选实时批量抽特征再打分
    └─ 按精排分取 top_k（若无法加载 torch，退化为粗筛分排序并提示）

关键约定：粗筛路径列表与精排索引按“完全相同的顺序、相同的集合”追加，
_open 时做 O(N) 逐位校验，防止历史损坏导致按行取特征错位。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .coarse import CoarseIndex
from .config import Config
from .fine import ResNetExtractor
from .io_utils import LOGGER, collect_images, human_bytes
from .store import IndexFiles, meta_of


def _wrap_frame(frame_sink, phase: str):
    """给可视化帧回调绑阶段标记（None -> None）。"""
    if frame_sink is None:
        return None
    return lambda path, data: frame_sink(path, data, phase)


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    rank: int
    path: str
    fine_score: float               # ResNet 余弦相似度（NaN=未精排）
    coarse_score: float             # 粗筛综合分（1 - 融合距离，越大越相似）
    d_hu: float                     # 原始 Hu 标准化距离
    d_fp: float                     # 原始二值指纹汉明比例
    box: Optional[Tuple[float, float, float, float]] = None  # 瓦片命中框(原图像素)，整图命中为 None
    match_kind: str = "full"        # full / tile（命中类型，混合检索展示用）


@dataclass
class Outcome:
    query: str
    hits: List[Hit] = field(default_factory=list)
    db_size: int = 0
    coarse_kept: int = 0
    coarse_only: bool = False
    self_excluded: bool = False
    times: dict = field(default_factory=dict)   # 各阶段耗时（秒）
    method: str = "full"            # full / tiles / hybrid（检索模式）


# ---------------------------------------------------------------------------
# 阶段边界事件：进度回调除了“计数”，还会收到 save / done 两个明确边界，
#   让 GUI 能在“特征提取完成”与“落盘完成”分别打点（性能图需要确切结束状态）。
# ---------------------------------------------------------------------------
def _phase(progress, done: int, total: int, phase: str) -> None:
    if progress is None:
        return
    try:
        progress(int(done), int(max(total, 1)), phase)
    except Exception:                       # noqa: BLE001 —— 打点失败不影响建库
        pass


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class HybridEngine:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # 让解码层跟随配置的 PNG 解码器（cv2 全尺寸 / pillow）
        from .io_utils import (set_big_decode_limit, set_png_decoder,
                               silence_png_noise)
        set_png_decoder(getattr(cfg, "png_decoder", "cv2"))
        if getattr(cfg, "silence_png_warnings", True):
            silence_png_noise(True)      # 屏蔽 libpng iCCP/cHRM stderr 噪音
        set_big_decode_limit(getattr(cfg, "big_decode_conc", 16))
        self.coarse = CoarseIndex(cfg)
        self.prefix: Optional[str] = None
        self.meta: dict = {}
        # 存储格式：npz（旧，整体读入）或 sidecar（可 mmap 的 .npy，快且省内存）。
        # 新建索引用 cfg.fast_load 决定；已有索引以 meta 标记为准（open 时覆盖）。
        self._storage: str = ("sidecar" if getattr(cfg, "fast_load", False)
                              else "npz")
        # 精排索引（懒加载 mmap）
        self._fine_feats = None          # (N,D) ndarray / memmap（L2 归一化）
        self._fine_npz = None
        # 精排特征提取器（懒加载）
        self._extractor: Optional[ResNetExtractor] = None

    # ==================================================================
    # 索引构建 / 增量 / 重建
    # ==================================================================
    def build(self, prefix: str, img_dir: Optional[str] = None,
              paths: Optional[List[str]] = None, limit: Optional[int] = None,
              force: bool = False, progress=None, frame_sink=None) -> int:
        """
        从零构建整套索引：粗筛必建，精排索引由 cfg.store_fine 决定。
        img_dir / paths 二选一（paths 供 GUI“勾选部分图片建索引”使用）。
        progress(done, total, phase)：phase ∈ {"fused", "coarse", "fine"}，
        需要 ResNet 全库索引时走“融合建库”（fused）：单遍解码同时产出
        粗筛指纹与 ResNet 特征，CPU 解码与 GPU 前向并行推进；
        只建粗筛时 phase="coarse"。
        frame_sink(path, data, phase)：可选可视化帧回调（phase 同上），
        粗筛帧 data=64×64 二值点阵，精排帧 data=16×16×3 RGB 采样象限；
        回调内异常不影响索引主流程。
        """
        files = IndexFiles(prefix)
        if files.meta_exists() and not force:
            raise FileExistsError(
                f"索引 {prefix}.* 已存在；请加 --force 重建，或用 add 增量入库")
        start = time.time()

        if paths is None:
            if not img_dir:
                raise ValueError("需要 img_dir 或 paths 之一")
            paths = collect_images(img_dir, self.cfg.extensions, limit=limit)
        elif limit is not None and limit > 0:
            paths = paths[:limit]

        self._prep_cache = self._make_prep_cache(prefix)
        has_fine = False
        feat_dim = None
        if self.cfg.store_fine:
            # 融合建库：单遍解码 + 解码线程并行产出两类特征
            self._get_extractor()            # 提前初始化模型（错误尽早暴露）
            fused_prog = self._fused_cb(progress)
            ok, feats = self._fused_ingest(paths, progress=fused_prog,
                                           frame_sink=frame_sink)
            if feats is None or len(ok) == 0:
                raise RuntimeError(
                    "没有可入库的图片（检查目录、扩展名或图片完整性）")
            if ok != self.coarse.paths:
                raise RuntimeError(
                    "融合建库内部顺序不一致（粗筛与精排结果集错位），请联系排查")
            n_done = len(ok)
            LOGGER.info("特征提取完成：%d 张（粗筛指纹 + ResNet 特征，单遍解码）",
                        n_done)
            _phase(progress, n_done, n_done, "save")   # 明确的阶段边界
            files.save_fine(self.coarse.paths, feats,
                            sidecar=self._storage == "sidecar")
            self._keep_fine(feats)
            has_fine = True
            feat_dim = feats.shape[1]
            n = self.coarse.size
        else:
            n = self.coarse.add_paths(
                paths, progress=self._coarse_cb(progress),
                frame_sink=_wrap_frame(frame_sink, "coarse"))
            if n == 0:
                raise RuntimeError(
                    "没有可入库的图片（检查目录、扩展名或图片完整性）")
            _phase(progress, n, n, "save")

        self._save_all(files, has_fine=has_fine, feature_dim=feat_dim)
        _phase(progress, self.coarse.size, self.coarse.size, "done")
        self._log_prep_cache()
        LOGGER.info("索引构建完成：%d 张，耗时 %.1f s，输出前缀 %s.*",
                    self.coarse.size, time.time() - start, prefix)
        return self.coarse.size

    def add(self, prefix: str, img_dir: Optional[str] = None,
            paths: Optional[List[str]] = None, limit: Optional[int] = None,
            progress=None, frame_sink=None) -> int:
        """增量入库：仅处理新路径/新内容（md5 去重）。img_dir / paths 二选一。
        progress / frame_sink 语义同 build（新图入库同样走融合单遍解码）。"""
        self.open(prefix)
        if paths is None:
            if not img_dir:
                raise ValueError("需要 img_dir 或 paths 之一")
            paths = collect_images(img_dir, self.cfg.extensions, limit=limit)
        elif limit is not None and limit > 0:
            paths = paths[:limit]
        self._prep_cache = self._make_prep_cache(prefix)
        old_n = self.coarse.size
        old_has_fine = self._fine_feats is not None
        start = time.time()

        if old_has_fine or self.cfg.store_fine:
            # 新图路径过滤（已在库的路径无需重新解码）
            existing = {os.path.normcase(os.path.abspath(x))
                        for x in self.coarse.paths}
            todo = [p for p in paths
                    if os.path.normcase(os.path.abspath(p)) not in existing]
            if not todo:
                LOGGER.info("无新增图片（路径与内容均重复）")
                return 0
            self._get_extractor()
            ok, new_feats = self._fused_ingest(
                todo, progress=self._fused_cb(progress),
                frame_sink=frame_sink)
            if new_feats is None or len(ok) == 0:
                LOGGER.info("无新增图片（内容均重复）")
                return 0
            added = len(ok)
            if self.coarse.paths[old_n:] != ok:
                raise RuntimeError("融合增量内部顺序不一致，请联系排查")
            LOGGER.info("特征提取完成：新增 %d 张", added)
            _phase(progress, added, added, "save")
            if old_has_fine:
                new_feats = np.concatenate(
                    [np.asarray(self._fine_feats), new_feats], axis=0)
            self.release_fine()            # 解除旧 mmap（Windows 无法替换被映射文件）
            self._keep_fine(new_feats)
            IndexFiles(prefix).save_fine(self.coarse.paths, new_feats,
                                         sidecar=self._storage == "sidecar")
            self._save_all(IndexFiles(prefix), has_fine=True,
                           feature_dim=new_feats.shape[1])
        else:
            added = self.coarse.add_paths(
                paths, progress=self._coarse_cb(progress),
                frame_sink=_wrap_frame(frame_sink, "coarse"))
            if added == 0:
                LOGGER.info("无新增图片（路径与内容均重复）")
                return 0
            _phase(progress, added, added, "save")
            self._save_all(IndexFiles(prefix), has_fine=False)
        _phase(progress, self.coarse.size, self.coarse.size, "done")
        self._log_prep_cache()
        LOGGER.info("增量入库完成：+%d 张（总计 %d），耗时 %.1f s",
                    added, self.coarse.size, time.time() - start)
        return added

    # ------------------------------------------------------------------
    # 预处理缓存（L2 内存 + L3 磁盘）
    # ------------------------------------------------------------------
    def _make_prep_cache(self, prefix: str):
        """按索引前缀创建/复用预处理缓存；关闭时返回 None。"""
        if not getattr(self.cfg, "prep_cache", True):
            try:
                from .fine import set_prep_cache
                set_prep_cache(None)
            except Exception:               # noqa: BLE001
                pass
            return None
        from .fine import _PRE_DOWNSCALE_SIDE, set_prep_cache
        from .prep_cache import PrepCache, default_cache_dir
        root = default_cache_dir(prefix)
        cache = getattr(self, "_prep_cache", None)
        if cache is None or cache.root != root or not cache.enabled:
            cache = PrepCache(root, model=self.cfg.model,
                              pre_side=_PRE_DOWNSCALE_SIDE)
        set_prep_cache(cache)               # 让 build-fine / 查询路径也能命中
        return cache

    def _log_prep_cache(self) -> None:
        cache = getattr(self, "_prep_cache", None)
        if cache is not None:
            LOGGER.info("%s", cache.summary())

    # ------------------------------------------------------------------
    # 融合建库（单遍解码：解码一次，同时出粗筛指纹 + ResNet 张量）
    # ------------------------------------------------------------------
    @staticmethod
    def _fused_cb(progress):
        if progress is None:
            return None
        return lambda done, total: progress(done, total, "fused")

    def _fused_ingest(self, paths: List[str], progress=None,
                      frame_sink=None):
        """
        核心融合流程：
          解码线程池（daemon，数量由 decode_workers 控制）逐张执行
          “读盘→解码 RGB→灰度→二值指纹(Hu+打包)→(可选帧回调)→中心窗口→
          torchvision 预处理张量”，每批解码完成后：
            1) 粗筛特征按序入库（路径+MD5 去重），返回接受掩码；
            2) 仅对“被接受”的张量批量 GPU 前向 → L2 归一化特征行；
        返回 (入库路径列表与粗筛一致顺序, 归一化特征矩阵|None)。
        期间 CPU 解码与 GPU 前向双缓冲并行，硬件不空等。
        """
        ex = self._get_extractor()
        prep = self._make_fused_prep(ex, frame_sink)

        def on_batch(ok_paths, tensors, payloads):
            # 1) 粗筛特征入库（结果顺序 = 追加顺序）
            accepted = self.coarse.add_results(ok_paths, payloads)
            keep = [i for i, a in enumerate(accepted) if a]
            if not keep:
                return [], None
            sub_ok = [ok_paths[i] for i in keep]
            sub_ts = [tensors[i] for i in keep]
            # 2) 只对新增图做 GPU 前向
            feats = ex._forward(sub_ts)              # noqa: SLF001 —— 同模块协作
            if feats is None:
                return [], None
            rows = feats.astype(np.float32)
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            rows /= norms
            return sub_ok, rows

        return ex.stream_decode(paths, prep=prep, progress=progress,
                                on_batch=on_batch)

    def _make_fused_prep(self, ex, frame_sink=None):
        """
        融合解码工作项：一次读盘 + 一次解码，产出 ResNet 张量与粗筛特征。
        返回 prep(path) -> (tensor|None, CoarseRecord|None)。
        """
        import cv2
        import hashlib

        from PIL import Image

        from .coarse import CoarseRecord, extract_binary_features
        from .fine import (_PRE_DOWNSCALE_PX, _PRE_DOWNSCALE_SIDE,
                           _center_quad_sample)
        from .io_utils import decode_rgb, read_bytes

        cfg = self.cfg
        # 预处理缓存（L2 内存 + L3 磁盘）：命中即跳过
        # “读原图(数 MB) + md5 + 解码(18~600ms) + 粗筛特征 + PIL 预处理”，
        # 只需 ~1.5ms 重建张量，且与全新建库**逐位一致**（PNG 无损）。
        cache = getattr(self, "_prep_cache", None)

        def prep(path):
            if cache is not None:
                hit = cache.get(path)
                if hit is not None:
                    return hit
            data = read_bytes(path)
            if data is None:
                return None, None
            try:
                rgb = decode_rgb(data)
                if rgb is None:
                    return None, None
                # A2：先统一降采样到 ≤2048 边长（重型图）再做后续所有处理，
                # 避免对全尺寸像素分别做灰度转换/二值化/Resize：
                #   大 PNG 130MP 一次 resize 后，gray/指纹/quad/tensor 全部
                #   在 ~2048 空间进行（粗筛指纹与 ResNet 224 中心窗语义不变）
                if rgb.shape[0] * rgb.shape[1] > _PRE_DOWNSCALE_PX:
                    scale = _PRE_DOWNSCALE_SIDE / max(rgb.shape[:2])
                    rgb = cv2.resize(
                        rgb, (max(1, int(rgb.shape[1] * scale)),
                              max(1, int(rgb.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                binary, hu, fp = extract_binary_features(gray, cfg)
                md5 = hashlib.md5(data).hexdigest() if cfg.dedup else ""
                if frame_sink is not None:
                    try:
                        frame_sink(path, binary, "coarse")
                    except Exception:  # noqa: BLE001 —— 可视化失败不影响任务
                        pass
                try:
                    if frame_sink is not None:
                        quad = _center_quad_sample(rgb)
                        try:
                            frame_sink(path, quad, "fine")
                        except Exception:  # noqa: BLE001
                            pass
                    tensor = ex.transform(Image.fromarray(rgb))
                except Exception:  # noqa: BLE001 —— 前向预处理失败仅丢该图
                    tensor = None
                if tensor is None:
                    return None, None
                rec = CoarseRecord(path=path, md5=md5, hu=hu, fp=fp)
                if cache is not None:
                    cache.put(path, tensor, rec, ex._mean, ex._std)
                return tensor, rec
            except Exception as e:  # noqa: BLE001 —— 单张失败不影响整体
                LOGGER.debug("融合解码失败 %s: %r", path, e)
                return None, None

        return prep

    def build_fine(self, prefix: str, limit: Optional[int] = None,
                   progress=None, frame_sink=None) -> int:
        """对已存在的粗筛索引补建/重建精排全库索引。
        progress / frame_sink 语义同 build（本操作只含 phase="fine"）。"""
        self.open(prefix)
        subset = self.coarse.paths
        if limit is not None:
            subset = subset[:limit]
        ok, feats = self._extract_fine(subset, progress=self._fine_cb(progress),
                                       frame_sink=_wrap_frame(frame_sink,
                                                              "fine"))
        if len(ok) != len(subset):
            raise RuntimeError(f"精排只成功 {len(ok)}/{len(subset)} 张，中止写入")
        if limit is not None:
            # 只允许在“全量重建”语义下使用 limit（写回仍按全库顺序，只重建子集需谨慎）
            raise RuntimeError("build-fine 暂不支持 --limit（会破坏顺序一致性）")
        # 释放旧 fine 的 mmap（Windows 下被映射的 .npy 无法被替换）
        LOGGER.info("精排特征提取完成：%d 张", len(ok))
        _phase(progress, len(ok), len(ok), "save")
        self.release_fine()
        IndexFiles(prefix).save_fine(self.coarse.paths, feats,
                                     sidecar=self._storage == "sidecar")
        self._keep_fine(feats)
        meta = self.meta
        meta["fine"] = {"exists": True, "feature_dim": feats.shape[1]}
        meta["storage"] = self._storage
        IndexFiles(prefix).save_meta(meta)
        _phase(progress, len(ok), len(ok), "done")
        LOGGER.info("精排索引已保存：%d 张 × %d 维 -> %s",
                    feats.shape[0], feats.shape[1],
                    IndexFiles(prefix).fine_path)
        return feats.shape[0]

    # ------------------------------------------------------------------
    # 内部：进度回调包装 / 精排抽取
    # ------------------------------------------------------------------
    @staticmethod
    def _coarse_cb(progress):
        """把外部进度回调绑上 coarse 阶段标记（None -> None）。"""
        if progress is None:
            return None
        return lambda done, total: progress(done, total, "coarse")

    @staticmethod
    def _fine_cb(progress):
        if progress is None:
            return None
        return lambda done, total: progress(done, total, "fine")

    def _extract_fine(self, paths: List[str],
                      progress=None, frame_sink=None) -> Tuple[List[str], np.ndarray]:
        """抽取 ResNet 特征，返回 (成功的路径列表, 归一化特征矩阵)。"""
        extractor = self._get_extractor()
        return extractor.extract_batch(paths, progress=progress,
                                       frame_sink=frame_sink)

    def _keep_fine(self, feats: np.ndarray) -> None:
        self._fine_feats = feats
        self._fine_npz = None

    def release_fine(self) -> None:
        """解除精排特征的 mmap 引用（含 LSH 表缓存里的引用）。

        Windows 下被映射的 .npy 无法被替换/删除，落盘前必须调用；
        粗筛侧由 CoarseIndex.detach_memmaps() 负责。"""
        self._fine_feats = None
        self._fine_npz = None
        cache = getattr(self, "_lsh_cache", None)
        if cache:
            cache.clear()

    def _save_all(self, files: IndexFiles, has_fine: bool,
                  feature_dim: Optional[int] = None) -> None:
        st = self.coarse.export_state()
        side = self._storage == "sidecar"
        files.save_coarse(st["paths"], st["md5s"], st["hu"], st["fp"],
                          st["hu_mean"], st["hu_std"], boxes=st.get("boxes"),
                          sidecar=side)
        meta = meta_of(self.cfg, self.coarse.size,
                       os.path.dirname(files.coarse_path), has_fine, feature_dim,
                       storage=self._storage)
        files.save_meta(meta)
        fine_txt = (f"，精排 {human_bytes(files.fine_size())}"
                    if has_fine and files.fine_exists() else "")
        LOGGER.info("索引落盘 %s.*（粗筛 %s%s）", files.prefix,
                    human_bytes(files.coarse_size() or 0), fine_txt)
        from .io_utils import stderr_noise_stats
        _st = stderr_noise_stats()
        if _st.get("suppressed"):
            LOGGER.info("已屏蔽 %d 条 libpng/iCCP stderr 噪音（不影响解码结果）",
                        _st["suppressed"])

    # ==================================================================
    # 打开索引
    # ==================================================================
    def open(self, prefix: str) -> None:
        files = IndexFiles(prefix)
        if not files.meta_exists():
            raise FileNotFoundError(f"索引 {prefix}.* 不存在，请先 build")
        self.prefix = prefix
        self.meta = files.load_meta()
        self._check_cfg_compat(self.meta.get("cfg", {}))
        # 存储格式以索引自身标记为准（旧 npz 索引照常可读）
        self._storage = files.storage_of(self.meta)

        st = files.load_coarse(sidecar=self._storage == "sidecar")
        n_paths = len(st["paths"])
        if (st["hu"] is not None and st["hu"].shape[0] != n_paths) or \
                (st["fp"] is not None and st["fp"].shape[0] != n_paths):
            raise RuntimeError(
                "粗筛索引文件损坏或来自旧版本（路径与特征行数不一致）；"
                "请重新 build")
        self.coarse.load_state(st["paths"], st["md5s"], st["hu"], st["fp"],
                               st["hu_mean"], st["hu_std"],
                               boxes=st.get("boxes"))
        self._fine_feats = None
        self._fine_npz = None
        if files.fine_exists(sidecar=self._storage == "sidecar"):
            fine = files.load_fine(mmap=True,
                                   sidecar=self._storage == "sidecar")
            feats = fine["features"]
            fine_paths = list(fine["paths"])
            if fine_paths != self.coarse.paths:
                raise RuntimeError(
                    "粗筛与精排索引路径不一致（历史损坏/被手动改动？），"
                    "请执行 build-fine 重建精排索引")
            if fine_paths and feats.shape[0] == 0:
                raise RuntimeError("精排索引为空文件，请执行 build-fine 重建")
            self._fine_feats = feats
        LOGGER.info("索引已加载：%d 张 <- %s.*（存储 %s）",
                    self.coarse.size, prefix, self._storage)

    def _check_cfg_compat(self, stored: dict) -> None:
        """
        与索引“几何/语义”强相关的参数必须和建库时一致，否则直接拒绝打开；
        融合权重等只影响查询排序的可调项允许不同（仅提示）。
        """
        fatal_keys = ("coarse_size", "coarse_blur", "use_hu", "use_fp",
                      "invert_binary", "png_decoder")
        soft_keys = ("hu_weight", "fp_weight")
        problems = []
        for k in fatal_keys:
            if k in stored and getattr(self.cfg, k, None) != stored[k]:
                problems.append(f"{k}: 索引={stored[k]} 当前={getattr(self.cfg, k, None)}")
        if problems:
            raise RuntimeError(
                "当前参数与建库参数不一致，距离语义会错乱，拒绝打开："
                + "；".join(problems)
                + "。请改用与 build 时一致的参数，或重新 build")
        for k in soft_keys:
            if k in stored and getattr(self.cfg, k, None) != stored[k]:
                LOGGER.info("融合权重与建库时不同（%s：索引=%s 当前=%s），"
                            "仅影响本次排序权重", k, stored[k],
                            getattr(self.cfg, k, None))
        # 精排索引存在时，模型必须一致，否则查询特征维度与库特征对不上
        if (self.meta.get("cfg", {}).get("model")
                and self.meta.get("fine", {}).get("exists")):
            stored_model = self.meta["cfg"]["model"]
            if stored_model != self.cfg.model:
                raise RuntimeError(
                    f"精排索引由 {stored_model} 构建，当前模型为 {self.cfg.model}；"
                    "请加 --model " + stored_model + " 或执行 build-fine 重建")

    # ==================================================================
    # 两级检索
    # ==================================================================
    def search(self, q_path: str, coarse_k: Optional[int] = None,
               top_k: Optional[int] = None) -> Outcome:
        if self.prefix is None or self.coarse.size == 0:
            raise RuntimeError("引擎未打开有效索引，请先 open(prefix)")
        coarse_k = coarse_k or self.cfg.coarse_k
        top_k = top_k or self.cfg.top_k
        if coarse_k < top_k:
            raise ValueError(f"coarse_k({coarse_k}) 必须 >= top_k({top_k})")

        t_all = time.time()
        times: dict = {}
        out = Outcome(query=q_path, db_size=self.coarse.size)
        q_abs = os.path.normcase(os.path.abspath(q_path))

        # ---- 阶段 1：二值法粗筛 ------------------------------------
        t0 = time.time()
        cand = self.coarse.coarse_search(q_path, coarse_k)
        times["粗筛(特征+扫描)"] = time.time() - t0

        # 若查询图就在库中，其自身必然是第 0 位候选：记录并剔除
        if cand:
            first_path = os.path.normcase(os.path.abspath(cand[0][1]))
            if self.cfg.exclude_self and first_path == q_abs:
                out.self_excluded = True
                cand = cand[1:]
        out.coarse_kept = len(cand)
        if not cand:
            LOGGER.warning("粗筛无候选，检查查询图与库内容差异")
            out.times = {"total": time.time() - t_all}
            return out

        # ---- 阶段 2：ResNet 精排 ------------------------------------
        if self._fine_feats is not None:
            t0 = time.time()
            idxs = np.asarray([c[0] for c in cand], dtype=np.int64)
            rows = np.asarray(self._fine_feats[idxs], dtype=np.float32)
            q = self._query_fine(q_path)
            if q is None:
                out.coarse_only = True
            else:
                qn = q / max(float(np.linalg.norm(q)), 1e-8)
                scores = rows @ qn
                order = np.argsort(scores, kind="stable")[::-1][:top_k]
                for rank, pos in enumerate(order, 1):
                    c = cand[int(pos)]
                    out.hits.append(Hit(rank=rank, path=c[1],
                                        fine_score=float(scores[int(pos)]),
                                        coarse_score=c[2], d_hu=c[3], d_fp=c[4]))
                times["精排(查询特征+打分)"] = time.time() - t0
        else:
            out.coarse_only = True

        # ---- 兜底：无精排时按粗筛分返回 ------------------------------
        if out.coarse_only:
            LOGGER.info("本次未做 ResNet 精排（无全库精排索引），按粗筛分排序返回")
            for rank, c in enumerate(cand[:top_k], 1):
                out.hits.append(Hit(rank=rank, path=c[1],
                                    fine_score=float("nan"),
                                    coarse_score=c[2], d_hu=c[3], d_fp=c[4]))
        times["total"] = time.time() - t_all
        out.times = times
        return out

    def _query_fine(self, q_path: str) -> Optional[np.ndarray]:
        """查询图的 ResNet 特征（L2 归一化一维向量，走单张同步快捷路径）。"""
        feat = self._get_extractor().extract_one(q_path)
        if feat is None:
            LOGGER.error("查询图精排特征提取失败: %s", q_path)
            return None
        return feat

    def _get_extractor(self) -> ResNetExtractor:
        if self._extractor is None:
            try:
                self._extractor = ResNetExtractor(self.cfg)
            except Exception as e:  # noqa: BLE001
                hint = ""
                if self.cfg.device == "cuda":
                    hint = ("（设备被设为 cuda 但当前环境无法使用；"
                            "可把设备改为 auto/cpu 后重试，"
                            "或安装 CUDA 版 torch）")
                elif "No module named 'torch" in repr(e):
                    hint = "（请先安装 torch 与 torchvision）"
                raise RuntimeError(
                    f"加载 ResNet 失败{hint}: {e!r}") from e
        return self._extractor

    # ==================================================================
    # 统计
    # ==================================================================
    def stats(self) -> dict:
        if self.prefix is None:
            raise RuntimeError("未打开索引")
        st = self.coarse.export_state()
        files = IndexFiles(self.prefix)
        return {
            "prefix": self.prefix,
            "n": self.coarse.size,
            "dedup": self.cfg.dedup,
            "features": {
                "hu": bool(self.cfg.use_hu),
                "fingerprint": bool(self.cfg.use_fp),
                "fp_bytes_per_image": self.coarse.n_bytes,
            },
            "fine": {
                "exists": self._fine_feats is not None,
                "rows": self._fine_feats.shape[0] if self._fine_feats is not None else 0,
                "dim": self._fine_feats.shape[1] if self._fine_feats is not None else 0,
            },
            "files": {
                "meta": files.meta_path,
                "coarse_size": human_bytes(files.coarse_size() or 0),
                "fine_size": (human_bytes(files.fine_size())
                              if files.fine_exists() else "-"),
            },
        }
