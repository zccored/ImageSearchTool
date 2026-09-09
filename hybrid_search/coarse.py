# -*- coding: utf-8 -*-
"""
第一阶段：二值法粗筛器（纯 CPU，单张特征提取毫秒级）。

特征由两部分组成（均可单独开关，权重可调）：
  1. Hu 不变矩（7 维）  —— 取自二值图上“面积最大轮廓”，经对数变换增强稳定性，
     对平移 / 旋转 / 缩放不敏感，适合“轮廓形状”粗筛。
  2. 二值图像指纹      —— 把 resize 后的二值图展平为 0/1 位图并打包成字节，
     用汉明距离（归一化到 [0,1] 的位差异比例）衡量，适合“布局/明暗结构”粗筛。

两路距离分别做“以第 coarse_k 近邻为基准”的单调尺度归一后按权重融合，
排序结果与原始距离等价（归一化不改变大小关系），并给出稳定的相对分数。

大图库提示：二值指纹打包后每张仅 (size*size)/8 字节，
5000 张 64x64 指纹约 2.6MB；200 万张约 1GB 且全部是 XOR/查表类运算，
将来可无缝换用 faiss IndexBinaryFlat 加速（本项目保持 numpy 零依赖）。
"""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .io_utils import LOGGER, collect_images, decode_gray, read_bytes

# 距离分量枚举：返回值元组中 d 的语义标签（展示用）
HIT_IDX = 0
HIT_PATH = 1
HIT_SCORE = 2
HIT_DHU = 3
HIT_DFP = 4


@dataclass
class CoarseRecord:
    """图库中单张图片的粗筛特征。"""
    path: str
    md5: str = ""
    hu: Optional[np.ndarray] = None          # (7,) 对数变换后的 Hu 矩
    fp: Optional[np.ndarray] = None          # (bytes,) 打包后的二值指纹
    box: Optional[Tuple[float, float, float, float]] = None  # 瓦片框(x0,y0,x1,y1)，整图条目为 None
    ok: bool = True
    err: str = ""


# ---------------------------------------------------------------------------
# 单图特征提取
# ---------------------------------------------------------------------------
def extract_binary_features(gray: np.ndarray, cfg: Config):
    """
    从灰度图直接算粗筛特征（供“融合建库”单遍解码复用，避免重复解码）：
    返回 (binary_64x64, hu|None, fp|None)。异常向调用方抛出。
    """
    binary = _to_binary(gray, cfg)
    hu = _hu_of_binary(binary) if cfg.use_hu else None
    fp = _fp_of_binary(binary, cfg) if cfg.use_fp else None
    return binary, hu, fp


def extract_coarse_single(path: str, cfg: Config,
                          frame_sink=None) -> CoarseRecord:
    """
    读取一张图并提取粗筛特征；失败时 ok=False 并带回原因。
    磁盘只读一次：整文件进内存后，MD5 与图像解码共用同一份字节。

    frame_sink(path, binary_64x64)：可选可视化回调。每成功处理一张图、
    在二值点阵算出后立刻调用（该 64×64 临时数组只活在这一帧，不驻留内存）。
    回调里抛出的任何异常都会被吞掉，绝不拖累索引主流程。
    """
    data = read_bytes(path)
    if data is None:
        return CoarseRecord(path=path, ok=False, err="无法读取文件")
    try:
        gray = decode_gray(data)
        if gray is None:
            return CoarseRecord(path=path, ok=False, err="无法解码")
        binary, hu, fp = extract_binary_features(gray, cfg)
        if frame_sink is not None:
            try:
                frame_sink(path, binary)
            except Exception:  # noqa: BLE001 —— 可视化失败不得影响索引
                pass
        md5 = hashlib.md5(data).hexdigest() if cfg.dedup else ""
        return CoarseRecord(path=path, md5=md5, hu=hu, fp=fp)
    except Exception as e:  # noqa: BLE001 —— 特征提取失败只影响单张图
        return CoarseRecord(path=path, ok=False, err=f"特征提取异常: {e!r}")


def auto_threads(requested: int) -> int:
    """0=自动：不超过 8 且不超过 CPU 核数，避免解码线程挤占内存与调度。"""
    if requested > 0:
        return requested
    return max(1, min(8, (os.cpu_count() or 2)))


def _to_binary(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """灰度 -> 高斯降噪 -> 统一尺寸 -> OTSU 二值化，返回 0/255 的 uint8 图。"""
    # 重型图（长图/全景）先降到 ≤1024 边长再做高斯/OTSU：
    # 指纹网格只有 64×64，提前降采样省几十倍内存与 CPU，不损失判别力
    if gray.size > 8_000_000:
        scale = 1024.0 / max(gray.shape)
        gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)),
                                 max(1, int(gray.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    k = cfg.coarse_blur
    if k > 1 and k % 2 == 0:
        k += 1  # 高斯核必须是奇数
    blurred = cv2.GaussianBlur(gray, (k, k), 0) if k > 1 else gray
    size = cfg.coarse_size
    resized = cv2.resize(blurred, (size, size), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cfg.invert_binary and np.count_nonzero(binary > 127) > binary.size // 2:
        binary = 255 - binary  # 白多则取反，弱化前景/背景极性差异
    return binary


def _hu_of_binary(binary: np.ndarray) -> np.ndarray:
    """二值图 -> 面积最大轮廓的 Hu 矩（7 维）；无轮廓时返回零向量。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7, dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(largest)).flatten()
    # 对数变换：Hu 矩动态范围横跨十几个数量级，直接算距离会被前两维主导
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).astype(np.float32)


def _fp_of_binary(binary: np.ndarray, cfg: Config) -> np.ndarray:
    """二值图 -> 0/1 位图 -> 按字节打包（np.packbits，MSB 在前）。"""
    bits = (binary > 127).reshape(-1)
    n_bits = cfg.coarse_size * cfg.coarse_size
    if bits.size != n_bits:
        raise ValueError(f"指纹位数异常: {bits.size} != {n_bits}")
    return np.packbits(bits.astype(np.uint8))


def _mapped(arr) -> bool:
    """数组（或其视图的 base 链）是否来自 mmap。"""
    a = arr
    for _ in range(8):
        if isinstance(a, np.memmap):
            return True
        a = getattr(a, "base", None)
        if not isinstance(a, np.ndarray):
            return False
    return False


# ---------------------------------------------------------------------------
# 索引集合：构建 / 追加 / 检索
# ---------------------------------------------------------------------------
class CoarseIndex:
    """全库粗筛特征的矩阵容器。矩阵在 finalize() 时一次性拼出，避免逐行拷贝。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths: List[str] = []
        self.md5s: List[str] = []
        self.boxes: List[Optional[tuple]] = []   # 瓦片框；整图/旧索引条目为 None
        # 从磁盘恢复的框矩阵（可 mmap）：不预先转成 Python 元组列表 ——
        # 444k 行元组化要 ~0.4s 且常驻 ~100MB，改为按需解析（box_at）。
        self._boxes_arr: Optional[np.ndarray] = None
        # hu 每维按库内均值/方差做 z-score，避免量纲差异主导距离
        self.hu_mean: np.ndarray = np.zeros(7, dtype=np.float32)
        self.hu_std: np.ndarray = np.ones(7, dtype=np.float32)
        self._hu_parts: List[np.ndarray] = []
        self._fp_parts: List[np.ndarray] = []
        self._hu: Optional[np.ndarray] = None      # (N,7)
        self._fp: Optional[np.ndarray] = None      # (N,bytes)
        self.n_bytes = (cfg.coarse_size * cfg.coarse_size + 7) // 8
        # 增量去重缓存：避免每批 add 都 O(N) 重建集合（实测万级瓦片时
        # 主线程每批耗时随库二次方增长，堵死流水线 → “后期性能下降”）
        self._md5_set: set = set()
        self._path_set: set = set()

    # -- 只读 ----------------------------------------------------------
    @property
    def hu(self) -> Optional[np.ndarray]:
        self.ensure_finalized()
        return self._hu

    @property
    def fp(self) -> Optional[np.ndarray]:
        self.ensure_finalized()
        return self._fp

    @property
    def size(self) -> int:
        return len(self.paths)

    # -- 构建 ----------------------------------------------------------
    def add_images_dir(self, img_dir: str, limit: Optional[int] = None,
                       progress=None, frame_sink=None) -> int:
        """递归扫描目录并追加新图特征（返回新增数）。"""
        paths = collect_images(img_dir, self.cfg.extensions, limit=limit)
        return self.add_paths(paths, progress=progress, frame_sink=frame_sink)

    def add_paths(self, paths: List[str], progress=None,
                  frame_sink=None) -> int:
        """
        追加特征。已知重复内容（md5 相同）会跳过；返回真正新增数。
        progress(done:int, total:int)：每处理完一张回调一次（供 GUI 进度条）。
        frame_sink(path, binary)：可选可视化回调，转发给每张的特征提取
        （worker 线程内同步调用，异常被吞，不影响索引）。
        """
        if not paths:
            return 0
        known_md5s = self._md5_set
        existing = self._path_set
        todo = [p for p in paths
                if os.path.normcase(os.path.abspath(p)) not in existing]
        LOGGER.info("粗筛：扫描到 %d 张，待处理 %d 张", len(paths), len(todo))
        if progress:
            progress(0, len(todo))

        if self.cfg.workers != 1 and auto_threads(self.cfg.workers) > 1:
            records = self._extract_parallel(todo,
                                             auto_threads(self.cfg.workers),
                                             frame_sink)
        else:
            # workers=1 或核数不足时保持串行（最省内存）
            records = [extract_coarse_single(p, self.cfg, frame_sink)
                       for p in todo]

        added = 0
        for done, rec in enumerate(records, 1):
            if not rec.ok:
                LOGGER.warning("跳过 %s：%s", rec.path, rec.err)
                if progress:
                    progress(done, len(todo))
                continue
            if self.cfg.dedup and rec.md5:
                if rec.md5 in known_md5s:
                    LOGGER.debug("去重跳过 %s（内容已存在）", rec.path)
                    if progress:
                        progress(done, len(todo))
                    continue
                known_md5s.add(rec.md5)
            self.paths.append(rec.path)
            self.md5s.append(rec.md5)
            self._path_set.add(os.path.normcase(os.path.abspath(rec.path)))
            self.boxes.append(rec.box)
            if rec.hu is not None:
                self._hu_parts.append(rec.hu.reshape(1, 7))
            if rec.fp is not None:
                self._fp_parts.append(rec.fp.reshape(1, -1))
            added += 1
            if progress:
                progress(done, len(todo))
        LOGGER.info("粗筛：新增 %d 张，库内总计 %d 张", added, self.size)
        return added

    def add_results(self, paths: List[str], records: List,
                    progress=None) -> List[bool]:
        """
        把“已在别处解码并算好特征”的结果入库（融合建库单遍解码流程用）：
        records[i] 对应 paths[i]，可为 None（解码/特征失败）或
        带 .md5/.hu/.fp 的对象。内部做路径 + MD5 去重。

        返回与 paths 等长的布尔掩码：True = 该图被接受入库，
        且接受顺序与 coarse.paths 追加顺序一致（供精排张量对齐过滤）。
        """
        if not paths:
            return []
        known_md5s = self._md5_set
        existing = self._path_set
        accepted: List[bool] = []
        added = 0
        for done, (p, rec) in enumerate(zip(paths, records), 1):
            ok = rec is not None
            # 路径去重只对“整图条目”生效（每路径至多 1 条）；
            # 瓦片条目同一原图有多块同路径，重复性由 md5(含框坐标)判定。
            if ok and rec.box is None and \
                    os.path.normcase(os.path.abspath(p)) in existing:
                ok = False
            if ok and self.cfg.dedup and rec.md5 and rec.md5 in known_md5s:
                ok = False
            if ok:
                known_md5s.add(rec.md5)
                self.paths.append(p)
                self.md5s.append(rec.md5)
                self._path_set.add(os.path.normcase(os.path.abspath(p)))
                self.boxes.append(rec.box)
                if rec.hu is not None:
                    self._hu_parts.append(rec.hu.reshape(1, 7))
                if rec.fp is not None:
                    self._fp_parts.append(rec.fp.reshape(1, -1))
                added += 1
            else:
                LOGGER.debug("融合入库跳过 %s", p)
            accepted.append(ok)
            if progress:
                progress(done, len(paths))
        return accepted

    def finalize(self) -> None:
        """把累积的新分块与既有矩阵合并（既有矩阵来自内存增量或磁盘加载），
        并重算 z-score 参数。绝不能用“只含新分块”的矩阵替换旧数据。"""
        if self._hu_parts:
            new_hu = (np.concatenate(self._hu_parts, axis=0)
                      if len(self._hu_parts) > 1 else self._hu_parts[0])
            self._hu = (new_hu if self._hu is None
                        else np.concatenate([self._hu, new_hu], axis=0))
            self._hu_parts = []
        if self._fp_parts:
            new_fp = (np.concatenate(self._fp_parts, axis=0)
                      if len(self._fp_parts) > 1 else self._fp_parts[0])
            self._fp = (new_fp if self._fp is None
                        else np.concatenate([self._fp, new_fp], axis=0))
            self._fp_parts = []
        self._update_hu_stats()

    def ensure_finalized(self) -> None:
        if self._hu_parts or self._fp_parts:
            self.finalize()

    # -- 从磁盘恢复 / 导出 ----------------------------------------------
    def load_state(self, paths: List[str], md5s: List[str],
                   hu: Optional[np.ndarray], fp: Optional[np.ndarray],
                   hu_mean: np.ndarray, hu_std: np.ndarray,
                   boxes: Optional[np.ndarray] = None) -> None:
        self.paths = list(paths)
        self.md5s = list(md5s)
        self.boxes = []
        self._boxes_arr = None
        if boxes is not None and len(boxes) == len(paths):
            if isinstance(boxes, np.ndarray):
                # 懒解析：保留数组（可能是 mmap），用 box_at() 取单个框
                self._boxes_arr = np.asarray(boxes, dtype=np.float32)
            else:
                self.boxes = [tuple(float(v) for v in b) for b in boxes]
        self._hu = hu
        self._fp = fp
        self.hu_mean = hu_mean
        self.hu_std = hu_std
        self._md5_set = set(m for m in self.md5s if m)
        self._path_set = {os.path.normcase(os.path.abspath(x))
                          for x in self.paths}
        self._hu_parts = []
        self._fp_parts = []

    # -- 瓦片框（数组懒解析 / 列表增量混合）-----------------------------
    def _boxes_n(self) -> int:
        return len(self._boxes_arr) if self._boxes_arr is not None else 0

    def has_boxes(self) -> bool:
        """是否存在瓦片框（整图索引为 False）。"""
        return self._boxes_arr is not None or any(
            b is not None for b in self.boxes)

    def box_at(self, i: int) -> Optional[tuple]:
        n = self._boxes_n()
        if i < n:
            return tuple(float(v) for v in self._boxes_arr[i])
        j = i - n
        if 0 <= j < len(self.boxes):
            return self.boxes[j]
        return None

    def detach_memmaps(self) -> None:
        """把 mmap 恢复态转为内存数组。

        Windows 下被映射的文件无法被替换/删除，落盘（save_coarse）前必须
        先解除映射，否则增量写盘会 PermissionError。
        注意：np.asarray(memmap) 得到的是“视图”而不是 memmap 实例，但视图
        的 .base 链仍持有映射 —— 必须顺着 base 链判断。"""
        for name in ("_hu", "_fp", "_boxes_arr"):
            arr = getattr(self, name)
            if _mapped(arr):
                setattr(self, name, np.array(arr))

    def export_state(self) -> dict:
        self.detach_memmaps()          # 落盘前解除映射（见上）
        self.ensure_finalized()
        n = len(self.paths)
        if self._hu is not None and self._hu.shape[0] != n:
            raise RuntimeError(
                f"内部状态不一致：paths={n} vs hu矩阵={self._hu.shape[0]} 行")
        if self._fp is not None and self._fp.shape[0] != n:
            raise RuntimeError(
                f"内部状态不一致：paths={n} vs 指纹矩阵={self._fp.shape[0]} 行")
        boxes = None
        if not self.boxes:                  # 纯加载态：直接复用数组
            if self._boxes_arr is not None:
                boxes = np.asarray(self._boxes_arr, dtype=np.float32)
        elif self._boxes_arr is None:
            if any(b is not None for b in self.boxes):
                boxes = np.asarray([(b if b is not None else (0, 0, 0, 0))
                                    for b in self.boxes], dtype=np.float32)
        else:                               # 加载后又增量追加：合并
            rows = [self.box_at(i) or (0.0, 0.0, 0.0, 0.0)
                    for i in range(n)]
            boxes = np.asarray(rows, dtype=np.float32)
        return {
            "paths": self.paths, "md5s": self.md5s,
            "hu": self._hu, "fp": self._fp,
            "hu_mean": self.hu_mean, "hu_std": self.hu_std,
            "boxes": boxes,
        }

    # -- 并行 ----------------------------------------------------------
    def _extract_parallel(self, paths: List[str], n_threads: int,
                          frame_sink=None) -> List[CoarseRecord]:
        out = [None] * len(paths)
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = {ex.submit(extract_coarse_single, p, self.cfg, frame_sink):
                    i for i, p in enumerate(paths)}
            for fut, i in futs.items():
                out[i] = fut.result()
        return [r for r in out if r is not None]

    # -- z-score 参数 ----------------------------------------------------
    def _update_hu_stats(self) -> None:
        hu = self._hu
        if hu is None or hu.shape[0] < 2:
            self.hu_mean = np.zeros(7, dtype=np.float32)
            self.hu_std = np.ones(7, dtype=np.float32)
            return
        mu = hu.mean(axis=0)
        sd = hu.std(axis=0)
        sd[sd < 1e-6] = 1.0
        self.hu_mean = mu.astype(np.float32)
        self.hu_std = sd.astype(np.float32)

    # -- 查询 ----------------------------------------------------------
    def query_record(self, q_path: str) -> CoarseRecord:
        rec = extract_coarse_single(q_path, self.cfg)
        if not rec.ok:
            raise RuntimeError(f"查询图无法处理: {q_path} —— {rec.err}")
        return rec

    def coarse_search(self, q_path: str, coarse_k: int, q_rec: Optional[CoarseRecord] = None):
        """
        粗筛检索。返回候选行元组列表：
          (idx, path, score, d_hu_raw, d_fp_raw)
        score = 1 - 融合距离（融合距离越小越相似，score 越大越相似）。
        idx 为候选在库矩阵中的行号，供精排阶段直接按行取 ResNet 特征。
        """
        n = self.size
        if n == 0:
            raise RuntimeError("索引为空，请先 build / add 构建图库索引")
        rec = q_rec if q_rec is not None else self.query_record(q_path)
        t0 = time.time()

        d_hu = None
        if self.cfg.use_hu and self.hu is not None:
            z = lambda a: (a.astype(np.float32) - self.hu_mean) / self.hu_std  # noqa: E731
            d_hu = np.sqrt(np.square(z(self.hu) - z(rec.hu)).sum(axis=1))

        d_fp = None
        if self.cfg.use_fp and self.fp is not None and rec.fp is not None:
            xor = np.bitwise_xor(self.fp, rec.fp)
            d_fp = _hamming_distance(xor) / float(self.n_bytes * 8)

        k_idx = min(max(coarse_k, 1), n - 1)
        combined = np.zeros(n, dtype=np.float64)
        if d_hu is not None:
            combined += self.cfg.hu_weight * _scale01(d_hu, k_idx)
        if d_fp is not None:
            combined += self.cfg.fp_weight * _scale01(d_fp, k_idx)

        order = np.argsort(combined, kind="stable")[:coarse_k]
        LOGGER.debug("粗筛扫描 %.1f ms（库 %d 张）", (time.time() - t0) * 1000, n)
        out = []
        for i in order:
            out.append((int(i), self.paths[i], float(1.0 - combined[i]),
                        float(d_hu[i]) if d_hu is not None else float("nan"),
                        float(d_fp[i]) if d_fp is not None else float("nan")))
        return out


def _scale01(d: np.ndarray, k_idx: int) -> np.ndarray:
    """
    以“第 k_idx 近”为基准做单调尺度归一：s = (d - d_min) / (d_k - d_min + eps)。
    最优 -> 0，第 k_idx 近 -> 1。单调性保证排序与按原始距离完全一致。
    """
    d_min = float(d.min())
    d_k = float(np.sort(d)[k_idx])
    denom = max(d_k - d_min, 1e-9)
    return (d - d_min) / denom


def _hamming_distance(xor: np.ndarray) -> np.ndarray:
    """
    逐行统计打包指纹的置位数（汉明距离）。三条路径：
    1) numpy>=2.0：np.bitwise_count 直接吃 uint64 字，硬件级 SIMD；
    2) 旧版 numpy：SWAR 逐位算法（5 条向量运算）统计 64 位字，
       不再把整库解包成 8 倍大的 uint8 位图（省内存、省带宽）；
    3) 字节数不是 8 的倍数时（非常规指纹边长）才退回整库解包，
       且显式用大整型累加，避免旧版 numpy 的 uint8 回绕风险。
    """
    n_bytes = xor.shape[1]
    if n_bytes % 8 == 0:
        # uint8 每 8 字节成组后 reinterpret 为 uint64（小端/大端位模式等价）
        words = xor.reshape(xor.shape[0], -1, 8).view(np.uint64)
        words = words.reshape(xor.shape[0], -1)
        if hasattr(np, "bitwise_count"):
            return np.bitwise_count(words).sum(axis=1, dtype=np.uint64)
        return _swar_popcount(words).sum(axis=1, dtype=np.uint64)
    bits = np.unpackbits(xor, axis=1)
    return bits.sum(axis=1, dtype=np.uint64)


def _swar_popcount(x: np.ndarray) -> np.ndarray:
    """64 位 SWAR 汉明重量：每字 5 条向量运算，无逐字节解包。"""
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    m8 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * m8) >> np.uint64(56)
