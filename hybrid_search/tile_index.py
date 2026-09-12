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
瓦片局部索引（tile index）：把大幅原图切成瓦片分别建库，实现“裁切图/局部图
搜原图 + 命中位置框”。

索引协议（独立前缀，与整图索引并存于 .gallery_index 下，例如：
  gallery        —— 整图索引（每图 1 条，原流程不变）
  gallery_tiles  —— 瓦片索引（本模块）
每条瓦片 = 一个条目：粗筛指纹(hu+fp) + ResNet 特征 + 原图路径 + 原图像素框 box。
小图（较短边 < min_side）不切，1 条整图框入库 —— 保证任意尺寸查询都有覆盖。

检索管线（三级漏斗，供局部/混合模式）：
  1) LSH 近似候选   —— 对瓦片 ResNet 特征做多表随机投影（SRP 式 LSH），
                       亚秒内取“近似最近邻”候选瓦片行（概率性召回，不保证全）；
  2) hash 特征查验 —— 对候选瓦片做二值指纹 Hamming 复核（64×64 打包指纹），
                       是“近似概率”内容查验：把 LSH 漏召/误召按内容哈希校准，
                       过滤后按原图聚合取组内最优瓦片；
  3) 整图切片收敛   —— 候选原图的全部瓦片特征切片与查询特征做精确余弦
                       （GPU matmul，CPU 回退），组内 max 得原图分 + 最优框。
性能画像（建库：解码每图一次，瓦片在解码图上复用；检索：LSH+复核为 CPU，
收敛精排为 GPU/CPU 矩阵运算）由 perfscope 局部索引档位量化。

注：LSH 是近似检索，召回以“概率”计；第 2/3 级保证最终排名的准确度。
"""
from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .coarse import CoarseRecord, extract_binary_features
from .engine import Hit, Outcome
from .io_utils import LOGGER, collect_images, decode_rgb, read_bytes
from .store import IndexFiles, meta_of

# ---------------------------------------------------------------------------
# 瓦片协议默认值（建库时固化进 meta，增量恢复用同一组参数）
# ---------------------------------------------------------------------------
TILE_DEFAULT = 512          # 瓦片边长（px，处理图空间）
OVERLAP_DEFAULT = 0.25      # 相邻瓦片重叠比例
MIN_SIDE_DEFAULT = 768      # 较短边低于该值的图不切（整图 1 块入库）
PRE_MAX_SIDE = 2048         # 解码后先统一缩放到 ≤2048 长边再做瓦片（超大图）
TILES_META_KEY = "tiles"    # meta.json 顶层键：{"tile","overlap","min_side",
#                             "pre_max","kind":"tile"} 存在即瓦片索引


def _axis_offsets(size: int, tile: int, stride: int) -> List[int]:
    """一维滑窗偏移：覆盖 [0,size)，最后一块贴边（去掉近重复尾块）。"""
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile, stride))
    last = size - tile
    if not xs or last - xs[-1] > 0:
        if xs and last - xs[-1] < max(1, stride // 2):
            xs[-1] = last               # 尾块与上一块几乎重合：贴边代替
        else:
            xs.append(last)
    return xs


def tile_grid(w: int, h: int, tile: int = TILE_DEFAULT,
              overlap: float = OVERLAP_DEFAULT) -> List[Tuple[int, int, int, int]]:
    """返回瓦片框列表 (x0,y0,x1,y1)，坐标在传入的 (w,h) 图像空间。"""
    stride = max(1, int(round(tile * (1.0 - overlap))))
    out = []
    for y0 in _axis_offsets(h, tile, stride):
        for x0 in _axis_offsets(w, tile, stride):
            out.append((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
    return out


def tiles_of_rgb(rgb: np.ndarray, tile: int, overlap: float,
                 min_side: int, pre_max: int) -> Tuple[np.ndarray, List[tuple], float]:
    """
    解码图 -> (处理图(≤pre_max 长边), 瓦片框列表(原图像素空间), 缩放比)。
    较短边 < min_side 的图只返回整图单块；坐标始终为原图像素。
    """
    h, w = rgb.shape[:2]
    if max(w, h) < min_side:
        return rgb, [(0, 0, w, h)], 1.0
    work = rgb
    scale = 1.0
    if max(w, h) > pre_max:
        scale = pre_max / float(max(w, h))
        work = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]
    grid = tile_grid(ww, wh, tile, overlap)
    inv = 1.0 / scale
    boxes = [(max(0, int(x0 * inv)), max(0, int(y0 * inv)),
              min(w, int(round(x1 * inv))), min(h, int(round(y1 * inv))))
             for (x0, y0, x1, y1) in grid]
    return work, boxes, scale


def _tile_md5(data: bytes, box: tuple) -> str:
    """瓦片去重标识：原文件内容 + 框坐标（同图同布局稳定；重复文件同布局自动去重）。"""
    return hashlib.md5(data + f"|{box[0]},{box[1]},{box[2]},{box[3]}".encode()
                       ).hexdigest()


def _decode_to_tiles(path: str, cfg, tile: int, overlap: float,
                     min_side: int, pre_max: int,
                     transform, frame_sink=None,
                     ) -> List[Tuple[np.ndarray, CoarseRecord]]:
    """读盘一次 -> 每瓦片 (tensor, CoarseRecord(带 box))；失败返回 []。
    frame_sink(path, data, phase)：低开销过程可视化帧 ——
      coarse 帧=每瓦片 64×64 二值点阵（已是提取副产物，零额外计算）；
      fine  帧=每文件首个瓦片的 16×16 RGB 象限采样（节流，避免逐瓦片开销）。
    """
    data = read_bytes(path)
    if data is None:
        return []
    try:
        rgb = decode_rgb(data)
        if rgb is None:
            return []
        work, boxes, _scale = tiles_of_rgb(rgb, tile, overlap, min_side, pre_max)
        out = []
        for bi, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            # 处理图坐标（缩放后）与 box 的对应：直接把处理图按比例裁块
            sx0, sy0, sx1, sy1 = (int(round(x0 * _scale)), int(round(y0 * _scale)),
                                  max(int(round(x1 * _scale)), int(round(x0 * _scale)) + 1),
                                  max(int(round(y1 * _scale)), int(round(y0 * _scale)) + 1))
            sw, sh = work.shape[1], work.shape[0]
            sx0, sy0 = max(0, sx0), max(0, sy0)
            sx1, sy1 = min(sw, sx1), min(sh, sy1)
            if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                continue
            tile_rgb = work[sy0:sy1, sx0:sx1]
            gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
            binary, hu, fp = extract_binary_features(gray, cfg)
            md5 = _tile_md5(data, box) if cfg.dedup else ""
            rec = CoarseRecord(path=path, md5=md5, hu=hu, fp=fp, box=box)
            try:
                tensor = transform(Image.fromarray(tile_rgb))
            except Exception:            # noqa: BLE001
                continue
            if frame_sink is not None:
                try:
                    frame_sink(path, binary, "coarse")   # 免费副产物
                    if bi == 0:
                        from .fine import _center_quad_sample  # noqa: PLC0415
                        frame_sink(path, _center_quad_sample(tile_rgb), "fine")
                except Exception:            # noqa: BLE001 —— 可视化失败无碍
                    pass
            out.append((tensor, rec))
        return out
    except Exception as e:               # noqa: BLE001 —— 单图失败不影响整体
        LOGGER.debug("瓦片解码失败 %s: %r", path, e)
        return []


# ---------------------------------------------------------------------------
# 瓦片化流水线（解码一次 -> 多瓦片 -> 批量前向，CPU/GPU 重叠）
# ---------------------------------------------------------------------------
class BuildTrace:
    """低开销建库轨迹（线程安全；仅记录时间戳/计数，不在热路径做事）：
      files  : 每张原图 {t0,t1(解码到瓦片就绪), tiles, bytes, path}
      batches: 每次 GPU 前向 {t0,t1,rows}
    用途：与 CPU/GPU 利用率采样对齐，诊断“空窗/突发”类利用率锯齿。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.files: List[dict] = []
        self.batches: List[dict] = []
        self.t_wall0 = time.monotonic()

    def on_file(self, t0: float, t1: float, tiles: int,
                nbytes: int, path: str) -> None:
        with self._lock:
            self.files.append({"t0": t0 - self.t_wall0, "t1": t1 - self.t_wall0,
                               "tiles": tiles, "bytes": nbytes,
                               "path": os.path.basename(path)})

    def on_batch(self, t0: float, t1: float, rows: int) -> None:
        with self._lock:
            self.batches.append({"t0": t0 - self.t_wall0, "t1": t1 - self.t_wall0,
                                 "rows": rows})

    def snapshot(self) -> dict:
        with self._lock:
            return {"files": list(self.files), "batches": list(self.batches)}


def _decode_to_crops(path: str, cfg, tile: int, overlap: float,
                      min_side: int, pre_max: int
                      ) -> List[Tuple[np.ndarray, tuple, bytes, bool]]:
    """一级：读盘一次 -> 解码(≤pre_max 处理图) -> 切块。
    返回 [(瓦片 RGB 副本, 原图框, 原始文件字节(供 MD5), 是否本图首块)]。
    特征提取不在此做 —— 交给全池并行的二级任务，消除单图串行长尾。"""
    data = read_bytes(path)
    if data is None:
        return []
    try:
        rgb = decode_rgb(data)
        if rgb is None:
            return []
        work, boxes, _scale = tiles_of_rgb(rgb, tile, overlap, min_side, pre_max)
        out = []
        for bi, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            sx0, sy0, sx1, sy1 = (int(round(x0 * _scale)), int(round(y0 * _scale)),
                                  max(int(round(x1 * _scale)),
                                      int(round(x0 * _scale)) + 1),
                                  max(int(round(y1 * _scale)),
                                      int(round(y0 * _scale)) + 1))
            sw, sh = work.shape[1], work.shape[0]
            sx0, sy0 = max(0, sx0), max(0, sy0)
            sx1, sy1 = min(sw, sx1), min(sh, sy1)
            if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                continue
            out.append((np.ascontiguousarray(work[sy0:sy1, sx0:sx1]),
                        box, data, bi == 0))
        return out
    except Exception as e:               # noqa: BLE001 —— 单图失败不影响整体
        LOGGER.debug("瓦片解码失败 %s: %r", path, e)
        return []


def _feature_one_tile(path: str, rgb_crop: np.ndarray, box: tuple,
                      data: bytes, first: bool, cfg, transform,
                      frame_sink=None
                      ) -> Optional[Tuple[np.ndarray, CoarseRecord]]:
    """二级：单瓦片特征（粗筛指纹+Hu + ResNet 预处理张量）。失败返回 None。"""
    try:
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        binary, hu, fp = extract_binary_features(gray, cfg)
        md5 = _tile_md5(data, box) if cfg.dedup else ""
        rec = CoarseRecord(path=path, md5=md5, hu=hu, fp=fp, box=box)
        try:
            tensor = transform(Image.fromarray(rgb_crop))
        except Exception:                # noqa: BLE001
            return None
        if frame_sink is not None:
            try:
                frame_sink(path, binary, "coarse")     # 免费副产物
                if first:
                    from .fine import _center_quad_sample  # noqa: PLC0415
                    frame_sink(path, _center_quad_sample(rgb_crop), "fine")
            except Exception:                # noqa: BLE001 —— 可视化失败无碍
                pass
        return tensor, rec
    except Exception as e:               # noqa: BLE001
        LOGGER.debug("瓦片特征失败 %s: %r", path, e)
        return None


def _ingest_tiles(engine, prefix: str, todo: List[str], progress=None,
                  frame_sink=None, tile: int = TILE_DEFAULT,
                  overlap: float = OVERLAP_DEFAULT,
                  min_side: int = MIN_SIDE_DEFAULT,
                  pre_max: int = PRE_MAX_SIDE,
                  trace: Optional[BuildTrace] = None) -> int:
    """把 todo(图片路径列表) 切成瓦片并入 engine（coarse 必须已 open/为空）。
    返回新增瓦片条数；完成后 coarse/fine/meta 已落盘。

    两级流水（解码/切块 -> 瓦片特征全池并行 -> 流式 GPU 前向）：
      * 任务 = (img: 解码+切块) 或 (tile: 指纹/Hu/ResNet 预处理)，同一 worker
        池竞争——大图瓦片特征不再由单个 worker 串行（实测单块≈4ms，
        JPEG 型单图特征≈90ms 高于解码≈40ms，串行是供给瓶颈）；
      * 图任务并发放行 DECODE_CONC 张（内存上界：2048 边 RGB≈12MB×并发+
        文件字节缓存），完成一张补投一张；
      * 瓦片就绪即送主线程，满 TILE_FWD_BATCH 或 tick 超时做一次 GPU 前向；
      * coarse 追加顺序 = 前向行顺序 = 瓦片完成顺序，落盘前逐位校验。
    """
    files = IndexFiles(prefix)
    old_n = engine.coarse.size
    ex = engine._get_extractor()
    workers = max(1, ex.decode_workers)
    n = len(todo)
    if n == 0:
        return 0
    t0 = time.time()
    TILE_FWD_BATCH = max(64, ex.batch)
    _cfg = engine.cfg
    TICK = max(2.0, float(getattr(_cfg, "tile_flush_ms", 20) or 20)) / 1000.0
    DECODE_CONC = min(workers, int(getattr(_cfg, "tile_decode_slots", 18) or 20))
    ready_q: "queue.Queue" = queue.Queue(maxsize=8 * workers)
    task_q: "queue.Queue" = queue.Queue()
    lock = threading.Lock()
    stop = threading.Event()
    n_img_done = 0
    n_img_active = 0
    n_tile_inflight = 0
    next_img = [0]
    end_sent = [False]

    def maybe_send_end():
        nonlocal n_img_done, n_img_active, n_tile_inflight
        with lock:
            if (n_img_done >= n and n_img_active == 0
                    and n_tile_inflight == 0 and not end_sent[0]):
                end_sent[0] = True
                ready_q.put(None)

    def feed_next():
        nonlocal n_img_active
        with lock:
            while next_img[0] < n and n_img_active < DECODE_CONC:
                idx = next_img[0]
                next_img[0] += 1
                n_img_active += 1
                task_q.put(("img", idx, todo[idx]))

    def worker():
        nonlocal n_img_done, n_img_active, n_tile_inflight
        while not stop.is_set():
            try:
                item = task_q.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = "?"
            try:
                kind = item[0]
                if kind == "img":
                    _i, path = item[1], item[2]
                    t_a = time.monotonic()
                    crops = _decode_to_crops(path, engine.cfg, tile, overlap,
                                             min_side, pre_max)
                    t_b = time.monotonic()
                    if trace is not None:
                        try:
                            nb = os.path.getsize(path)
                        except OSError:
                            nb = 0
                        trace.on_file(t_a, t_b, len(crops), nb, path)
                    with lock:
                        n_img_done += 1
                        n_img_active -= 1
                        n_tile_inflight += len(crops)
                    for rgb_crop, box, data, first in crops:
                        task_q.put(("tile", path, rgb_crop, box, data, first))
                    if progress:
                        try:
                            progress(n_img_done, n)
                        except Exception:          # noqa: BLE001
                            pass
                    feed_next()
                    maybe_send_end()
                else:
                    path, rgb_crop, box, data, first = (item[1], item[2],
                                                        item[3], item[4],
                                                        item[5])
                    out = _feature_one_tile(path, rgb_crop, box, data, first,
                                            engine.cfg, ex.transform,
                                            frame_sink)
                    with lock:
                        n_tile_inflight -= 1
                    if out is not None:
                        ready_q.put((path, [out]))
                    maybe_send_end()
            except Exception as e:                  # noqa: BLE001 —— 单项失败不中断
                LOGGER.debug("瓦片任务异常 %s: %r", item, e)
                if kind == "tile":
                    with lock:
                        n_tile_inflight -= 1
                    maybe_send_end()
            finally:
                task_q.task_done()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(workers)]
    for t in threads:
        t.start()

    def feed():
        feed_next()
        task_q.join()

    threading.Thread(target=feed, daemon=True).start()

    all_ok: List[str] = []
    all_feats: List[np.ndarray] = []
    acc_paths, acc_ts, acc_recs = [], [], []
    acc_t0 = time.monotonic()
    ended = False

    def flush():
        nonlocal acc_paths, acc_ts, acc_recs, acc_t0
        if not acc_ts:
            acc_t0 = time.monotonic()
            return
        paths, tensors, recs = acc_paths, acc_ts, acc_recs
        acc_paths, acc_ts, acc_recs = [], [], []
        acc_t0 = time.monotonic()
        try:
            accepted = engine.coarse.add_results(paths, recs)
            keep = [i for i, a in enumerate(accepted) if a]
            if keep:
                sub_paths = [paths[i] for i in keep]
                sub_ts = [tensors[i] for i in keep]
                tb0 = time.monotonic()
                feats = ex._forward(sub_ts)     # noqa: SLF001 同项目协作
                tb1 = time.monotonic()
                if trace is not None:
                    trace.on_batch(tb0, tb1, len(sub_ts))
                if feats is not None:
                    rows = feats.astype(np.float32)
                    norms = np.linalg.norm(rows, axis=1, keepdims=True)
                    norms[norms < 1e-8] = 1.0
                    rows /= norms
                    all_feats.append(rows)
                    all_ok.extend(sub_paths)
        except Exception as e:                   # noqa: BLE001 —— 单批失败跳过
            LOGGER.warning("瓦片批次处理失败（%d 块）已跳过: %r", len(paths), e)

    while not ended:
        try:
            item = ready_q.get(timeout=TICK)
        except queue.Empty:
            # 结束守护：计数归零即收尾（不依赖可能丢失的结束哨兵）
            with lock:
                all_done = (n_img_done >= n and n_img_active == 0
                            and n_tile_inflight == 0)
            if all_done:
                flush()
                ended = True
                break
            if (time.monotonic() - t0 > 5.0 and
                    time.monotonic() - getattr(_ingest_tiles, "_dbg_t", 0) > 2.0):
                _ingest_tiles._dbg_t = time.monotonic()
                LOGGER.warning(
                    "[ingest-wait] img_done=%d/%d active=%d inflight=%d "
                    "task_q=%d ready_q=%d acc=%d",
                    n_img_done, n, n_img_active, n_tile_inflight,
                    task_q.qsize(), ready_q.qsize(), len(acc_ts))
            flush()
            continue
        if item is None:
            flush()
            ended = True
            break
        if isinstance(item, tuple) and item and item[0] == "ERR":
            LOGGER.error("瓦片流水线异常: %r", item[1])
            raise RuntimeError(f"瓦片流水线失败: {item[1]!r}")
        _path, tiles = item
        for tensor, rec in tiles:
            acc_paths.append(rec.path)
            acc_ts.append(tensor)
            acc_recs.append(rec)
        if len(acc_ts) >= TILE_FWD_BATCH or \
                (acc_ts and time.monotonic() - acc_t0 >= TICK):
            flush()

    stop.set()
    for t in threads:
        t.join(timeout=3)

    added = len(all_ok)
    if all_feats:
        feats_all = np.concatenate(all_feats, axis=0).astype(np.float32)
        if engine.coarse.paths[old_n:] != all_ok:
            raise RuntimeError("瓦片增量顺序不一致（粗筛与精排错位），请联系排查")
        # 落盘：coarse(含 boxes) + fine(旧特征追加或新建) + meta
        st = engine.coarse.export_state()
        old_feats = engine._fine_feats
        if old_feats is not None and len(old_feats):
            feats_all = np.concatenate(
                [np.asarray(old_feats, dtype=np.float32), feats_all], axis=0)
        side = getattr(engine, "_storage", "npz") == "sidecar"
        # 解除 fine 的 mmap（含 LSH 缓存引用）：Windows 下被映射的 .npy
        # 无法被替换，必须先释放；feats_all 已是拼接后的新数组
        LOGGER.info("瓦片特征提取完成：%d 块", added)
        _phase(progress, added)
        engine.release_fine()
        old_feats = None
        files.save_coarse(st["paths"], st["md5s"], st["hu"], st["fp"],
                          st["hu_mean"], st["hu_std"], boxes=st.get("boxes"),
                          sidecar=side)
        files.save_fine(st["paths"], feats_all, sidecar=side)
        engine._keep_fine(feats_all)
        meta = meta_of(engine.cfg, engine.coarse.size,
                       os.path.dirname(files.coarse_path), True,
                       feats_all.shape[1], storage=engine._storage)
        meta[TILES_META_KEY] = {"kind": "tile", "tile": tile,
                                "overlap": overlap, "min_side": min_side,
                                "pre_max": pre_max}
        files.save_meta(meta)
    else:
        # 全部被去重/失败：也要把（可能为空的）coarse 状态保存，保证前缀可用
        st = engine.coarse.export_state()
        side = getattr(engine, "_storage", "npz") == "sidecar"
        files.save_coarse(st["paths"], st["md5s"], st["hu"], st["fp"],
                          st["hu_mean"], st["hu_std"], boxes=st.get("boxes"),
                          sidecar=side)
        meta = meta_of(engine.cfg, engine.coarse.size,
                       os.path.dirname(files.coarse_path),
                       engine._fine_feats is not None,
                       None, storage=engine._storage)
        meta[TILES_META_KEY] = {"kind": "tile", "tile": tile,
                                "overlap": overlap, "min_side": min_side,
                                "pre_max": pre_max}
        files.save_meta(meta)
    rate = added / max(time.time() - t0, 1e-6)
    _phase(progress, added, "done")
    LOGGER.info("瓦片索引：+%d 块（原图 %d 张），库内共 %d 块，吞吐 %.1f 块/秒",
                added, n, engine.coarse.size, rate)
    return added


def _phase(progress, done: int, phase: str = "save") -> None:
    """向进度回调发送阶段边界事件（save/done），失败不影响建库。"""
    if progress is None:
        return
    try:
        progress(int(done), int(max(done, 1)), phase)
    except Exception:                       # noqa: BLE001
        pass


def build_tiles(engine, prefix: str, img_dir: Optional[str] = None,
                paths: Optional[List[str]] = None, force: bool = False,
                progress=None, frame_sink=None,
                tile: int = TILE_DEFAULT, overlap: float = OVERLAP_DEFAULT,
                min_side: int = MIN_SIDE_DEFAULT,
                pre_max: int = PRE_MAX_SIDE,
                trace: Optional[BuildTrace] = None) -> int:
    """从零构建瓦片索引；索引已存在且未 force 时报错。返回瓦片条目数。"""
    files = IndexFiles(prefix)
    if files.meta_exists() and not force:
        raise FileExistsError(
            f"瓦片索引 {prefix}.* 已存在；加 force 重建，或走增量")
    if paths is None:
        if not img_dir:
            raise ValueError("需要 img_dir 或 paths 之一")
        paths = collect_images(img_dir, engine.cfg.extensions)
    if not paths:
        LOGGER.warning("瓦片建库：没有图片可处理")
        return 0
    # 全新引擎实例（避免与其它已打开前缀混用）
    from .engine import HybridEngine
    eng = HybridEngine(engine.cfg) if engine.coarse.size else engine
    return _ingest_tiles(eng, prefix, paths, progress=progress,
                         frame_sink=frame_sink, tile=tile, overlap=overlap,
                         min_side=min_side, pre_max=pre_max, trace=trace)


def add_tiles(engine, prefix: str, img_dir: Optional[str] = None,
              paths: Optional[List[str]] = None, progress=None,
              frame_sink=None, trace: Optional[BuildTrace] = None) -> int:
    """瓦片索引增量（前缀必须已存在；切块参数从 meta 恢复）。"""
    files = IndexFiles(prefix)
    if not files.meta_exists():
        raise FileNotFoundError(f"瓦片索引 {prefix}.* 不存在，请先 build-tiles")
    if engine.prefix != prefix:
        engine.open(prefix)
    tp = (engine.meta.get(TILES_META_KEY) or {})
    tile = int(tp.get("tile", TILE_DEFAULT))
    overlap = float(tp.get("overlap", OVERLAP_DEFAULT))
    min_side = int(tp.get("min_side", MIN_SIDE_DEFAULT))
    pre_max = int(tp.get("pre_max", PRE_MAX_SIDE))
    if paths is None:
        if not img_dir:
            raise ValueError("需要 img_dir 或 paths 之一")
        paths = collect_images(img_dir, engine.cfg.extensions)
    # 去重集合只构建一次（历史问题：集合写在列表推导的 if 里，每个待处理
    # 文件都要重建一遍全库路径集合 —— 444k 索引实测 530ms/张，42k 张需 6 小时）
    known = {os.path.normcase(os.path.abspath(x))
             for x in engine.coarse.paths}
    todo = [p for p in paths
            if os.path.normcase(os.path.abspath(p)) not in known]
    LOGGER.info("瓦片增量：扫描到 %d 张，待处理 %d 张", len(paths), len(todo))
    return _ingest_tiles(engine, prefix, todo, progress=progress,
                         frame_sink=frame_sink, tile=tile, overlap=overlap,
                         min_side=min_side, pre_max=pre_max, trace=trace)


# ---------------------------------------------------------------------------
# LSH（随机投影多表）近似最近邻
# ---------------------------------------------------------------------------
class LshTables:
    """SRP 式多表 LSH：sign(X @ P) 拼接为桶键，查询取多表并集。

    近似召回（概率性）：表数/位数越高召回越高、查询越慢；桶内候选上限
    cap_per_bucket 防止高频桶爆炸。所有随机矩阵种子固定 -> 结果可复现。
    """

    def __init__(self, n_bits: int = 12, n_tables: int = 8, seed: int = 20260908,
                 cap_per_bucket: int = 256, max_candidates: int = 6000):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.seed = seed
        self.cap_per_bucket = cap_per_bucket
        self.max_candidates = max_candidates
        self._tables: List[dict] = []
        self._projs: List[np.ndarray] = []
        self._dim = 0
        self._rows = 0

    def fit(self, feats: np.ndarray) -> None:
        """feats：(N,D) 已 L2 归一化。构建所有表（内存行号 = 矩阵行序）。"""
        self._dim = feats.shape[1]
        self._rows = feats.shape[0]
        self._tables = []
        self._projs = []
        base = np.random.RandomState(self.seed)
        X = np.asarray(feats, dtype=np.float32)
        t0 = time.time()
        for _ in range(self.n_tables):
            P = base.normal(size=(self._dim, self.n_bits)).astype(np.float32)
            self._projs.append(P)
            keys = _hash_keys(X, P)
            self._tables.append(_bucketize(keys, self.cap_per_bucket))
        LOGGER.info("LSH 建表：%d 行 × %d 表 × %d bit，%.1f s",
                    self._rows, self.n_tables, self.n_bits, time.time() - t0)

    def query(self, q: np.ndarray, top_override: Optional[int] = None) -> np.ndarray:
        """返回候选行号（去重，可能包含近似漏召 —— 由调用方做 hash 复核）。"""
        cap = top_override or self.max_candidates
        if not self._tables:
            raise RuntimeError("LSH 表未构建（先 fit）")
        cand: List[np.ndarray] = []
        for table, P in zip(self._tables, self._projs):
            key = _hash_keys(q.reshape(1, -1), P)[0]
            bucket = table.get(int(key))
            if bucket is not None and len(bucket):
                cand.append(bucket)
        if not cand:
            return np.zeros(0, dtype=np.int64)
        rows = np.unique(np.concatenate(cand))
        return rows[:cap]


def _hash_keys(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """sign 投影 -> 每行一个桶键（int64）。"""
    bits = (X @ P >= 0).astype(np.uint8)          # (N, b)
    mult = (1 << np.arange(bits.shape[1], dtype=np.uint64))
    return bits.astype(np.uint64) @ mult


def _bucketize(keys: np.ndarray, cap: int) -> dict:
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    if sk.size == 0:
        return {}
    bounds = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
    out: dict = {}
    for i in range(len(bounds) - 1):
        seg = order[bounds[i]:bounds[i + 1]]
        if len(seg) > cap:
            seg = seg[:cap]
        out[int(sk[bounds[i]])] = seg
    return out


# ---------------------------------------------------------------------------
# 检索：LSH 候选 -> hash 复核 -> 原图聚合 -> 整图切片精排
# ---------------------------------------------------------------------------
def group_origins(paths: List[str]) -> Tuple[np.ndarray, dict]:
    """把瓦片行按“原图(origin)”分组 —— 兼容任意完成顺序（两级流水下
    同一原图的瓦片不再保证连续）。
    返回 (origins 数组(唯一), rows_of: origin -> np.int64 行号数组)。"""
    arr = np.asarray(paths, dtype=object)
    if arr.size == 0:
        return np.zeros(0, dtype=object), {}
    uniq, inv = np.unique(arr, return_inverse=True)
    cnt = np.bincount(inv)
    order = np.argsort(inv, kind="stable")
    bounds = np.cumsum(cnt)[:-1]
    groups = np.split(order, bounds)
    rows_of = {str(uniq[i]): g.astype(np.int64) for i, g in enumerate(groups)}
    return uniq, rows_of


def _matmul_scores(rows: np.ndarray, q: np.ndarray, gpu: bool) -> np.ndarray:
    """rows (M,D) × q(D) 余弦（q 已归一化；rows 已归一化）。GPU 优先，失败回退 CPU。"""
    if gpu:
        try:
            import torch
            if torch.cuda.is_available():
                t = torch.from_numpy(np.ascontiguousarray(rows)).to("cuda")
                qq = torch.from_numpy(q).to("cuda")
                with torch.no_grad():
                    return (t @ qq).cpu().numpy().astype(np.float32)
        except Exception as e:                     # noqa: BLE001
            LOGGER.debug("GPU 打分不可用回退 CPU: %r", e)
    return np.asarray(rows, dtype=np.float32) @ q.astype(np.float32)


def search_tiles(engine, q_path: str, top_k: int = 10,
                 coarse_k: int = 300, method: str = "lsh",
                 lsh_bits: int = 12, lsh_tables: int = 8,
                 recheck_hits: int = 900, gpu: bool = True) -> Outcome:
    """
    局部（瓦片）索引检索。method：
      lsh    —— LSH 近似候选 + 指纹 hash 复核（默认，亚秒级候选）
      coarse —— 全库指纹线性扫描做候选（对照基线，不建 LSH 表）
    三级：候选 -> 指纹复核 -> 原图组 max -> 组内整瓦片切片精确余弦 -> Top-K。
    """
    t_all = time.time()
    times: dict = {}
    coarse = engine.coarse
    paths = coarse.paths
    n = coarse.size
    feats = engine._fine_feats
    has_box = coarse.has_boxes()          # 框按需解析（可能是 mmap 数组）.boxes
    out = Outcome(query=q_path, db_size=n, method="tiles")

    # ---- 查询特征：指纹（hash 查验用）+ ResNet（精排用）---------------
    t0 = time.time()
    q_rec = coarse.query_record(q_path)
    times["查询图指纹"] = time.time() - t0

    q_abs = os.path.normcase(os.path.abspath(q_path))

    # ---- 1) 候选 ------------------------------------------------------
    t0 = time.time()
    cand_rows: np.ndarray
    if method == "lsh":
        if feats is None:
            raise RuntimeError("LSH 检索需要瓦片精排索引（fine），请先建全库特征")
        lsh = _lsh_for(engine, feats, lsh_bits, lsh_tables)
        cand_rows = lsh.query(q=_query_fine_feat(engine, q_path, times),
                              top_override=coarse_k * 40)
    else:
        cand_rows = np.arange(n, dtype=np.int64)
    times["LSH/候选获取"] = time.time() - t0

    if cand_rows.size == 0:
        LOGGER.warning("LSH 无候选（查询与库差异过大）")
        out.times = {"total": time.time() - t_all}
        return out

    # ---- 查询特征（指纹 + ResNet）:ResNet 全程只提取一次 ---------------
    q_cache: dict = {"v": None}

    def get_q():
        if q_cache["v"] is None:
            q_cache["v"] = _query_fine_feat(engine, q_path, times)
        return q_cache["v"]

    # ---- 2) hash 内容查验（近似概率）：过滤明确无关瓦片 -----------------
    t0 = time.time()
    from .coarse import _hamming_distance   # noqa: PLC0415 同包复用
    cand_fp = coarse.fp[cand_rows] if coarse.fp is not None else None
    if cand_fp is not None and q_rec.fp is not None:
        xor = np.bitwise_xor(cand_fp, np.asarray(q_rec.fp, dtype=np.uint8))
        d = _hamming_distance(xor) / float(coarse.n_bytes * 8)
    else:
        d = np.zeros(len(cand_rows), dtype=np.float64)
    keep = d < 0.62                       # 宽松内容过滤（JPEG 重压缩噪声下
    # 同源瓦片指纹差仍可能 ~0.3-0.5；0.62 只剔除明确不相关内容）
    cand_rows = cand_rows[keep]
    cand_d = d[keep]
    times["hash查验(指纹过滤)"] = time.time() - t0
    if cand_rows.size == 0:
        out.times = {"total": time.time() - t_all}
        return out

    # ---- 3) 原图粗排聚合：候选瓦片精确余弦 -> 组内 max -> coarse_k origins
    t0 = time.time()
    o_list: List[Tuple[str, Tuple[float, int]]] = []
    q_feat = get_q()
    if q_feat is not None:
        sub = np.asarray(feats[cand_rows], dtype=np.float32)
        sub_norm = np.linalg.norm(sub, axis=1, keepdims=True)
        sub_norm[sub_norm < 1e-8] = 1.0
        sub = sub / sub_norm
        gpu_ok = gpu and getattr(engine._get_extractor(), "device", "") == "cuda"
        cos = _matmul_scores(sub, q_feat, gpu_ok)
        order = np.argsort(cos, kind="stable")[::-1]
        seen: set = set()
        agg: Dict[str, Tuple[float, int]] = {}
        for pos in order:
            row = int(cand_rows[pos])
            origin = paths[row]
            if origin in seen:
                continue
            seen.add(origin)
            agg[origin] = (float(cos[pos]), row)
            if len(agg) >= coarse_k:
                break
        o_list = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)
    else:
        # 无精排特征：退回指纹复核分聚合
        agg2: Dict[str, Tuple[float, int]] = {}
        order2 = np.argsort(cand_d, kind="stable")
        for pos in order2:
            row = int(cand_rows[pos])
            origin = paths[row]
            score = 1.0 - float(cand_d[pos])
            if origin not in agg2 or score > agg2[origin][0]:
                agg2[origin] = (score, row)
            if len(agg2) >= coarse_k:
                break
        o_list = sorted(agg2.items(), key=lambda kv: kv[1][0], reverse=True)
    o_list = o_list[:coarse_k]
    times["原图粗排(瓦片余弦组max)"] = time.time() - t0
    if not o_list:
        out.times = {"total": time.time() - t_all}
        return out

    # ---- 4) 整图切片收敛：候选原图的全部瓦片特征切片精确余弦 ------------
    t0 = time.time()
    _orig_names, rows_of = group_origins(paths)
    o_rows = [(o, rows_of[str(o)]) for o, _score in o_list]
    sel_rows = np.concatenate([g for _o, g in o_rows]).astype(np.int64)

    if feats is not None:
        q = q_feat if q_feat is not None else get_q()
        if q is None:
            out.coarse_only = True
        else:
            sub = np.asarray(feats[sel_rows], dtype=np.float32)
            sub_norm = np.linalg.norm(sub, axis=1, keepdims=True)
            sub_norm[sub_norm < 1e-8] = 1.0
            sub = sub / sub_norm
            gpu_ok = gpu and getattr(engine._get_extractor(), "device", "") == "cuda"
            scores = _matmul_scores(sub, q, gpu_ok)
            times["收敛精排(整图切片×q)"] = time.time() - t0
            # 组内 max -> (原图分, 最优瓦片行)
            best: Dict[int, Tuple[float, int]] = {}
            k = 0
            for oi, (_o, g) in enumerate(o_rows):
                cnt = len(g)
                seg = scores[k:k + cnt]
                if seg.size:
                    bi = int(np.argmax(seg))
                    best[oi] = (float(seg[bi]), int(sel_rows[k + bi]))
                k += cnt
            order2 = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
            for rank, (oi, (sc, row)) in enumerate(order2[:top_k], 1):
                origin = o_list[oi][0]
                box = coarse.box_at(int(row)) if has_box else None
                if engine.cfg.exclude_self and \
                        os.path.normcase(os.path.abspath(origin)) == q_abs:
                    out.self_excluded = True
                    continue
                out.hits.append(Hit(rank=rank, path=origin,
                                    fine_score=sc,
                                    coarse_score=o_list[oi][1][0],
                                    d_hu=float("nan"), d_fp=float("nan"),
                                    box=box, match_kind="tile"))
    else:
        # 无精排：按复核分返回
        out.coarse_only = True
        for rank, (o, (sc, row)) in enumerate(o_list[:top_k], 1):
            box = coarse.box_at(int(row)) if has_box else None
            if engine.cfg.exclude_self and \
                    os.path.normcase(os.path.abspath(o)) == q_abs:
                out.self_excluded = True
                continue
            out.hits.append(Hit(rank=rank, path=o, fine_score=float("nan"),
                                coarse_score=sc,
                                d_hu=float("nan"), d_fp=float("nan"),
                                box=box, match_kind="tile"))
    times["total"] = time.time() - t_all
    out.times = times
    out.coarse_kept = len(cand_rows)
    return out


def search_tiles_tiled(engine, q_path: str, top_k: int = 10,
                       coarse_k: int = 300, method: str = "lsh",
                       lsh_bits: int = 12, lsh_tables: int = 8,
                       auto_tile: bool = True,
                       tile: int = TILE_DEFAULT,
                       overlap: float = OVERLAP_DEFAULT) -> Outcome:
    """瓦片索引检索（查询侧自动切块版）。

    背景（真实案例）：查询若是一张“较大的局部图”（例如整图横切一半/大半），
    整图单块会被 Resize 到 224 与库中 512px 瓦片尺度错配，在几十万瓦片库中
    无法区分（cos≈0.87-0.89 的近邻成片，正确原图排不进来）。

    本函数：查询图按与建库一致的瓦片协议(512px+25% 重叠)切成若干块，
    每块独立走 LSH/线性候选 + hash 过滤 + 瓦片余弦；跨块按原图聚合取
    “任一查询块×任一瓦片”的最大余弦；最后对聚合出的 top origins 做全瓦片
    × 全部查询块矩阵精排（防 LSH 漏行），输出与 search_tiles 相同的 Outcome。

    小查询（较短边 < min_tile_side=768，即与建库“不切块”语义一致）自动
    回退单块 search_tiles，开销不变。"""
    from .coarse import _hamming_distance   # noqa: PLC0415
    t_all = time.time()
    times: dict = {}
    data = read_bytes(q_path)
    if data is None:
        return search_tiles(engine, q_path, top_k=top_k, coarse_k=coarse_k,
                            method=method, lsh_bits=lsh_bits,
                            lsh_tables=lsh_tables)
    rgb = decode_rgb(data)
    if rgb is None:
        return search_tiles(engine, q_path, top_k=top_k, coarse_k=coarse_k,
                            method=method, lsh_bits=lsh_bits,
                            lsh_tables=lsh_tables)
    h, w = rgb.shape[:2]
    if auto_tile and max(w, h) < MIN_SIDE_DEFAULT:
        return search_tiles(engine, q_path, top_k=top_k, coarse_k=coarse_k,
                            method=method, lsh_bits=lsh_bits,
                            lsh_tables=lsh_tables)
    coarse = engine.coarse
    paths = coarse.paths
    feats = engine._fine_feats
    n = coarse.size
    if feats is None:
        raise RuntimeError("瓦片检索需要精排索引（fine）")
    has_box = coarse.has_boxes()
    ex = engine._get_extractor()
    gpu_ok = ex.device == "cuda"

    # ---- 查询切块：与建库同协议（≤2048 处理空间）----------------------
    t0 = time.time()
    work, qboxes, _sc = tiles_of_rgb(rgb, tile, overlap,
                                     MIN_SIDE_DEFAULT, PRE_MAX_SIDE)
    blocks = []                     # (crop_rgb, 指纹fp, q特征)
    import cv2 as _cv2
    for b in qboxes:
        x0, y0, x1, y1 = b
        crop = work[y0:y1, x0:x1]
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            continue
        gray = _cv2.cvtColor(crop, _cv2.COLOR_RGB2GRAY)
        _bin, _hu, fp = extract_binary_features(gray, engine.cfg)
        t = ex.transform(Image.fromarray(crop))
        feat = ex._forward([t])
        if feat is None:
            continue
        qf = np.asarray(feat[0], dtype=np.float32)
        nn = float(np.linalg.norm(qf))
        blocks.append((crop, fp, qf / nn if nn > 1e-8 else qf))
    times["查询切块×特征"] = time.time() - t0
    if not blocks:
        return search_tiles(engine, q_path, top_k=top_k, coarse_k=coarse_k,
                            method=method, lsh_bits=lsh_bits,
                            lsh_tables=lsh_tables)

    # ---- 逐块候选 + hash 过滤 + 瓦片余弦 -> 跨块 origin 聚合 -----------
    t0 = time.time()
    n_bytes = coarse.n_bytes
    agg: Dict[str, Tuple[float, int]] = {}
    seen_cand = 0
    for _crop, fp_q, qf in blocks:
        if method == "lsh":
            lsh = _lsh_for(engine, feats, lsh_bits, lsh_tables)
            cand = lsh.query(q=qf, top_override=coarse_k * 40)
        else:
            cand = np.arange(n, dtype=np.int64)
        if cand.size == 0:
            continue
        seen_cand += cand.size
        if fp_q is not None and coarse.fp is not None:
            xor = np.bitwise_xor(coarse.fp[cand],
                                 np.asarray(fp_q, dtype=np.uint8))
            d = _hamming_distance(xor) / float(n_bytes * 8)
            cand = cand[d < 0.62]
        if cand.size == 0:
            continue
        sub = np.asarray(feats[cand], dtype=np.float32)
        subn = np.linalg.norm(sub, axis=1, keepdims=True)
        subn[subn < 1e-8] = 1.0
        sub = sub / subn
        cos = _matmul_scores(sub, qf, gpu_ok)
        for pos in np.argsort(cos, kind="stable")[::-1]:
            row = int(cand[pos])
            origin = paths[row]
            sc = float(cos[pos])
            prev = agg.get(origin)
            if prev is None or sc > prev[0]:
                agg[origin] = (sc, row)
            if len(agg) >= coarse_k * 4:
                break
    times["分块候选聚合"] = time.time() - t0
    o_list = sorted(agg.items(), key=lambda kv: kv[1][0],
                    reverse=True)[:coarse_k]
    if not o_list:
        out0 = Outcome(query=q_path, db_size=n, method="tiles")
        out0.times = {"total": time.time() - t_all}
        return out0

    # ---- 全瓦片 × 全部查询块 矩阵精排（防 LSH/聚合漏行）-----------------
    t0 = time.time()
    _og, rows_of = group_origins(paths)
    o_rows = [(o, rows_of[str(o)]) for o, _s in o_list]
    sel = np.concatenate([g for _o, g in o_rows]).astype(np.int64)
    R = np.asarray(feats[sel], dtype=np.float32)
    Rn = np.linalg.norm(R, axis=1, keepdims=True)
    Rn[Rn < 1e-8] = 1.0
    R = R / Rn
    Q = np.stack([qf for _c, _f, qf in blocks]).astype(np.float32)
    S = R @ Q.T                       # 行(瓦片) × 列(查询块)
    times["收敛精排(全瓦片×查询块)"] = time.time() - t0
    q_abs = os.path.normcase(os.path.abspath(q_path))
    out = Outcome(query=q_path, db_size=n, method="tiles")
    best: Dict[int, Tuple[float, int]] = {}
    k = 0
    n_qb = Q.shape[0]
    for oi, (_o, g) in enumerate(o_rows):
        cnt = len(g)
        seg = S[k:k + cnt]
        if seg.size:
            p = int(np.argmax(seg))
            bi, _bq = divmod(p, n_qb)          # 瓦片行(组内) × 查询块
            best[oi] = (float(seg[bi, _bq]), int(sel[k + bi]))
        k += cnt
    order2 = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    for rank, (oi, (sc, row)) in enumerate(order2[:top_k], 1):
        origin = o_list[oi][0]
        box = coarse.box_at(int(row)) if has_box else None
        if engine.cfg.exclude_self and \
                os.path.normcase(os.path.abspath(origin)) == q_abs:
            out.self_excluded = True
            continue
        out.hits.append(Hit(rank=rank, path=origin, fine_score=sc,
                            coarse_score=o_list[oi][1][0],
                            d_hu=float("nan"), d_fp=float("nan"),
                            box=box, match_kind="tile"))
    times["total"] = time.time() - t_all
    out.times = times
    out.coarse_kept = seen_cand
    return out


# engine 级缓存：LSH 表挂在引擎实例上（随引擎一起回收）。
# 历史问题：曾用模块级全局 dict 以 id(feats) 为键缓存，键值里强引用 memmap
# 与桶表且永不淘汰 —— 每搜一次图就常驻 ~1GB（444k 瓦片库实测 RSS 每次 +0.4GB），
# 表现为“搜图后内存不释放”。挂到 engine 上后，引擎被回收即释放。
def _lsh_for(engine, feats, n_bits: int, n_tables: int) -> LshTables:
    cache = getattr(engine, "_lsh_cache", None)
    if cache is None:
        cache = {}
        engine._lsh_cache = cache      # noqa: SLF001 同模块协作
    hit = cache.get(id(feats))
    if hit is not None and hit[0] is feats:
        _lsh, _bits, _tabs = hit[1], hit[2], hit[3]
        if _bits == n_bits and _tabs == n_tables:
            return _lsh
    lsh = LshTables(n_bits=n_bits, n_tables=n_tables)
    lsh.fit(np.asarray(feats, dtype=np.float32))
    cache.clear()                      # 同一引擎只保留一份（feats 变了就换）
    cache[id(feats)] = (feats, lsh, n_bits, n_tables)
    return lsh


def _query_fine_feat(engine, q_path: str, times: dict) -> Optional[np.ndarray]:
    t0 = time.time()
    f = engine._query_fine(q_path) if hasattr(engine, "_query_fine") else None
    times.setdefault("查询图ResNet", 0.0)
    if f is not None:
        times["查询图ResNet"] += time.time() - t0
    return f


def default_tiles_prefix(root: str) -> str:
    """GUI 约定：<图库根>/.gallery_index/gallery_tiles"""
    return os.path.join(root, ".gallery_index", "gallery_tiles")


def tiles_prefix_of(prefix: str) -> str:
    """由整图索引前缀推导瓦片前缀：同目录下命名 gallery_tiles"""
    return os.path.join(os.path.dirname(prefix), "gallery_tiles")


def merge_hybrid(out_full: Outcome, out_tiles: Outcome, query: str,
                 top_k: int) -> Outcome:
    """混合检索合并：同一原图在两套索引里各可能命中，按原图路径去重，
    相似度取两路较高值重新统计，命中类型标注 full/tile/both。"""
    best: dict = {}
    for h in list(out_full.hits) + list(out_tiles.hits):
        fs = h.fine_score
        if fs != fs:                     # NaN
            fs = -1.0
        prev = best.get(h.path)
        if prev is None or fs > prev["fine"]:
            best[h.path] = {"fine": fs, "coarse": h.coarse_score,
                            "box": h.box,
                            "kind": ("both" if (prev is not None and
                                                prev["kind"] != h.match_kind)
                                     else h.match_kind)}
    out = Outcome(query=query,
                  db_size=max(out_full.db_size, out_tiles.db_size),
                  times={**out_full.times,
                         **{f"瓦片·{k}": v for k, v in out_tiles.times.items()}},
                  method="hybrid")
    ranked = sorted(best.items(), key=lambda kv: kv[1]["fine"],
                    reverse=True)[:top_k]
    for rank, (path, v) in enumerate(ranked, 1):
        out.hits.append(Hit(rank=rank, path=path,
                            fine_score=v["fine"] if v["fine"] > 0 else
                            float("nan"),
                            coarse_score=v["coarse"],
                            d_hu=float("nan"), d_fp=float("nan"),
                            box=v["box"], match_kind=v["kind"]))
    out.coarse_kept = out_full.coarse_kept + out_tiles.coarse_kept
    out.self_excluded = out_full.self_excluded or out_tiles.self_excluded
    return out
