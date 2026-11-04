# -*- coding: utf-8 -*-
"""瓦片建库的多进程后端：worker 解码+切块+粗筛，父进程统一 GPU 前向并落盘。

仅当 `cfg.tile_backend` 解析为 processes 且 `cfg.prep == "torch"` 时启用；
任何启动/通信失败由调用方回退到线程后端。
"""

from __future__ import annotations

import gc
import math
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import shared_memory

import numpy as np

from .coarse import CoarseRecord
from .io_utils import LOGGER
from .tile_index import _save_tiles
from . import tile_mp_worker as W

_SLOT_BATCH = 64


def resolve_tile_backend(cfg, frame_sink=None) -> str:
    """auto/threads/processes -> threads|processes（保守判定，失败由调用方兜底）。"""
    mode = str(getattr(cfg, "tile_backend", "auto") or "auto").lower()
    if frame_sink is not None:              # 可视化帧只能在线程内产生
        return "threads"
    if mode == "threads":
        return "threads"
    if str(getattr(cfg, "prep", "cpu")) != "torch":
        return "threads"                    # 非 GPU 预处理路径语义不同
    if mode == "processes":
        return "processes"
    if getattr(sys, "frozen", False):
        # 打包版默认仍走线程；等 freeze_support 全链路验证后再放开 auto
        return "threads"
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:                       # noqa: BLE001
        cuda = False
    cpu = os.cpu_count() or 4
    return "processes" if (cuda and cpu >= 8) else "threads"


def _resolve_workers(cfg) -> int:
    n = int(getattr(cfg, "tile_mp_workers", 0) or 0)
    if n > 0:
        return max(1, n)
    return max(4, min(12, (os.cpu_count() or 8) // 2))


def build_tiles_mp(engine, prefix: str, paths, *, progress=None,
                   tile: int, overlap: float, min_side: int,
                   pre_max: int) -> int:
    """多进程收集 + 父进程 GPU 前向 + 复用 `_save_tiles` 落盘。返回新增块数。"""
    cfg = engine.cfg
    n = len(paths)
    old_n = engine.coarse.size
    workers = _resolve_workers(cfg)
    slots = max(64, workers * 8)
    side = int(tile + math.ceil(tile * float(overlap)) + 2)
    slot_bytes = side * side * 3

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    msg_q = ctx.Queue()
    slot_q = ctx.Queue()
    for i in range(slots):
        slot_q.put(i)
    shm = shared_memory.SharedMemory(create=True, size=slots * slot_bytes)

    procs = []
    t0 = time.time()
    try:
        for _ in range(workers):
            p = ctx.Process(target=W.run,
                            args=(job_q, msg_q, slot_q, shm.name, slot_bytes,
                                  cfg, tile, overlap, min_side, pre_max))
            p.start()
            procs.append(p)
        for i, path in enumerate(paths):
            job_q.put((i, path))

        ex = engine._get_extractor()
        pending = []        # [(img, ti, rec, view, slot)]
        rows = []           # [(img, ti, feat)]
        rec_map = {}
        done_imgs = 0
        tiles = 0
        errors = 0

        def flush():
            if not pending:
                return
            ts = [p[3] for p in pending]
            feats = ex.forward_rgb(ts)
            for (img, ti, _rec, _arr, _slot), row in zip(pending, feats):
                rows.append((img, ti, row.astype(np.float32)))
            for _img, _ti, _rec, _arr, slot in pending:
                slot_q.put(slot)
            pending.clear()
            del ts

        while done_imgs < n:
            try:
                msg = msg_q.get(timeout=300)
            except Exception as e:                       # noqa: BLE001
                raise RuntimeError(f"多进程瓦片解码超时：{e!r}") from e
            if msg[0] == "img_done":
                done_imgs += 1
                if progress:
                    try:
                        progress(done_imgs, n)
                    except Exception:                    # noqa: BLE001
                        pass
                continue
            if msg[0] == "err":
                errors += 1
                LOGGER.warning("瓦片多进程解码失败 %s: %s", msg[1], msg[2])
                continue
            img_idx, ti, path, box, md5, hu, fp, slot, shape, nbytes = msg
            arr = np.frombuffer(shm.buf, dtype=np.uint8, count=nbytes,
                                offset=slot * slot_bytes).reshape(shape)
            rec = CoarseRecord(path=path, md5=md5, hu=hu, fp=fp, box=box)
            rec_map[(img_idx, ti)] = rec
            pending.append((img_idx, ti, rec, arr, slot))
            tiles += 1
            if len(pending) >= _SLOT_BATCH:
                flush()

        flush()
    finally:
        try:
            pending.clear()          # 释放共享内存视图后再 close（否则 BufferError）
        except NameError:
            pass
        for _ in procs:
            try:
                job_q.put(None)
            except Exception:                            # noqa: BLE001
                pass
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        gc.collect()
        try:
            shm.buf.release()
        except Exception:                            # noqa: BLE001
            pass
        gc.collect()
        try:
            shm.close()
            shm.unlink()
        except Exception as e:                           # noqa: BLE001
            LOGGER.debug("共享内存清理告警: %r", e)

    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    feats = np.stack([rows[i][2] for i in order]) if order else np.empty((0, 512))
    recs = [rec_map[(rows[i][0], rows[i][1])] for i in order]
    with engine.coarse.append_transaction():
        accepted = engine.coarse.add_results([r.path for r in recs], recs)
    keep = [i for i, ok in enumerate(accepted) if ok]
    all_ok = [recs[i].path for i in keep]
    all_feats = [feats[keep]] if keep else []
    added = _save_tiles(engine, prefix, old_n, all_ok, all_feats,
                        tile, overlap, min_side, pre_max, progress)
    LOGGER.info("瓦片多进程：%d 进程 %d 槽，%d 块（错误 %d），%.1f 块/s",
                workers, slots, tiles, errors, tiles / max(time.time() - t0, 1e-6))
    return added
