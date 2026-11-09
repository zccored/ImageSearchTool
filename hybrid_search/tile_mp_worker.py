# -*- coding: utf-8 -*-
"""瓦片多进程 worker：读盘→解码→切块→粗筛，写入共享内存槽。

本模块**不得** import torch / hybrid_search.engine：子进程只带 cv2+numpy，
避免每个进程重复加载 CUDA 运行时。
"""

from __future__ import annotations

import hashlib
from multiprocessing import shared_memory

import cv2
import numpy as np

from .coarse import extract_binary_features
from .io_utils import decode_rgb, read_bytes


def _axis_offsets(size: int, tile: int, stride: int):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile, stride))
    last = size - tile
    if not xs or last - xs[-1] > 0:
        if xs and last - xs[-1] < max(1, stride // 2):
            xs[-1] = last
        else:
            xs.append(last)
    return xs


def tile_grid(w: int, h: int, tile: int, overlap: float):
    stride = max(1, int(round(tile * (1.0 - overlap))))
    out = []
    for y0 in _axis_offsets(h, tile, stride):
        for x0 in _axis_offsets(w, tile, stride):
            out.append((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
    return out


def tiles_of_rgb(rgb, tile, overlap, min_side, pre_max):
    h, w = rgb.shape[:2]
    if max(w, h) < min_side:
        return rgb, [(0, 0, w, h)], 1.0
    work, scale = rgb, 1.0
    if max(w, h) > pre_max:
        scale = pre_max / float(max(w, h))
        work = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]
    inv = 1.0 / scale
    boxes = [(max(0, int(x0 * inv)), max(0, int(y0 * inv)),
              min(w, int(round(x1 * inv))), min(h, int(round(y1 * inv))))
             for (x0, y0, x1, y1) in tile_grid(ww, wh, tile, overlap)]
    return work, boxes, scale


def _md5_of(base, box):
    h = base.copy()
    h.update(f"|{box[0]},{box[1]},{box[2]},{box[3]}".encode())
    return h.hexdigest()


def run(job_q, msg_q, slot_q, shm_name, slot_bytes, cfg,
        tile, overlap, min_side, pre_max):
    cv2.setNumThreads(1)
    shm = shared_memory.SharedMemory(name=shm_name)
    shm_view = shm.buf
    while True:
        job = job_q.get()
        if job is None:
            break
        img_idx, path = job
        n_tiles = 0
        try:
            data = read_bytes(path)
            rgb = decode_rgb(data) if data is not None else None
            if rgb is None:
                msg_q.put(("img_done", img_idx, 0))
                continue
            work, boxes, scale = tiles_of_rgb(rgb, tile, overlap, min_side, pre_max)
            base = hashlib.md5(data)
            for ti, box in enumerate(boxes):
                x0, y0, x1, y1 = box
                sx0 = int(round(x0 * scale)); sy0 = int(round(y0 * scale))
                sx1 = max(int(round(x1 * scale)), sx0 + 1)
                sy1 = max(int(round(y1 * scale)), sy0 + 1)
                sh, sw = work.shape[:2]
                sx0, sy0 = max(0, sx0), max(0, sy0)
                sx1, sy1 = min(sw, sx1), min(sh, sy1)
                if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                    continue
                crop = work[sy0:sy1, sx0:sx1]
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                _binary, hu, fp = extract_binary_features(gray, cfg)
                md5 = _md5_of(base, box)
                slot = slot_q.get()
                off = slot * slot_bytes
                shm_view[off:off + crop.nbytes] = crop.tobytes()
                msg_q.put((img_idx, ti, path, box, md5, hu, fp, slot,
                           crop.shape, crop.nbytes))
                n_tiles += 1
        except Exception as e:                       # noqa: BLE001
            msg_q.put(("err", img_idx, repr(e)))
        msg_q.put(("img_done", img_idx, n_tiles))
    try:
        shm_view.release()
        shm.close()
    except Exception:                                # noqa: BLE001
        pass
