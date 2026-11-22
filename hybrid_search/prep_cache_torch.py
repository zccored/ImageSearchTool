# -*- coding: utf-8 -*-
"""torch 预处理结果缓存（独立缓存池，fp16 精度）。

与 `prep_cache.py`（PIL/uint8 缓存）**物理隔离**：
  * 目录 `<prefix目录>/prep_cache_torch`；
  * 签名含 model/resize/crop/prep=torch/fp16/v1，读到旧格式直接视为未命中；
  * 条目 = 预处理后的 [3,224,224] float16（zlib）+ CoarseRecord（hu/fp/md5）。

用途：`prep=torch` 的重复建库/增量不再读原图与解码，直接复用同源张量。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import struct
import threading
import time
import zlib
from typing import Optional, Tuple

import numpy as np
import torch

from .coarse import CoarseRecord
from .io_utils import LOGGER

MAGIC = b"ITC1"
DIRNAME = "prep_cache_torch"


def _sig_of(model: str, side: int, crop: int) -> str:
    raw = f"{model}|resize{side}|crop{crop}|torch|fp16|v1".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


class TorchPrepCache:
    def __init__(self, prefix: str, model: str, side: int = 256,
                 crop: int = 224, enabled: bool = True):
        self.root = os.path.join(os.path.dirname(os.path.abspath(prefix)), DIRNAME)
        self.enabled = bool(enabled)
        self.sig = _sig_of(model, side, crop)
        self.shape = (3, crop, crop)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.errors = 0
        self.bytes_written = 0
        self.saved_ms = 0.0
        self._q: "queue.Queue" = queue.Queue(maxsize=4096)
        self._stop = threading.Event()
        self._thread = None
        if self.enabled:
            try:
                os.makedirs(self.root, exist_ok=True)
                self._thread = threading.Thread(target=self._writer, daemon=True)
                self._thread.start()
            except Exception as e:                       # noqa: BLE001
                LOGGER.warning("torch 预处理缓存不可用：%r", e)
                self.enabled = False

    # ---- 键与路径 ----------------------------------------------------
    @staticmethod
    def key_for(path: str) -> Optional[str]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        raw = f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def file_of(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".bin")

    # ---- 读 ----------------------------------------------------------
    def get(self, path: str) -> Optional[Tuple[object, CoarseRecord]]:
        if not self.enabled:
            return None
        key = self.key_for(path)
        if key is None:
            return None
        fn = self.file_of(key)
        try:
            with open(fn, "rb") as fh:
                magic = fh.read(4)
                if magic != MAGIC:
                    self.misses += 1
                    return None
                (hlen,) = struct.unpack("<I", fh.read(4))
                head = json.loads(fh.read(hlen).decode("utf-8"))
                if head.get("sig") != self.sig:
                    self.misses += 1
                    return None
                payload = zlib.decompress(fh.read())
            arr = torch.from_numpy(
                np.frombuffer(payload, dtype="<f2").reshape(self.shape).astype(np.float32))
            fp = np.frombuffer(base64.b64decode(head["fp"]), dtype=np.uint8)
            hu = np.asarray(head["hu"], dtype=np.float32)
            rec = CoarseRecord(path=path, md5=head["md5"], hu=hu, fp=fp)
            self.hits += 1
            return arr, rec
        except FileNotFoundError:
            self.misses += 1
            return None
        except Exception as e:                           # noqa: BLE001
            self.errors += 1
            LOGGER.debug("torch 缓存读取失败 %s: %r", fn, e)
            return None

    # ---- 写 ----------------------------------------------------------
    def put_async(self, path: str, tensor_cpu, rec: CoarseRecord) -> None:
        """tensor_cpu: [3,H,W] CPU float（通常 fp16）。写入失败不影响建库。"""
        if not self.enabled:
            return
        key = self.key_for(path)
        if key is None:
            return
        try:
            t = tensor_cpu.detach().to("cpu", dtype=None)
            arr = t.numpy().astype("<f2", copy=False)
            if tuple(arr.shape) != self.shape:
                return
            head = {
                "sig": self.sig,
                "md5": rec.md5 or "",
                "hu": np.asarray(rec.hu, dtype=np.float32).tolist(),
                "fp": base64.b64encode(np.asarray(rec.fp, dtype=np.uint8).tobytes()).decode("ascii"),
            }
            blob = json.dumps(head, separators=(",", ":")).encode("utf-8")
            self._q.put_nowait((key, blob, arr))          # 压缩交给写线程，避免卡 GPU 消费
        except queue.Full:
            self.errors += 1
        except Exception as e:                           # noqa: BLE001
            self.errors += 1
            LOGGER.debug("torch 缓存入队失败 %r", e)

    def _writer(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                key, blob, arr = item
                body = zlib.compress(arr.tobytes(), 0)   # 存储档：压缩不是瓶颈，写入更快
                fn = self.file_of(key)
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                tmp = fn + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(MAGIC)
                    fh.write(struct.pack("<I", len(blob)))
                    fh.write(blob)
                    fh.write(body)
                os.replace(tmp, fn)
                self.writes += 1
                self.bytes_written += len(body) + len(blob) + 8
            except Exception as e:                       # noqa: BLE001
                self.errors += 1
                LOGGER.debug("torch 缓存写入失败 %r", e)
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 60.0) -> None:
        if not self.enabled:
            return
        t0 = time.time()
        while self._q.unfinished_tasks and time.time() - t0 < timeout:
            time.sleep(0.05)

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush()
        try:
            self._q.put_nowait(None)
        except Exception:                                # noqa: BLE001
            pass

    def summary(self) -> str:
        return ("torch预处理缓存：命中 %d / 未命中 %d / 写入 %d（%.1f MB）/ 错误 %d"
                % (self.hits, self.misses, self.writes,
                   self.bytes_written / 1e6, self.errors))
