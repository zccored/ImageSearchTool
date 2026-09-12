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

"""融合建库的“预处理结果”缓存（L2 内存 + L3 磁盘）。

为什么需要：实测融合建库 400 张真实图，CPU 成本 ≈85 ms/张，其中解码 18~600ms
（PNG 无法域缩放、大 JPEG 熵解码占主导），而 **GPU 前向只要 0.41 ms/张**——
GPU 空窗 70%+ 的根因是 CPU 侧供给不足，不是显存/IO 调度。缓存让“重复建库”
（重建整图索引、改参数重算、再建瓦片前的整图流程）**完全跳过读原图与解码**。

缓存内容（键 = 绝对路径 + 文件大小 + mtime_ns，命中即等价于同一份文件）：
  * ResNet 输入：PIL 处理后的 **224×224 裁剪**，PNG 无损 ≈53 KB
    → 命中后解码 + 归一化 ≈1.5 ms，且张量与全新建库**逐位一致**；
  * 粗筛特征：打包指纹（512 B）+ Hu（7×f32）+ 内容 MD5，直接复用，无漂移；
  * 预处理签名（模型/尺寸）：签名不符视为未命中，避免换模型后读到旧产物。

实测（24 张真实图验证）：PNG 无损往返 **24/24 张量逐位一致**；JPEG q97 会漂到
余弦 0.9988，故采用 PNG。3.7 万张约占磁盘 1.9 GB。
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np

from .io_utils import LOGGER

MAGIC = b"IHPC2"
CACHE_DIRNAME = "prep_cache"
_L2_MAX_BYTES = 256 * 1024 * 1024      # 内存缓存上限（张量 602KB/张）


def default_cache_dir(prefix: str) -> str:
    """索引前缀 -> 缓存目录：<索引目录>/prep_cache。"""
    return os.path.join(os.path.dirname(os.path.abspath(prefix)), CACHE_DIRNAME)


def _sig_of(model: str, pre_side: int) -> str:
    raw = f"{model}|resize256|crop224|pre{pre_side}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


class PrepCache:
    """单文件/张的预处理缓存；线程安全，失败一律降级为“未命中”。"""

    def __init__(self, root: str, model: str = "", pre_side: int = 2048,
                 enabled: bool = True, mem_bytes: int = _L2_MAX_BYTES):
        self.root = root
        self.enabled = bool(enabled)
        self.sig = _sig_of(model, pre_side)
        self._mem_bytes = int(mem_bytes)
        self._mem: "OrderedDict[str, tuple]" = OrderedDict()
        self._mem_used = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.bytes_written = 0
        self.saved_ms = 0.0            # 累计省下的 CPU 时间（估算）

    # ---- 键与路径 ----------------------------------------------------
    @staticmethod
    def key_for(path: str) -> Optional[str]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        raw = f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|" \
              f"{st.st_mtime_ns}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def file_of(self, key: str) -> str:
        return os.path.join(self.root, key[:2], key + ".bin")

    # ---- 读 ----------------------------------------------------------
    def get(self, path: str) -> Optional[Tuple[object, object]]:
        """返回 (tensor, CoarseRecord) 或 None（未命中/已禁用/损坏）。"""
        if not self.enabled:
            return None
        key = self.key_for(path)
        if key is None:
            return None
        with self._lock:                        # L2：本进程内直接复用张量
            hit = self._mem.get(key)
            if hit is not None:
                self._mem.move_to_end(key)
                self.hits += 1
                return hit
        fp = self.file_of(key)
        try:
            with open(fp, "rb") as f:
                blob = f.read()
        except OSError:
            with self._lock:
                self.misses += 1
            return None
        try:
            out = self._decode_blob(blob, path)
        except Exception as e:                  # noqa: BLE001 —— 坏缓存当作未命中
            LOGGER.debug("预处理缓存损坏（已忽略）: %r", e)
            out = None
        if out is None:
            with self._lock:
                self.misses += 1
            return None
        tensor, rec = out
        self._put_mem(key, out)
        with self._lock:
            self.hits += 1
            self.saved_ms += 80.0               # 经验值：命中省去 ~80ms/张
        return out

    def _decode_blob(self, blob: bytes, path: str):
        import io

        import torch
        from PIL import Image

        from .coarse import CoarseRecord

        if blob[:5] != MAGIC:
            return None
        (hlen,) = struct.unpack("<I", blob[5:9])
        head = json.loads(blob[9:9 + hlen].decode("utf-8"))
        if head.get("sig") != self.sig:
            return None                          # 预处理契约变了 -> 未命中
        img = Image.open(io.BytesIO(blob[9 + hlen:])).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        # 与 _make_fused_prep 一致：PIL 张量 = ToTensor + Normalize
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        t = t.float().div_(255.0)
        t = t.sub_(torch.as_tensor(head["mean"]).view(3, 1, 1)).div_(
            torch.as_tensor(head["std"]).view(3, 1, 1))
        fp = np.frombuffer(bytes.fromhex(head["fp_hex"]), dtype=np.uint8)
        hu = np.asarray(head["hu"], dtype=np.float32)
        rec = CoarseRecord(path=path, md5=head["md5"], hu=hu, fp=fp)
        return t.contiguous(), rec

    # ---- 写 ----------------------------------------------------------
    def put(self, path: str, tensor, rec, mean, std) -> None:
        if not self.enabled or tensor is None or rec is None:
            return
        key = self.key_for(path)
        if key is None:
            return
        try:
            import io

            import torch
            from PIL import Image

            # (3,224,224) float32 -> uint8 RGB（逆向归一化，无损存 PNG）
            # 注意：必须 **四舍五入**（round）而不是截断（byte() 会 floor），
            # 否则浮点误差会让个别像素差 1 级，精排特征出现 ~5e-4 的余弦漂移。
            t = tensor.detach().cpu()
            m = torch.as_tensor(mean).view(3, 1, 1)
            s = torch.as_tensor(std).view(3, 1, 1)
            img_arr = ((t * s + m) * 255.0).round_().clamp_(0, 255).to(
                torch.uint8)
            arr = img_arr.permute(1, 2, 0).numpy()
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(arr)).save(
                buf, format="PNG", compress_level=1)
            png = buf.getvalue()
            head = {
                "sig": self.sig, "md5": rec.md5 or "",
                "hu": np.asarray(rec.hu, dtype=np.float32).tolist(),
                "fp_hex": bytes(np.asarray(rec.fp, dtype=np.uint8)).hex(),
                "mean": np.asarray(mean, dtype=np.float32).tolist(),
                "std": np.asarray(std, dtype=np.float32).tolist(),
            }
            hb = json.dumps(head, separators=(",", ":")).encode("utf-8")
            blob = MAGIC + struct.pack("<I", len(hb)) + hb + png
            fp_path = self.file_of(key)
            os.makedirs(os.path.dirname(fp_path), exist_ok=True)
            tmp = fp_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, fp_path)
            with self._lock:
                self.writes += 1
                self.bytes_written += len(blob)
        except Exception as e:                  # noqa: BLE001 —— 缓存失败不影响建库
            LOGGER.debug("预处理缓存写入失败（忽略）: %r", e)
            return
        self._put_mem(key, (tensor, rec))

    def _put_mem(self, key: str, item) -> None:
        import sys

        size = 602 * 1024 if "torch" in sys.modules else 0
        with self._lock:
            self._mem[key] = item
            self._mem.move_to_end(key)
            self._mem_used += size
            while self._mem_used > self._mem_bytes and len(self._mem) > 1:
                _k, _v = self._mem.popitem(last=False)
                self._mem_used -= size

    # ---- 统计/维护 ---------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "hits": self.hits,
                    "misses": self.misses, "writes": self.writes,
                    "bytes_written": self.bytes_written,
                    "saved_ms": self.saved_ms, "dir": self.root,
                    "mem_items": len(self._mem)}

    def summary(self) -> str:
        s = self.stats()
        if not s["enabled"]:
            return "预处理缓存：已关闭"
        if not (s["hits"] or s["writes"]):
            return "预处理缓存：本次未使用"
        return (f"预处理缓存：命中 {s['hits']} 张（约省 "
                f"{s['saved_ms'] / 1000:.0f}s CPU）、写入 {s['writes']} 张"
                f"（{s['bytes_written'] / 2 ** 20:.0f} MB）-> {s['dir']}")

    def clear(self) -> int:
        """删除磁盘缓存，返回删除的文件数。"""
        n = 0
        for dirpath, _dirs, files in os.walk(self.root):
            for f in files:
                try:
                    os.remove(os.path.join(dirpath, f))
                    n += 1
                except OSError:
                    pass
        with self._lock:
            self._mem.clear()
            self._mem_used = 0
        return n
