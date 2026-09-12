# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 缩略图磁盘缓存：<索引目录>/thumbs/ 96px，后台线程解码 + 滑出释放
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""缩略图磁盘缓存：<索引目录>/thumbs/<前2位>/<key>.jpg（默认 96px）。

为什么要落盘缓存：真实图库里 24~27MP 的 JPEG 单张解码约 120ms（熵解码占
主导，draft 降采样只能省一半左右），浏览一屏十几张就要一两秒；缓存成 96px
小图后，读取+解码 <2ms，页面可秒开，也不再需要每次解码全图。

key 取图片 MD5（去重报告里已有；没有时退化为“路径的 MD5”），因此同一份
内容的多份副本共用一张缩略图。

本模块不依赖 Tk：解码/落盘都在工作线程里做，主线程只负责把 PIL 图转成
PhotoImage（Tk 对象必须主线程创建）。
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Optional

from PIL import Image, ImageOps

__all__ = ["ThumbCache"]

THUMB_SIZE = 96
THUMB_QUALITY = 82


class ThumbCache:
    def __init__(self, root: str, size: int = THUMB_SIZE,
                 quality: int = THUMB_QUALITY):
        self.dir = os.path.join(root, "thumbs")
        self.size = int(size)
        self.quality = int(quality)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ---- 路径 --------------------------------------------------------
    @staticmethod
    def key_for(path: str, md5: str = "") -> str:
        if md5:
            return md5
        norm = os.path.normcase(os.path.abspath(path))
        return hashlib.md5(norm.encode("utf-8")).hexdigest()

    def file_of(self, key: str) -> str:
        return os.path.join(self.dir, key[:2], key + ".jpg")

    def exists(self, key: str) -> bool:
        return os.path.exists(self.file_of(key))

    # ---- 读写 --------------------------------------------------------
    def get(self, key: str) -> Optional[Image.Image]:
        p = self.file_of(key)
        try:
            with Image.open(p) as im:
                im.load()
                out = im.convert("RGB")
            with self._lock:
                self.hits += 1
            return out
        except Exception:               # noqa: BLE001
            with self._lock:
                self.misses += 1
            return None

    def put(self, key: str, im: Image.Image) -> None:
        p = self.file_of(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        im.convert("RGB").save(tmp, "JPEG", quality=self.quality,
                               optimize=False)
        os.replace(tmp, p)

    def make(self, src_path: str, key: str) -> Optional[Image.Image]:
        """解码原图 → 缩略图 → 落盘；返回缩略图（线程内调用）。"""
        try:
            with Image.open(src_path) as im:
                im.draft("RGB", (self.size * 4, self.size * 4))
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                im.load()
            im.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
            self.put(key, im)
            return im
        except Exception:               # noqa: BLE001
            return None

    # ---- 统计 --------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses,
                    "dir": self.dir}
