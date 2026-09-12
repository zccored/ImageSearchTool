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

"""检索结果可视化：把 Top-K 结果拼成一张缩略图总览 PNG。"""
from __future__ import annotations

import os
from typing import List, Optional

from PIL import Image

from .io_utils import LOGGER

THUMB = 176          # 每格缩略图边长
PAD = 8              # 格子间距
COLS = 5             # 每行格数
BG = (24, 24, 28)


def _fit_thumb(path: str) -> Image.Image:
    """按比例缩略并居中填充到 THUMB 方格（cover）。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
    im.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (THUMB, THUMB), BG)
    x = (THUMB - im.width) // 2
    y = (THUMB - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def save_contact_sheet(paths: List[str], out_path: str,
                       captions: Optional[List[str]] = None) -> bool:
    """
    把 paths 拼成网格总览图保存。captions 与 paths 等长（可选，文本由 CLI 另行输出，
    这里只画图，避免字体缺失产生乱码方块）。失败返回 False 不抛错。
    """
    if not paths:
        return False
    rows = (len(paths) + COLS - 1) // COLS
    cols = min(COLS, len(paths))
    W = cols * THUMB + (cols + 1) * PAD
    H = rows * THUMB + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), BG)
    for i, p in enumerate(paths):
        try:
            cell = _fit_thumb(p)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("缩略图失败 %s: %r", p, e)
            continue
        r, c = divmod(i, COLS)
        x = PAD + c * (THUMB + PAD)
        y = PAD + r * (THUMB + PAD)
        sheet.paste(cell, (x, y))
        # 第一格用亮色描边标记“第 1 名”
        if i == 0:
            sheet = _border(sheet, (x - 1, y - 1), THUMB + 2, (255, 200, 40))
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        sheet.save(out_path)
        LOGGER.info("结果总览图已保存: %s", out_path)
        return True
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("保存总览图失败 %s: %r", out_path, e)
        return False


def _border(img: Image.Image, xy, size: int, color) -> Image.Image:
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([xy[0], xy[1], xy[0] + size - 1, xy[1] + size - 1], outline=color,
                width=3)
    return img
