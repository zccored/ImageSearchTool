# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 模拟图库生成器：造“同组近重复 + 跨组异图”的测试数据用于验证检索
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
模拟图库生成器：造一批“同组近重复 + 跨组异图”的测试数据，用于验证检索系统。

原理：每个“分组”先画一张随机“底图”（随机形状/纹理/文字块的彩色合成图），
再从它派生出 N 张“变体”（旋转 ±8°、缩放、平移、亮度/对比度抖动、模糊、
噪声、JPEG 压缩、随机分辨率）作为图库成员；其中 1~2 张保留为查询图（不进库）。

因此“以图搜图”的理想结果 = 与查询图同组的图库图片（近似重复/相似性检索）。
文件命名统一带分组标记 g<编号>，供 eval 命令按正则解析真值：
  图库:   <out>/db/img_g0001_v3.jpg
  查询:   <out>/queries/q_g0001_v1.jpg

用法示例：
  python make_test_dataset.py --out ./test_data --db 5000 --per-group 8 --queries 60
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

JPEG_EXT = ".jpg"


def rng_for(seed: int) -> random.Random:
    return random.Random(seed)


def draw_base(r: random.Random, rng: np.random.RandomState) -> Image.Image:
    """画一张随机内容的“底图”（形状、色带、噪点、伪文字块）。"""
    w = r.randint(240, 560)
    h = r.randint(240, 560)
    # 随机纵向渐变背景（(h,w,3)）
    c1 = tuple(r.randint(0, 255) for _ in range(3))
    c2 = tuple(r.randint(0, 255) for _ in range(3))
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]          # (h,1)
    bg = (np.asarray(c1, np.float32) * (1 - t)
          + np.asarray(c2, np.float32) * t)                       # (h,3)
    bg = np.repeat(bg[:, None, :], w, axis=1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(bg, "RGB")
    d = ImageDraw.Draw(img)

    # 若干个随机图形
    for _ in range(r.randint(2, 6)):
        color = tuple(r.randint(0, 255) for _ in range(3))
        kind = r.choice(("ellipse", "rect", "poly", "line", "arc"))
        x0, y0 = r.randint(0, w - 40), r.randint(0, h - 40)
        x1, y1 = min(w, x0 + r.randint(30, 200)), min(h, y0 + r.randint(30, 200))
        if kind == "ellipse":
            d.ellipse([x0, y0, x1, y1], fill=color)
        elif kind == "rect":
            d.rectangle([x0, y0, x1, y1], outline=color, width=r.randint(2, 10))
        elif kind == "poly":
            pts = [(r.randint(0, w), r.randint(0, h)) for _ in range(3)]
            d.polygon(pts, fill=color)
        elif kind == "line":
            d.line([x0, y0, r.randint(0, w), r.randint(0, h)], fill=color,
                   width=r.randint(3, 14))
        else:
            d.arc([x0, y0, x1, y1], start=r.randint(0, 360),
                  end=r.randint(0, 360), fill=color, width=r.randint(3, 12))

    # 噪点 + 轻微纹理
    noise = rng.normal(0, 14, (h, w, 1)).clip(0, 255).astype(np.uint8)
    arr = np.asarray(img, dtype=np.int16) + noise
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
    return img


def make_variant(base: Image.Image, r: random.Random) -> Image.Image:
    """从底图派生一个近重复变体。"""
    img = base
    # 1) 小幅几何扰动（旋转/缩放/平移组合）
    ang = r.uniform(-8, 8)
    scale = r.uniform(0.92, 1.08)
    zoom = int(min(img.width, img.height) * r.uniform(0.86, 1.0))
    img = ImageOps.fit(img, (max(zoom, 16), max(zoom, 16)),
                       Image.Resampling.BILINEAR)
    img = img.rotate(ang, resample=Image.Resampling.BILINEAR,
                     fillcolor=(128, 128, 128))
    if scale != 1.0:
        img = img.resize((max(int(img.width * scale), 16),
                          max(int(img.height * scale), 16)),
                         Image.Resampling.BILINEAR)

    # 2) 色彩/亮度/对比度抖动
    if r.random() < 0.9:
        img = ImageEnhance.Brightness(img).enhance(r.uniform(0.7, 1.3))
    if r.random() < 0.9:
        img = ImageEnhance.Contrast(img).enhance(r.uniform(0.7, 1.3))
    if r.random() < 0.4:
        img = ImageEnhance.Color(img).enhance(r.uniform(0.5, 1.4))

    # 3) 轻微模糊 + 高斯噪声
    if r.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.2, 0.9)))
    arr = np.asarray(img, dtype=np.float32)
    if r.random() < 0.6:
        arr = arr + rng_noise(arr.shape, r)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")

    # 4) 随机最终分辨率（保内容近似）
    nw = r.randint(180, 640)
    nh = int(nw * img.height / img.width * r.uniform(0.94, 1.06))
    img = img.resize((nw, max(nh, 16)), Image.Resampling.LANCZOS)
    return img


def rng_noise(shape, r: random.Random) -> np.ndarray:
    rng = np.random.RandomState(r.randint(0, 2 ** 31))
    return rng.normal(0, 6, shape)


def save_jpeg(img: Image.Image, path: str, r: random.Random) -> None:
    quality = r.randint(62, 96)
    img.save(path, quality=quality, optimize=False)


def main() -> int:
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
    ap = argparse.ArgumentParser(
        description="生成模拟图库（近重复分组）用于测试二值法+ResNet 检索系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default="./test_data", help="输出根目录")
    ap.add_argument("--db", type=int, default=5000, help="图库图片总数")
    ap.add_argument("--per-group", type=int, default=8,
                    help="每组派生的变体数（图库中每组占位）")
    ap.add_argument("--queries", type=int, default=60, help="查询图数量")
    ap.add_argument("--seed", type=int, default=20250420, help="随机种子")
    a = ap.parse_args()

    groups_needed = -(-a.db // a.per_group)          # 组数（向上取整）
    db_total = groups_needed * a.per_group
    os.makedirs(os.path.join(a.out, "db"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "queries"), exist_ok=True)
    master = random.Random(a.seed)

    db_count = 0
    query_count = 0
    for g in range(groups_needed):
        gseed = master.randint(0, 2 ** 31)
        r = rng_for(gseed)
        rng = np.random.RandomState(gseed)
        base = draw_base(r, rng)
        # 每组保留 0~1 张做查询（查询张从该组变体里扣掉，不进库）
        keep_as_query = query_count < a.queries and g % 2 == 0
        variants = [make_variant(base, rng_for(gseed + i))
                    for i in range(a.per_group)]
        first = 0
        if keep_as_query:
            save_jpeg(variants[0],
                      os.path.join(a.out, "queries", f"q_g{g:05d}_v0{JPEG_EXT}"), r)
            query_count += 1
            first = 1
        for vi in range(first, a.per_group):
            save_jpeg(variants[vi],
                      os.path.join(a.out, "db",
                                   f"img_g{g:05d}_v{vi}{JPEG_EXT}"), r)
            db_count += 1
        if (g + 1) % 50 == 0:
            print(f"已生成 {g + 1}/{groups_needed} 组（图库 {db_count}，查询 {query_count}）")

    print(f"\n完成：图库 {db_count} 张 -> {os.path.join(a.out, 'db')}")
    print(f"查询 {query_count} 张 -> {os.path.join(a.out, 'queries')}")
    print(f"共 {groups_needed} 组（组号 g%05d，用于 eval 按正则解析真值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
