# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 探针：224 裁剪缓存落盘格式定案与逐位一致性验证
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""缓存落盘格式定案：缓存“PIL 处理后的 224×224 裁剪”，测
  * 无损 PNG 往返 → 张量应逐位一致（漂移 0）
  * JPEG q97/q95 往返 → 张量余弦漂移与体积
另测命中时的重建耗时（解码 + 归一化）。

用法: python devtools/probe_cache_exact.py [张数]
"""
import io
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.fine import ResNetExtractor, _PRE_DOWNSCALE_PX, _PRE_DOWNSCALE_SIDE  # noqa: E402
from hybrid_search.io_utils import collect_images, decode_rgb, read_bytes  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
cfg = Config()
ex = ResNetExtractor(cfg)
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)

rows = {"png": [], "jpg97": [], "jpg95": []}
sizes = {k: [] for k in rows}
t_rebuild = []
for p in paths:
    data = read_bytes(p)
    if data is None:
        continue
    rgb = decode_rgb(data)
    if rgb is None:
        continue
    if rgb.shape[0] * rgb.shape[1] > _PRE_DOWNSCALE_PX:
        s = _PRE_DOWNSCALE_SIDE / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * s)),
                               max(1, int(rgb.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    # 复刻既有路径：PIL Resize(256) + CenterCrop(224) -> 得到 224 裁剪
    im = Image.fromarray(rgb)
    h, w = im.size[1], im.size[0]
    s = 256.0 / min(h, w)
    im256 = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      Image.Resampling.BILINEAR)
    W, H = im256.size
    x0, y0 = int(round((W - 224) / 2)), int(round((H - 224) / 2))
    crop = im256.crop((x0, y0, x0 + 224, y0 + 224))
    tensor0 = ex.transform(crop)
    fa = ex._forward([tensor0])[0]
    for tag, kw in (("png", dict(format="PNG", compress_level=1)),
                    ("jpg97", dict(format="JPEG", quality=97, subsampling=0)),
                    ("jpg95", dict(format="JPEG", quality=95, subsampling=0))):
        buf = io.BytesIO()
        crop.save(buf, **kw)
        blob = buf.getvalue()
        sizes[tag].append(len(blob))
        tt = time.time()
        back = Image.open(io.BytesIO(blob)).convert("RGB")
        tensor1 = ex.transform(back)
        t_rebuild.append(time.time() - tt)
        fb = ex._forward([tensor1])[0]
        cos = float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb)))
        rows[tag].append(cos)
        if tag == "png":
            rows.setdefault("png_bit_same", []).append(
                bool(torch.equal(tensor0, tensor1)))

n = max(len(rows["png"]), 1)
print(f"样本 {n} 张（缓存内容是 PIL 处理后的 224×224 裁剪）\n")
print(f"{'格式':<8}{'单张体积':>10}{'张量余弦(最小)':>16}{'逐位一致':>10}")
for tag in ("png", "jpg97", "jpg95"):
    same = sum(rows.get("png_bit_same", [])) if tag == "png" else 0
    print(f"{tag:<8}{statistics.median(sizes[tag]) / 1024:9.1f}K"
          f"{min(rows[tag]):16.6f}"
          f"{(str(same) + '/' + str(n)) if tag == 'png' else '-':>10}")
print(f"\n命中重建耗时（解码 224 裁剪 + 归一化）中位 "
      f"{statistics.median(t_rebuild) * 1000:.2f} ms")
print(f"3.7 万张磁盘占用：PNG {statistics.median(sizes['png']) * 37683 / 2**30:.2f} GB / "
      f"JPEG97 {statistics.median(sizes['jpg97']) * 37683 / 2**30:.2f} GB / "
      f"JPEG95 {statistics.median(sizes['jpg95']) * 37683 / 2**30:.2f} GB")
