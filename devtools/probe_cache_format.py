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

"""决定预处理缓存的存储格式：256 短边图的 JPEG(q95/q97) vs PNG 无损，
对 ResNet 特征与粗筛指纹的影响，以及单文件体积。

用法: python devtools/probe_cache_format.py [张数]
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
from PIL import Image  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.coarse import extract_binary_features  # noqa: E402
from hybrid_search.fine import ResNetExtractor, _PRE_DOWNSCALE_PX, _PRE_DOWNSCALE_SIDE  # noqa: E402
from hybrid_search.io_utils import collect_images, decode_rgb, read_bytes  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
cfg = Config()
ex = ResNetExtractor(cfg)
paths = collect_images(r"F:\视频", cfg.extensions, limit=N)

res = {k: {"cos": [], "ham": []} for k in ("png", "jpg95", "jpg97")}
sizes = {"png": [], "jpg95": [], "jpg97": [], "raw256": []}
t_enc = {k: [] for k in sizes}
t_dec = {k: [] for k in sizes}

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
    # 基线：直接从解码图算（= 现有建库路径）
    gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _b, hu0, fp0 = extract_binary_features(gray0, cfg)
    t0 = ex.transform(Image.fromarray(rgb))
    f0 = ex._forward([t0])[0]
    # 256 短边中间图
    h, w = rgb.shape[:2]
    s = 256.0 / min(h, w)
    mid = cv2.resize(rgb, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA) if abs(s - 1) > 1e-6 else rgb
    sizes["raw256"].append(mid.nbytes)
    im = Image.fromarray(mid)
    for tag, kw in (("png", dict(format="PNG", compress_level=1)),
                    ("jpg95", dict(format="JPEG", quality=95, subsampling=0)),
                    ("jpg97", dict(format="JPEG", quality=97, subsampling=0))):
        buf = io.BytesIO()
        t0 = time.time(); im.save(buf, **kw); t_enc[tag].append(time.time() - t0)
        blob = buf.getvalue(); sizes[tag].append(len(blob))
        t0 = time.time(); back = cv2.imdecode(np.frombuffer(blob, np.uint8),
                                              cv2.IMREAD_COLOR)[:, :, ::-1]
        t_dec[tag].append(time.time() - t0)
        gray1 = cv2.cvtColor(np.ascontiguousarray(back), cv2.COLOR_RGB2GRAY)
        _b, hu1, fp1 = extract_binary_features(gray1, cfg)
        ham = float(np.unpackbits(np.bitwise_xor(fp0, fp1)).sum()) / (len(fp0) * 8)
        t1 = ex.transform(Image.fromarray(np.ascontiguousarray(back)))
        f1 = ex._forward([t1])[0]
        cos = float(f0 @ f1 / (np.linalg.norm(f0) * np.linalg.norm(f1)))
        res[tag]["cos"].append(cos)
        res[tag]["ham"].append(ham)

print(f"样本 {len(res['png']['cos'])} 张\n")
print(f"{'格式':<8}{'体积中位':>10}{'编码中位':>10}{'解码中位':>10}"
      f"{'特征余弦(最小)':>16}{'指纹汉明(最大)':>16}")
for tag in ("png", "jpg95", "jpg97"):
    if not res[tag]["cos"]:
        continue
    print(f"{tag:<8}{statistics.median(sizes[tag]) / 1024:9.1f}K"
          f"{statistics.median(t_enc[tag]) * 1000:9.2f}ms"
          f"{statistics.median(t_dec[tag]) * 1000:9.2f}ms"
          f"{min(res[tag]['cos']):16.6f}{max(res[tag]['ham']) * 100:15.3f}%")
print(f"\n（原始 256 短边位图 {statistics.median(sizes['raw256']) / 1024:.0f} KB/张 → "
      f"3.7 万张：PNG {statistics.median(sizes['png']) * 37683 / 2**30:.2f} GB / "
      f"JPEG95 {statistics.median(sizes['jpg95']) * 37683 / 2**30:.2f} GB）")
