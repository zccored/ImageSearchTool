# -*- coding: utf-8 -*-
"""在大图（>8MP）上对比 cv2 预处理与 torchvision 的耗时与特征差异。"""
import os
import pickle
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
from hybrid_search.fine import ResNetExtractor, _PRE_DOWNSCALE_PX, _PRE_DOWNSCALE_SIDE  # noqa: E402
from hybrid_search.io_utils import decode_rgb, read_bytes  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "perf_reports", "dedup_report_F视频.pkl")
rep = pickle.load(open(CACHE, "rb"))
cand = []
for g in rep.groups[:200]:
    for m in g.members[:1]:
        if m.w * m.h > 8_000_000 and os.path.exists(m.path):
            cand.append(m.path)
    if len(cand) >= 10:
        break
print(f"大图样本 {len(cand)} 张（{cand[0][:60] if cand else '-'}）")

ex = ResNetExtractor(Config())
pil_t, cv_t, cos = [], [], []
for p in cand:
    rgb = decode_rgb(read_bytes(p))
    if rgb is None:
        continue
    if rgb.shape[0] * rgb.shape[1] > _PRE_DOWNSCALE_PX:
        s = _PRE_DOWNSCALE_SIDE / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * s)),
                               max(1, int(rgb.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    t0 = time.time(); a = ex.transform(Image.fromarray(rgb)); pil_t.append(time.time() - t0)
    t0 = time.time(); b = ex.prep(rgb); cv_t.append(time.time() - t0)
    fa = ex._forward([a])[0]
    fb = ex._forward([b])[0]
    cos.append(float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb))))
    print(f"  {os.path.basename(p)[:34]:<34} {rgb.shape[1]}x{rgb.shape[0]}  "
          f"PIL {pil_t[-1] * 1000:6.1f}ms  cv2 {cv_t[-1] * 1000:6.1f}ms  "
          f"cos {cos[-1]:.5f}")
print(f"\n均值：PIL {np.mean(pil_t) * 1000:.1f} ms，cv2 {np.mean(cv_t) * 1000:.1f} ms，"
      f"提速 {np.mean(pil_t) / np.mean(cv_t):.2f}x；特征余弦 {np.mean(cos):.5f}"
      f"（最小 {np.min(cos):.5f}）")
