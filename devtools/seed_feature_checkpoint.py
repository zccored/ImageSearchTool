# -*- coding: utf-8 -*-
"""把现有完整索引迁移成断点特征检查点（不重新解码/前向）。

用法：
    python devtools/seed_feature_checkpoint.py --prefix F:\视频\.gallery_index\gallery [--force]

只写 `<prefix目录>/feature_checkpoint/`，不改任何 gallery.* 索引文件。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_search.coarse import CoarseRecord          # noqa: E402
from hybrid_search.config import Config                # noqa: E402
from hybrid_search.feature_checkpoint import FeatureCheckpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="从现有完整索引生成断点特征检查点")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--force", action="store_true", help="已有检查点时也重写")
    a = ap.parse_args()

    meta_path = a.prefix + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    mc = meta.get("cfg", {})

    cfg = Config()
    for k in ("model", "prep", "coarse_size", "coarse_blur", "binarize_norm",
              "invert_binary", "use_hu", "use_fp"):
        if k in mc and mc[k] is not None:
            setattr(cfg, k, mc[k])
    cfg.store_fine = True

    fine = np.load(a.prefix + ".fine.npy", mmap_mode="r")
    paths = [str(x) for x in np.load(a.prefix + ".paths.npy", allow_pickle=True)]
    hu = np.load(a.prefix + ".hu.npy", mmap_mode="r")
    fp = np.load(a.prefix + ".fp.npy", mmap_mode="r")
    md5 = [str(x) for x in np.load(a.prefix + ".md5s.npy", allow_pickle=True)]
    if fine.ndim != 2 or fine.shape[0] != len(paths):
        print("索引形状不匹配，拒绝")
        return 2

    ck = FeatureCheckpoint(a.prefix, cfg, int(fine.shape[1]), enabled=True)
    existing = ck.load(None)
    if existing and not a.force:
        print("检查点已存在 %d 条；加 --force 才重写" % len(existing))
        ck.close()
        return 0
    if existing or os.path.exists(ck.manifest_path):
        ck.reset()

    t = time.perf_counter()
    recs = [CoarseRecord(path=p, md5=m, hu=np.asarray(hu[i], dtype=np.float32),
                         fp=np.asarray(fp[i], dtype=np.uint8))
            for i, (p, m) in enumerate(zip(paths, md5))]
    n = ck.append(paths, np.asarray(fine, dtype=np.float32), recs, commit=False)
    ck.commit()
    ck.close()
    print("seeded %d/%d entries in %.1fs, %.1f MB"
          % (n, len(paths), time.perf_counter() - t, ck._bytes / 1e6))

    ck2 = FeatureCheckpoint(a.prefix, cfg, int(fine.shape[1]), enabled=True)
    loaded = ck2.load(None)
    print("verify", "ok" if loaded is not None else "FAIL", "entries",
          0 if loaded is None else len(loaded), "bytes", ck2._bytes)
    ck2.close()
    return 0 if loaded is not None else 3


if __name__ == "__main__":
    sys.exit(main())
