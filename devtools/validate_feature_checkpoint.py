# -*- coding: utf-8 -*-
r"""校验断点特征检查点（结构层：签名/CRC/哈希/维度/范数/stat）。

用法：
    python devtools/validate_feature_checkpoint.py \
        --prefix D:\...\gallery [--model resnet18] [--feature-dim 512] [--json report.json]

退出码 0=可信；1=不可信/不存在。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_search.config import Config          # noqa: E402
from hybrid_search.feature_checkpoint import FeatureCheckpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="断点特征检查点结构校验")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--feature-dim", type=int, default=512)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cfg = Config()
    cfg.model = a.model
    cfg.prep = "torch"
    ck = FeatureCheckpoint(a.prefix, cfg, a.feature_dim, enabled=True)
    entries = ck.load(None)
    report = {
        "prefix": a.prefix,
        "root": ck.root,
        "ok": entries is not None,
        "entries": 0 if entries is None else len(entries),
        "records": ck._n_records,
        "bytes": ck._bytes,
        "skipped_changed": ck.skipped_changed,
        "skipped_missing": ck.skipped_missing,
        "signature": ck.sig,
    }
    ck.close()
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
