# -*- coding: utf-8 -*-
"""按分发 manifest 回滚：把已移动文件搬回源，并恢复路径数组备份。

用法：
    python devtools/rollback_distribution.py --manifest <distribution_manifest_*.jsonl>

manifest 由 hybrid_search/distribution.py 写入；
同目录的 distribution_backup_* 保存了重写前的 *.paths.npy/*.fine_paths.npy。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_search.distribution import rollback  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="回滚图库分发")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--no-restore-backup", action="store_true",
                    help="只搬文件，不恢复路径数组")
    a = ap.parse_args()
    if not os.path.exists(a.manifest):
        print("manifest 不存在：" + a.manifest)
        return 2
    rep = rollback(a.manifest, restore_backup=not a.no_restore_backup)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
