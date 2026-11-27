# -*- coding: utf-8 -*-
"""图库联邦 manifest 查看/校验工具（独立插件工具，不进入主流程）。

用法：
    python devtools/federation_tool.py --total "W:\\ImageGalleryTemp\\.gallery_index" --verify
    python devtools/federation_tool.py --start "D:\\kemono" --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_search.federation import discover, load_total, verify  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="图库联邦/链式索引 manifest 工具")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--start", help="分发目录 / 节点 marker / 总索引目录")
    g.add_argument("--total", help="总索引目录（读 federation.json）")
    ap.add_argument("--verify", action="store_true", help="校验全部节点与索引文件")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args()

    if a.verify:
        rep = verify(a.total or a.start)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if rep.get("ok") else 1
    if a.total:
        nodes = load_total(a.total)
    else:
        nodes = discover(a.start)
    rows = [n.as_dict() for n in nodes]
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        for n in rows:
            print("%s  root=%s  available=%s  files=%d  linked=%d" % (
                n["node_id"], n["root"], n["available"],
                len(n["index_files"]), len(n["linked"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
