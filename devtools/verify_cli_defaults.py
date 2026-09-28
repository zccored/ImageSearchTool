# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 验证：CLI 各子命令的 argparse 默认值不与 config.Config 漂移
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
验证 CLI 参数默认值没有和 `Config` 漂移（回归，判据 0/11）。

**为什么需要它**：`cli.main()` 对**所有**子命令都调 `cli._apply_feature_args()`，
把 argparse 解析出的值写进 `Config`。只要某个子命令的 argparse 默认值与
`hybrid_search/config.py` 的 `Config` 不一致，就会出现

    「CLI `build` 写进索引 meta 的参数」≠「`stats`/`compact` 读索引时的基准」
    →  stor._check_cfg_compat() 直接拒绝打开（rc=2）

的真实故障（2026-09-27 就是这么踩到的：`--png-decoder` 的 argparse 默认曾是 `"cv2"`，
而 `Config.png_decoder` 默认是 `"libdeflate"`；`--fast-load` 不传还会把侧车默认关掉）。

**判据**：对每个子命令用「不传任何开关」的参数集跑一遍 `_apply_feature_args()`，
结果必须与 pristine `Config()` **逐字段完全相等**；全部相等 → 打印 0/N → 退出码 0。

⚠️ 新增/修改 CLI 参数后必跑：要么让 argparse 默认值**等于** Config 默认（单一事实来源），
要么写成「不覆盖」（`default=None` / `store_true` + 仅显式传参才覆盖）。

用法: python -E devtools/verify_cli_defaults.py [--verbose]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search import cli                        # noqa: E402
from hybrid_search.config import Config              # noqa: E402

# 各子命令的“最小合法位置参数”（只为让 argparse 解析通过）
POSITIONAL = {
    "build": ["<图库目录>"],
    "add": ["<增量目录>"],
    "build-tiles": ["<图库目录>"],
    "add-tiles": ["<增量目录>"],
    "search": ["<查询图>"],
    "eval": ["<查询目录>"],
    "ingest": ["<request.json>"],
    "bench": ["<图库目录>"],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CLI 参数默认值 vs Config 漂移检查")
    ap.add_argument("--verbose", action="store_true",
                    help="打印每个子命令的逐字段结果（默认只打印漂移的）")
    args = ap.parse_args(argv)

    parser = cli.build_parser()
    # 从解析器本身取子命令清单，避免漏掉新加的命令
    subs = [a for a in parser._actions
            if isinstance(a, argparse._SubParsersAction)][0]
    cmds = list(subs.choices.keys())

    drifted = []
    for cmd in cmds:
        argvs = [cmd] + POSITIONAL.get(cmd, [])
        ns = parser.parse_args(argvs)
        cfg = Config()
        cli._apply_feature_args(cfg, ns)
        base = Config()
        diff = {f.name: [getattr(base, f.name), getattr(cfg, f.name)]
                for f in dataclasses.fields(Config)
                if getattr(base, f.name) != getattr(cfg, f.name)}
        if diff:
            drifted.append(cmd)
            print(f"✗ {cmd:<12} 漂移（Config 默认 → CLI 实际）: "
                  f"{json.dumps(diff, ensure_ascii=False)}")
        elif args.verbose:
            print(f"✓ {cmd:<12} 无漂移")

    print(f"\n子命令数: {len(cmds)}｜漂移: {len(drifted)}/{len(cmds)}")
    if drifted:
        print("判据不满足：请把上述 argparse 默认值改成与 Config 一致，"
              "或改成“仅显式传参才覆盖”。")
        return 1
    print("结果: 全部通过（argparse 默认值未改 Config）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
