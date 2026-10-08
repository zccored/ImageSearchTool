# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 回退本轮实验性改动（大图旁路 / cv2_threads）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE.
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""把本轮两个**净负收益**的实验改动回退到改动前状态：
  ① png_fast：大图（>=4MP）RGB 也走旁路 → 恢复为"仅 RGBA 走旁路"
  ② cv2_threads：config/engine/cli/ab_build_bench 里的旋钮与 setNumThreads 调用
回退后由调用方跑完整性校验（单元测试 + 逐位一致 + 导入/CLI）。

用法: python -E devtools/revert_round.py [--dry]
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP_KEYS = ("cv2_threads", "_BIG_BYPASS_PX", "CV2_THREADS")
# 只删"本轮新加"的注释行，避免误伤历史注释
DROP_COMMENT_KEYS = ("OpenCV 内部并行线程数", "跨图解码已由本项目自己的线程池并行",
                     "OpenCV 内部线程数", "大图额外开旁路阈值", "RGBA 一律走旁路",
                     "实测 >=4MP 只占", "环境变量透传", "见 _BYPASS_CT：不值得开旁路的档")
FILES = ["hybrid_search/png_fast.py", "hybrid_search/config.py", "hybrid_search/engine.py",
         "hybrid_search/cli.py", "devtools/ab_build_bench.py"]
RESTORE_IF = '    if ctype not in _BYPASS_CT:                    # 见 _BYPASS_CT：不值得开旁路的档'
SKIP_BUMP = '        _bump("skip_ctype%d" % ctype)'


def main() -> int:
    dry = "--dry" in sys.argv
    for rel in FILES:
        p = os.path.join(_HERE, rel)
        if not os.path.exists(p):
            print("SKIP 缺失", rel)
            continue
        src = open(p, encoding="utf-8").read()
        lines, out, changed = src.split("\n"), [], 0
        for ln in lines:
            s = ln.strip()
            if any(k in ln for k in DROP_KEYS) or any(k in ln for k in DROP_COMMENT_KEYS):
                changed += 1
                continue
            out.append(ln)
        # 修掉因删行而失去 if 的 _bump（保持缩进正确）
        for i, ln in enumerate(out):
            if ln == SKIP_BUMP and (i == 0 or not out[i - 1].lstrip().startswith("if ")):
                out.insert(i, RESTORE_IF)
                changed += 1
        # 收掉可能留下的连续空行（最多留两行）
        cleaned = []
        for ln in out:
            if ln.strip() == "" and cleaned[-2:] == ["", ""]:
                continue
            cleaned.append(ln)
        new = "\n".join(cleaned)
        if changed and not dry:
            open(p, "w", encoding="utf-8").write(new)
        print("%-32s 删除/修正 %d 行%s" % (rel, changed, "（dry）" if dry else ""))
    print("回退完成" if not dry else "仅预演，未写盘")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
