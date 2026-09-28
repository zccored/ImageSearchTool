# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — devtools 本机路径的唯一来源
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""devtools 里所有「本机路径」的唯一来源（把真实路径从源码里赶出去）。

**为什么要有这个文件**：这些脚本原先把真实图库根、靶子图、仓库绝对路径硬编码在源码里，
公开仓库会泄露本机目录结构。现在统一改成三层取值，优先级从高到低：

  1. **环境变量**（`GALLERY_ROOT` / `TARGET_IMAGE` / `REPO_ROOT` / `THIRD_PARTY_LIBDEFLATE`）
  2. **`devtools/local_paths.py`** —— 本机真实路径放这里，该文件已被 `.gitignore` 排除，**不入库**
  3. **占位符**（`<图库根>` 等）：取值失败时不静默跑错，而是在真正用到时报清晰错误

**本机一次性初始化**（之后所有 devtools 脚本照旧直接跑）：

    python -E devtools/paths.py --init     # 生成 devtools/local_paths.py 模板，填上你的路径
    python -E devtools/paths.py --show     # 看看当前各值解析成什么

脚本用法：

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import GALLERY_ROOT, TARGET_IMAGE      # 或 require("GALLERY_ROOT")
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = os.path.join(_HERE, "local_paths.py")

# 变量名 -> (占位符, 用途说明)
SPEC = {
    "GALLERY_ROOT": ("<图库根>", "靶子图库根目录（只读；索引写独立目录）"),
    "TARGET_IMAGE": ("<图库根>/<靶子图>.png", "检索/自检用的目标查询图"),
    "REPO_ROOT": ("<仓库根>", "本仓库的绝对路径（脚本自举 sys.path 用）"),
    "THIRD_PARTY_LIBDEFLATE": ("<第三方 libdeflate.dll>", "对照用的第三方 libdeflate.dll 路径"),
}


def _from_local(name):
    """从 devtools/local_paths.py（不入库）取值；文件不存在/没有该项则返回 None。"""
    if not os.path.exists(_LOCAL):
        return None
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        import local_paths
    except Exception:                                     # noqa: BLE001 —— 坏文件不该炸脚本
        return None
    v = getattr(local_paths, name, None)
    return v if isinstance(v, str) and v.strip() else None


def get(name, env=None):
    """按 环境变量 → local_paths.py → 占位符 取值。"""
    v = os.environ.get(name)
    if v and v.strip():
        return v
    v = _from_local(name)
    if v:
        return v
    return SPEC.get(name, ("", ""))[0]


def is_placeholder(v):
    return not v or (v.startswith("<") and v.endswith(">"))


def require(name):
    """要真正用这个路径时调用：仍是占位符就带清楚提示抛错，避免"跑半天没结果"。"""
    v = get(name)
    if is_placeholder(v):
        raise SystemExit(
            "[paths] %s 未配置（当前值 %r）。\n"
            "        用途：%s\n"
            "        解决：设环境变量 %s，或运行 python -E devtools/paths.py --init\n"
            "        生成 devtools/local_paths.py 后填上本机路径（该文件不入库）。"
            % (name, v, SPEC.get(name, ("", "?"))[1], name))
    return v


# 模块级常量：脚本 `from paths import GALLERY_ROOT` 即可
GALLERY_ROOT = get("GALLERY_ROOT")
TARGET_IMAGE = get("TARGET_IMAGE")
REPO_ROOT = get("REPO_ROOT")
THIRD_PARTY_LIBDEFLATE = get("THIRD_PARTY_LIBDEFLATE")

_TEMPLATE = '''# -*- coding: utf-8 -*-
"""本机路径（**不入库**，已被 .gitignore 排除）。填好即可，所有 devtools 脚本会自动读到。"""

GALLERY_ROOT = r"%s"
TARGET_IMAGE = r"%s"
REPO_ROOT = r"%s"
# 下面这项只有做第三方 libdeflate 对照的脚本用得到；没有就留空
THIRD_PARTY_LIBDEFLATE = r"%s"
'''


def _init():
    if os.path.exists(_LOCAL):
        print("[paths] 已存在，不覆盖：%s" % _LOCAL)
        return 0
    with open(_LOCAL, "w", encoding="utf-8") as f:
        f.write(_TEMPLATE % tuple(SPEC[k][0] for k in
                                  ("GALLERY_ROOT", "TARGET_IMAGE", "REPO_ROOT",
                                   "THIRD_PARTY_LIBDEFLATE")))
    print("[paths] 已生成模板：%s\n        请把里面的占位符改成你的真实路径。" % _LOCAL)
    return 0


def _show():
    print("local_paths.py: %s（%s）" % (_LOCAL, "存在" if os.path.exists(_LOCAL) else "不存在"))
    for k in SPEC:
        src = ("环境变量" if os.environ.get(k) else
               "local_paths.py" if _from_local(k) else "占位符")
        print("  %-24s %-28s [%s]" % (k, get(k), src))
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "--show"
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    if arg == "--init":
        raise SystemExit(_init())
    raise SystemExit(_show())
