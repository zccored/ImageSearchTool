# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 原生扩展构建脚本
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""构建 hybrid_search/native/_pngfast 扩展（Cython → C → 共享库）。

本机实测可用路线（无 MSVC / 无 cmake）：Cython 生成 C，再用 MinGW gcc 直接编译成 .pyd，
不需要 setuptools 的构建后端（distutils 在 3.12 已移除，MinGW 后端容易翻车）。

用法:
    python -E devtools/build_native.py            # 构建（已存在且更新则跳过）
    python -E devtools/build_native.py --force    # 强制重建
构建产物：hybrid_search/native/_pngfast.<ext-suffix>（不随源码分发，本机按需生成）
"""
import glob
import os
import shutil
import subprocess
import sys
import sysconfig

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYX = os.path.join(_HERE, "hybrid_search", "native", "_pngfast.pyx")
BUILD = os.path.join(_HERE, "build", "native")


def main() -> int:
    force = "--force" in sys.argv
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    out = os.path.join(_HERE, "hybrid_search", "native", "_pngfast" + suffix)
    if os.path.exists(out) and not force and os.path.getmtime(out) > os.path.getmtime(PYX) \
            and os.path.getmtime(out) > os.path.getmtime(
                os.path.join(_HERE, "hybrid_search", "native", "png_filter_simd.c")):
        print("已是最新：", out)
        return 0
    os.makedirs(BUILD, exist_ok=True)
    cfile = os.path.join(BUILD, "_pngfast.c")

    cy = shutil.which("cython") or shutil.which("cython3")
    if not cy:
        try:
            from Cython.Compiler.Main import compile as cy_compile, CompilationOptions
            from Cython.Compiler.Main import default_options
            print("用 Cython 模块 API 生成 C …")
            opts = CompilationOptions(default_options, output_file=cfile,
                                      language_level=3)
            res = cy_compile(PYX, options=opts)
            if res.num_errors:
                print("Cython 编译失败", res.num_errors)
                return 1
        except Exception as e:                     # noqa: BLE001
            print("Cython 不可用：", e)
            return 1
    else:
        cmd = [cy, "-3", "-o", cfile, PYX]
        print("运行:", " ".join(cmd))
        if subprocess.call(cmd) != 0:
            return 1

    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        print("找不到 gcc/cc：本机需要 MinGW（设置 MINGW_HOME，例如 <MinGW>\\bin\\gcc.exe）")
        return 1
    inc = sysconfig.get_paths()["include"]
    libs = os.path.join(sys.base_prefix, "libs")
    nat = os.path.join(_HERE, "hybrid_search", "native")
    csrc = os.path.join(nat, "png_filter_simd.c")          # SIMD 反滤波内核（第 2 个翻译单元）
    cmd = [cc, "-O3", "-shared", "-fno-strict-aliasing", "-I" + nat,
           "-I" + inc, "-o", out, cfile, csrc]
    if os.name == "nt":
        cmd += ["-L" + libs, "-lpython%d%d" % sys.version_info[:2]]
    else:
        cmd += ["-fPIC"]
    print("运行:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("编译失败 rc=", rc)
        return rc
    print("构建完成:", out, os.path.getsize(out), "bytes")
    # 自检：在当前进程里 import 一次
    sys.path.insert(0, os.path.join(_HERE, "hybrid_search", "native"))
    try:
        import importlib
        m = importlib.import_module("_pngfast")
        print("import 自检 OK:", m.__file__)
    except Exception as e:                         # noqa: BLE001
        print("import 自检失败：", type(e).__name__, e)
        return 1
    stale = [p for p in glob.glob(os.path.join(_HERE, "hybrid_search", "native",
                                               "_pngfast*")) if p.endswith(suffix) and p != out]
    for p in stale:
        print("提示：存在旧后缀副本（未删除）：", os.path.basename(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
