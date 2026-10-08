# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 自己编一份 libdeflate.dll
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""用 MinGW gcc 把 libdeflate 源码编成 hybrid_search/native/libdeflate.dll。

为什么要自己编：`png_fast` 的加载顺序是
    IMAGE_SEARCH_LIBDEFLATE_DLL → hybrid_search/native/libdeflate.dll → 兜底第三方随包 DLL
兜底那份来路不明（某网盘客户端的随包 DLL），只适合测量，不适合作为正式依赖。
把自编的 DLL 放到 hybrid_search/native/ 即可自动优先使用。

源码获取（任选其一，本机网络对 pypi 镜像的 sdist 下载很慢，建议手动下）：
  * https://github.com/ebiggers/libdeflate/releases 的 libdeflate-1.2x.tar.gz
  * pip 下载 imagecodecs 的 sdist（内含 3rdparty/libdeflate 源码树）

用法:
    python -E devtools/build_libdeflate.py <libdeflate源码目录或tar.gz> [--check]
    python -E devtools/build_libdeflate.py --check          # 只验证当前生效的 DLL
"""
import ctypes
import glob
import hashlib
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

NATIVE = os.path.join(_HERE, "hybrid_search", "native")
OUT = os.path.join(NATIVE, "libdeflate.dll")


def sha1(p: str) -> str:
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:16]


def check(dll: str = None, trials: int = 400) -> int:
    """行为验证：导出符号齐全 + 随机/真实流上与原 zlib 逐位一致。"""
    if dll is None:
        from hybrid_search import png_fast
        pair = png_fast._load_libdeflate()
        if not pair:
            print("当前没有可用的 libdeflate：", png_fast._LDF_REASON)
            return 1
        dll = pair[1]
    lib = ctypes.CDLL(dll)
    need = ("libdeflate_alloc_decompressor", "libdeflate_free_decompressor",
            "libdeflate_zlib_decompress_ex")
    for s in need:
        try:
            getattr(lib, s)
        except AttributeError:
            print("缺少导出符号:", s, "（该 DLL 不可用）")
            return 1
    lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
    lib.libdeflate_zlib_decompress_ex.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    lib.libdeflate_zlib_decompress_ex.restype = ctypes.c_int
    h = lib.libdeflate_alloc_decompressor()
    rng = random.Random(7)
    ok = bad = 0
    for i in range(trials):
        if i % 2:
            n = rng.randint(0, 1 << 16)
            raw = bytes(rng.getrandbits(8) for _ in range(min(n, 4096))) * max(1, n // 4096)
        else:
            raw = (b"png-scanline-like\x00" * rng.randint(1, 4000))
        z = zlib.compress(raw, rng.choice([0, 1, 6, 9]))
        buf = np.empty(len(raw) or 1, np.uint8)
        ao = ctypes.c_size_t()
        rc = lib.libdeflate_zlib_decompress_ex(
            ctypes.c_void_p(h), z, len(z), buf.ctypes.data_as(ctypes.c_void_p),
            len(raw), None, ctypes.byref(ao))
        if rc == 0 and ao.value == len(raw) and buf[:ao.value].tobytes() == raw:
            ok += 1
        else:
            bad += 1
    print("DLL: %s\n  sha1=%s  大小=%d bytes" % (dll, sha1(dll), os.path.getsize(dll)))
    print("  行为验证 %d 次往返：逐位一致 %d，异常 %d" % (trials, ok, bad))
    return 0 if bad == 0 else 1


def build(src: str) -> int:
    tmp = None
    if os.path.isfile(src) and src.lower().endswith((".tar.gz", ".tgz")):
        tmp = tempfile.mkdtemp(prefix="libdeflate_")
        with tarfile.open(src) as t:
            t.extractall(tmp)
        cands = [d for d in glob.glob(os.path.join(tmp, "*")) if os.path.isdir(d)]
        src = cands[0] if cands else tmp
        # imagecodecs sdist 里源码在 3rdparty/libdeflate
        deep = glob.glob(os.path.join(src, "**", "libdeflate.h"), recursive=True)
        if deep:
            src = os.path.dirname(deep[0])
    if not os.path.exists(os.path.join(src, "libdeflate.h")):
        print("没找到 libdeflate.h，路径不对：", src)
        return 1
    ccs = sorted(glob.glob(os.path.join(src, "lib", "*.c"))) + \
        sorted(glob.glob(os.path.join(src, "lib", "x86", "*.c")))
    if not ccs:
        print("没找到 lib/*.c：", src)
        return 1
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        print("找不到 gcc（需要 MinGW）")
        return 1
    os.makedirs(NATIVE, exist_ok=True)
    cmd = [cc, "-O2", "-shared", "-DLIBDEFLATE_DLL", "-I" + src, "-I" + os.path.join(src, "lib"),
           "-o", OUT] + ccs
    print("编译 %d 个源文件 → %s" % (len(ccs), OUT))
    rc = subprocess.call(cmd)
    if rc != 0:
        # 某些版本 x86 目录不存在/不需要
        cmd = [cc, "-O2", "-shared", "-I" + src, "-o", OUT] + \
            sorted(glob.glob(os.path.join(src, "lib", "*.c")))
        print("重试（不带 x86 目录）：", " ".join(cmd[:6]), "…")
        rc = subprocess.call(cmd)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    if rc != 0 or not os.path.exists(OUT):
        print("编译失败 rc=", rc)
        return rc or 1
    print("构建完成:", OUT, os.path.getsize(OUT), "bytes")
    return check(OUT)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--check" in sys.argv and not args:
        return check()
    if not args:
        print(__doc__)
        return check()
    rc = build(args[0])
    if rc == 0 and "--check" in sys.argv:
        return check()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
