# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 获取 TurboJPEG 官方/conda-forge 预编译 DLL
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""取 libjpeg-turbo 的 turbojpeg.dll（C2 对照用）。

**为什么走 conda-forge，而不跑官方安装器**：
  * 官方 Windows 二进制 `libjpeg-turbo-<ver>-vc-x64.exe` 是 **NSIS 安装器**，
    静默安装会写注册表/可能改 PATH —— 与"不擅自改用户环境"的纪律冲突；
  * conda-forge 的 `.conda` 包就是 zip(zstd(tar))，**纯 Python 就能解**，不跑任何安装器；
  * 实测解出的 turbojpeg.dll 只导入 KERNEL32.dll（无 MSVC 运行时、无 libwinpthread、
    无 zlib 依赖），与项目现有的 libdeflate.dll 一样是"单文件即插即用"。

zstd 解压依赖（任一）：imagecodecs（本项目 PNG 可选依赖）/ zstandard / pyzstd。

用法:
  python -E devtools/fetch_turbojpeg.py                    # 解到 %TEMP%\\c2_jpeg
  python -E devtools/fetch_turbojpeg.py --out <目录>
  python -E devtools/fetch_turbojpeg.py --check            # 顺带做行为自检
  python -E devtools/fetch_turbojpeg.py --proxy http://127.0.0.1:7897
"""
import argparse
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import urllib.request
import zipfile
import zlib

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_VER = "3.2.0"
DEFAULT_BUILD = "hfd05255_1"
CHANNEL = "https://conda.anaconda.org/conda-forge/win-64"


def zstd_decode(buf: bytes) -> bytes:
    """zstd 解压（尽量用已有依赖，不额外装包）。"""
    try:
        import imagecodecs
        return bytes(imagecodecs.zstd_decode(buf))
    except Exception:                                     # noqa: BLE001
        pass
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompressobj().decompress(buf)
    except Exception:                                     # noqa: BLE001
        pass
    try:
        import pyzstd
        return pyzstd.decompress(buf)
    except Exception as e:                                # noqa: BLE001
        raise RuntimeError(
            "需要 zstd 解压能力：请 pip install zstandard（或 imagecodecs / pyzstd）") from e


def fetch(url: str, dst: str, proxy: str = "") -> str:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("   已存在，跳过下载：%s" % dst)
        return dst
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    op = urllib.request.build_opener(*handlers)
    print("   下载 %s" % url)
    with op.open(url, timeout=180) as r, open(dst + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dst + ".part", dst)
    print("   -> %s（%d B）" % (dst, os.path.getsize(dst)))
    return dst


def extract_turbojpeg(pkg: str, outdir: str, names=("turbojpeg.dll", "jpeg8.dll")) -> dict:
    """从 .conda（zip 内含 pkg-*.tar.zst）里抽出所需 DLL。"""
    got = {}
    with zipfile.ZipFile(pkg) as z:
        inner = [n for n in z.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst")]
        if not inner:
            raise RuntimeError("包内没有 pkg-*.tar.zst：%s" % z.namelist())
        raw = zstd_decode(z.read(inner[0]))
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for m in tf.getmembers():
                base = os.path.basename(m.name)
                if m.isfile() and base in names:
                    dst = os.path.join(outdir, base)
                    with open(dst, "wb") as f:
                        f.write(tf.extractfile(m).read())
                    got[base] = dst
    return got


def dll_imports(path: str):
    """读 PE 导入表（只为确认"无额外运行时依赖"）。"""
    b = open(path, "rb").read()
    pe = struct.unpack_from("<I", b, 0x3C)[0]
    nsec, = struct.unpack_from("<H", b, pe + 6)
    soh, = struct.unpack_from("<H", b, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", b, opt)
    ddoff = opt + (112 if magic == 0x20B else 96)
    imp_rva, _ = struct.unpack_from("<II", b, ddoff + 8)
    st = pe + 24 + soh
    secs = []
    for i in range(nsec):
        o = st + 40 * i
        vsz, va, rsz, ra = struct.unpack_from("<IIII", b, o + 8)
        secs.append((va, vsz, ra, rsz))

    def off(rva):
        for va, vsz, ra, rsz in secs:
            if va <= rva < va + max(vsz, rsz):
                return ra + (rva - va)
        return None

    out, i = [], off(imp_rva)
    if i is None:
        return out
    while True:
        _ol, _ts, _fc, nrva, _frva = struct.unpack_from("<IIIII", b, i)
        if nrva == 0:
            break
        o = off(nrva)
        end = b.index(b"\x00", o)
        out.append(b[o:end].decode("latin1"))
        i += 20
    return out


def check(dll: str) -> int:
    """行为自检：导出符号 + 与 cv2 在同一缩放档下逐位一致 + 灰度/渐进样例。"""
    import ctypes
    import numpy as np
    import cv2
    from PIL import Image

    print("\n[--check] 1) 导出符号")
    lib = ctypes.WinDLL(dll)
    need = ("tj3Init", "tj3Decompress8", "tj3DecompressHeader", "tj3Destroy",
            "tjInitDecompress", "tjDecompress2")
    miss = []
    for s in need:
        try:
            getattr(lib, s)
        except AttributeError:
            miss.append(s)
    print("   必需符号缺失：%s" % (miss or "无"))

    print("[--check] 2) 依赖")
    imps = dll_imports(dll)
    print("   导入表：%s" % imps)

    print("[--check] 3) 与 cv2 同档逐位一致（合成样例）")
    from turbojpeg import TJPF_RGB, TurboJPEG
    tj = TurboJPEG(dll)
    rng = np.random.default_rng(3)
    bad = 0
    cases = []
    for side in (512, 3000, 6000):                       # 覆盖全解 / 1/2 / 1/4 三档
        base = (rng.random((side, side, 3)) * 255).astype(np.uint8)
        for kw, tag in (({"quality": 85, "subsampling": 0}, "4:4:4"),
                        ({"quality": 85, "subsampling": 2}, "4:2:0"),
                        ({"quality": 85, "progressive": True}, "渐进")):
            buf = io.BytesIO()
            Image.fromarray(base).save(buf, "JPEG", **kw)
            cases.append(("%dx%d %s" % (side, side, tag), buf.getvalue()))
    buf = io.BytesIO()
    Image.fromarray((rng.random((2000, 2000)) * 255).astype(np.uint8)).save(
        buf, "JPEG", quality=85, **{"subsampling": 2})
    cases.append(("2000x2000 灰度", buf.getvalue()))

    scale = {512: (1, 1), 3000: (1, 2), 6000: (1, 4)}
    n_ok = 0
    for tag, data in cases:
        w, h = tj.decode_header(data)[:2]
        sf = scale.get(min(scale, key=lambda s: abs(s - max(w, h))), (1, 1))
        flag = {(1, 1): cv2.IMREAD_COLOR_RGB, (1, 2): cv2.IMREAD_REDUCED_COLOR_2,
                (1, 4): cv2.IMREAD_REDUCED_COLOR_4}[sf]
        a = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        b = tj.decode(data, pixel_format=TJPF_RGB, scaling_factor=sf)
        same = a.shape == b.shape and np.array_equal(a, b)
        n_ok += bool(same)
        print("   %-22s 缩放 1/%d  %s  cv2 %s / tj %s"
              % (tag, sf[1], "逐位一致" if same else "**不一致**", a.shape, b.shape))
        bad += (not same)
    print("[--check] 结论：%s（%d/%d 一致）"
          % ("通过" if (not bad and not miss) else "**失败**", n_ok, len(cases)))
    return 0 if (not bad and not miss) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=DEFAULT_VER)
    ap.add_argument("--build", default=DEFAULT_BUILD)
    ap.add_argument("--out", default=os.path.join(os.environ.get("TEMP", "."), "c2_jpeg"))
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    fn = "libjpeg-turbo-%s-%s.conda" % (a.version, a.build)
    pkg = os.path.join(a.out, fn)
    print("== 取 libjpeg-turbo %s（conda-forge win-64）" % a.version)
    fetch("%s/%s" % (CHANNEL, fn), pkg, a.proxy)
    print("   sha256 %s" % hashlib.sha256(open(pkg, "rb").read()).hexdigest()[:32])
    got = extract_turbojpeg(pkg, a.out)
    for k, v in sorted(got.items()):
        print("   抽出 %-16s %10d B  %s" % (k, os.path.getsize(v), v))
    if "turbojpeg.dll" not in got:
        print("**失败**：包里没有 turbojpeg.dll"); return 1
    meta = {"version": a.version, "build": a.build, "channel": CHANNEL, "package": fn,
            "size": os.path.getsize(got["turbojpeg.dll"]),
            "sha256": hashlib.sha256(open(got["turbojpeg.dll"], "rb").read()).hexdigest(),
            "crc32": "%08x" % (zlib.crc32(open(got["turbojpeg.dll"], "rb").read()) & 0xffffffff),
            "imports": dll_imports(got["turbojpeg.dll"]), "extracted": sorted(got)}
    mf = os.path.join(a.out, "turbojpeg.build.json")
    with open(mf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("   清单 %s" % mf)
    rc = check(got["turbojpeg.dll"]) if a.check else 0
    print("\n完成。设 TURBOJPEG_DLL=%s 可让基准脚本用它。" % got["turbojpeg.dll"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
