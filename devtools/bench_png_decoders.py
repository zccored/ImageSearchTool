# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PNG 解码库横向对比（靶子图库 <图库根>，只读）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""PNG 解码库对比：cv2(libpng) / imagecodecs-libpng / libspng / Pillow / libvips。

两段：
  A) 纯 DEFLATE 解压（PNG 的 IDAT 流）：Python zlib vs imagecodecs zlib vs zlib-ng
     —— PNG 解码 84% 的时间在这里（前面已实测），所以这段决定上限。
  B) 完整解码：逐库单独计时（交替配对，消时钟漂移），并逐位比对是否与现方案一致
     （一致 = 换库不用重建索引、不用重新验证召回）。

只读：只读图片、不写任何索引、不删任何文件。
用法: python -E devtools/bench_png_decoders.py [每类张数] [轮数]
"""
import os
import random
import statistics as st
import struct
import sys
import time
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

GALLERY_INDEX = os.path.join(GALLERY_ROOT, ".gallery_index")
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def idat_of(data: bytes) -> bytes:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return b""
    i, out = 8, []
    while i + 8 <= len(data):
        (ln,) = struct.unpack(">I", data[i:i + 4])
        typ = data[i + 4:i + 8]
        if typ == b"IDAT":
            out.append(data[i + 8:i + 8 + ln])
        elif typ == b"IEND":
            break
        i += 12 + ln
    return b"".join(out)


# ---------------------------------------------------------------- 各库包装
def rgb_of_cv2(b):
    import cv2
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR_RGB)


def _to_rgb8(a):
    """把各库的原生输出规整成 cv2 语义的 RGB uint8（丢 alpha、16bit 取高 8 位）。"""
    if a is None:
        return None
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.ndim == 3 and a.shape[2] == 2:          # 灰度+alpha
        a = np.repeat(a[:, :, :1], 3, axis=2)
    return np.ascontiguousarray(a)


def make_imagecodecs(fn_name):
    import imagecodecs

    def dec(b):
        return _to_rgb8(getattr(imagecodecs, fn_name)(b))
    return dec


def rgb_of_pillow(b):
    import io

    from PIL import Image
    with Image.open(io.BytesIO(b)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


_VIPS = {"mod": None, "err": ""}


def rgb_of_vips(b):
    if _VIPS["mod"] is None and not _VIPS["err"]:
        try:
            import pyvips
            pyvips.concurrency_set(1)              # 与其它库同为单线程口径
            _VIPS["mod"] = pyvips
        except Exception as e:                     # noqa: BLE001
            _VIPS["err"] = repr(e)[:100]
    if _VIPS["mod"] is None:
        raise RuntimeError("libvips 不可用: " + _VIPS["err"])
    im = _VIPS["mod"].Image.new_from_buffer(b, "", access="sequential")
    if im.format == "uchar" and im.bands >= 3:
        im = im.extract_band(0, n=3)
    else:
        im = im.colourspace("srgb").extract_band(0, n=3)
    mem = im.write_to_memory()
    arr = np.ndarray(shape=(im.height, im.width, 3), dtype=np.uint8, buffer=mem)
    return np.ascontiguousarray(arr)


def main() -> int:
    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                     allow_pickle=True)]
    pngs = [p for p in paths if p.lower().endswith(".png")]
    random.seed(11)
    heavy_pool = pngs[:]           # 后续按文件大小排序挑大图
    heavy_pool.sort(key=lambda p: -os.path.getsize(p))
    heavy = heavy_pool[:N_EACH]
    rest = [p for p in pngs if p not in set(heavy)]
    mixed = random.sample(rest, min(N_EACH, len(rest)))
    groups = [("超大 PNG（磁盘最大 %d 张）" % N_EACH, heavy),
              ("随机 PNG %d 张" % N_EACH, mixed)]

    print("=" * 84)
    print("[A] 纯 DEFLATE 解压（PNG 的 IDAT 流，PNG 解码 84% 的时间所在）")
    try:
        import imagecodecs
        variants = [("Python zlib " + zlib.ZLIB_VERSION, lambda b: zlib.decompress(b)),
                    ("imagecodecs zlib " + imagecodecs.zlib_version().split()[-1],
                     lambda b: imagecodecs.zlib_decode(b)),
                    ("imagecodecs zlib-ng " + imagecodecs.zlibng_version().split()[-1],
                     lambda b: imagecodecs.zlibng_decode(b))]
    except Exception as e:                         # noqa: BLE001
        print("   imagecodecs 不可用:", e)
        variants = [("Python zlib " + zlib.ZLIB_VERSION, lambda b: zlib.decompress(b))]
    blobs = []
    for p in mixed + heavy:
        try:
            d = open(p, "rb").read()
        except OSError:
            continue
        i = idat_of(d)
        if len(i) > 1024:
            blobs.append(i)
    print("   样本 %d 个 IDAT 流，合计 %.1f MB（解出 %.1f MB）"
          % (len(blobs), sum(map(len, blobs)) / 2 ** 20,
             sum(len(zlib.decompress(b)) for b in blobs) / 2 ** 20))
    raw_total = sum(len(zlib.decompress(b)) for b in blobs)
    for name, fn in variants:
        t0 = time.perf_counter()
        for b in blobs:
            fn(b)
        el = time.perf_counter() - t0
        print("   %-32s %6.3f s  单流中位 %6.1f ms  %6.0f MB/s(解出)"
              % (name, el, el / len(blobs) * 1000, raw_total / el / 2 ** 20))

    print()
    print("=" * 84)
    print("[B] 完整 PNG 解码（交替配对计时，等价 RGB uint8 输出）")
    decoders = [("cv2(现状 libpng)", rgb_of_cv2),
                ("Pillow", rgb_of_pillow)]
    try:
        import imagecodecs  # noqa: F401
        decoders += [("imagecodecs png(libpng %s)" % imagecodecs.png_version().split()[-1],
                      make_imagecodecs("png_decode")),
                     ("imagecodecs spng(libspng %s)" % imagecodecs.spng_version().split()[-1],
                      make_imagecodecs("spng_decode"))]
    except Exception as e:                         # noqa: BLE001
        print("   imagecodecs 不可用:", e)
    decoders.append(("libvips(pyvips)", rgb_of_vips))

    for gname, files in groups:
        data = []
        for p in files:
            try:
                data.append((p, open(p, "rb").read()))
            except OSError:
                pass
        if not data:
            continue
        print("\n--- %s（%d 张，合计 %.1f MB）---"
              % (gname, len(data), sum(len(b) for _, b in data) / 2 ** 20))
        # 先做逐位一致性（以 cv2 为基准）
        ref = {p: rgb_of_cv2(b) for p, b in data}
        for name, fn in decoders:
            same = bad = err = 0
            for p, b in data:
                try:
                    a = fn(b)
                except Exception:                  # noqa: BLE001
                    err += 1
                    continue
                if a is None:
                    err += 1
                elif a.shape == ref[p].shape and np.array_equal(a, ref[p]):
                    same += 1
                else:
                    bad += 1
            print("   %-34s 与现方案逐位一致 %2d/%d  不一致 %d  失败 %d"
                  % (name, same, len(data), bad, err))
        # 交替配对计时
        acc = {name: 0.0 for name, _ in decoders}
        n_ok = {name: 0 for name, _ in decoders}
        for rnd in range(ROUNDS):
            for k, (p, b) in enumerate(data):
                order = decoders if (k + rnd) % 2 == 0 else list(reversed(decoders))
                for name, fn in order:
                    t0 = time.perf_counter()
                    try:
                        fn(b)
                    except Exception:              # noqa: BLE001
                        continue
                    acc[name] += time.perf_counter() - t0
                    n_ok[name] += 1
        print("   %-34s %10s %10s %8s" % ("库", "中位ms/张", "相对现状", "MB/s"))
        base = None
        for name, _ in decoders:
            if n_ok[name] == 0:
                print("   %-34s %10s" % (name, "不可用"))
                continue
            ms = acc[name] / n_ok[name] * 1000
            if base is None:
                base = ms
            mb = sum(len(b) for _, b in data) * ROUNDS / acc[name] / 2 ** 20
            print("   %-34s %10.1f %9.2fx %8.0f" % (name, ms, base / ms, mb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
