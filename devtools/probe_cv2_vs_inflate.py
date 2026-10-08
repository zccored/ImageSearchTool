# -*- coding: utf-8 -*-
"""口径校正：cv2 全 PNG 解码 vs 纯 inflate，用**总量**口径给出"换 inflate 的理论上限"。

逻辑：PNG 解码 = inflate + 反滤波/色彩/拷贝。若 cv2 的全解码总时间已经 ≤ 纯 libdeflate
inflate 总时间，则说明 cv2 自带的 inflate 至少不慢于 libdeflate，换库**上限为零**；
否则上限 = cv2总时间 / libdeflate_inflate总时间（假定反滤波等开销不变）。

用法: python -E devtools/probe_cv2_vs_inflate.py [每档张数=6] [单张上限MB=192]
"""
import ctypes
import hashlib
import json
import os
import random
import struct
import sys
import time
import zlib
from collections import defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT, THIRD_PARTY_LIBDEFLATE  # noqa: E402

EXTS = [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"]
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAX_RAW_MB = int(sys.argv[2]) if len(sys.argv) > 2 else 192
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
CT = {0: "灰度", 2: "RGB", 3: "调色板", 4: "灰度+a", 6: "RGBA"}
BPP = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
DLL = os.environ.get("IMAGE_SEARCH_LIBDEFLATE_DLL", THIRD_PARTY_LIBDEFLATE)


def ihdr(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return w, h, data[24], data[25]


def idat_of(data):
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


def main() -> int:
    from hybrid_search.io_utils import collect_images
    pngs = [p for p in collect_images(GALLERY_ROOT, EXTS) if p.lower().endswith(".png")]
    random.seed(17)
    random.shuffle(pngs)
    buckets = defaultdict(list)
    for p in pngs:
        try:
            with open(p, "rb") as f:
                info = ihdr(f.read(33))
        except OSError:
            continue
        if not info or info[2] != 8:
            continue
        mp = info[0] * info[1] / 1e6
        bn = "0-1MP" if mp < 1 else "1-4MP" if mp < 4 else "4-12MP" if mp < 12 else "12+MP"
        k = "%s/%s" % (CT.get(info[3], "?"), bn)
        if len(buckets[k]) < N_EACH:
            buckets[k].append(p)
        if sum(len(v) for v in buckets.values()) >= N_EACH * 8:
            break

    lib = ctypes.CDLL(DLL)
    lib.libdeflate_alloc_decompressor.restype = ctypes.c_void_p
    lib.libdeflate_zlib_decompress_ex.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    lib.libdeflate_zlib_decompress_ex.restype = ctypes.c_int
    h = lib.libdeflate_alloc_decompressor()
    scratch = [None]

    def ldf(b, n):
        if scratch[0] is None or scratch[0].size < n:
            scratch[0] = np.empty(max(n, 1 << 20), dtype=np.uint8)
        ao = ctypes.c_size_t()
        rc = lib.libdeflate_zlib_decompress_ex(
            ctypes.c_void_p(h), b, len(b), scratch[0].ctypes.data_as(ctypes.c_void_p),
            n, None, ctypes.byref(ao))
        if rc != 0:
            raise RuntimeError("rc=%d" % rc)
        return scratch[0][:ao.value]

    items = []
    for k in sorted(buckets):
        for p in buckets[k]:
            data = open(p, "rb").read()
            info = ihdr(data)
            raw = (info[0] * BPP.get(info[3], 4) * info[2] // 8 + 1) * info[1]
            if raw > MAX_RAW_MB * 2 ** 20:
                continue
            items.append({"name": os.path.basename(p), "bytes": data,
                          "idat": idat_of(data), "raw": raw})

    cvt = {"cv2": 0.0, "pyzlib": 0.0, "libdeflate": 0.0}
    per = []
    for it in items:
        buf = np.frombuffer(it["bytes"], dtype=np.uint8)
        cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)                     # warmup
        t = {}
        for _ in range(REPS):
            t0 = time.perf_counter()
            cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            t["cv2"] = t.get("cv2", 0.0) + (time.perf_counter() - t0)
            t0 = time.perf_counter()
            zlib.decompress(it["idat"], 15, it["raw"])
            t["pyzlib"] = t.get("pyzlib", 0.0) + (time.perf_counter() - t0)
            t0 = time.perf_counter()
            ldf(it["idat"], it["raw"])
            t["libdeflate"] = t.get("libdeflate", 0.0) + (time.perf_counter() - t0)
        for k in t:
            cvt[k] += t[k] / REPS
        per.append({"name": it["name"], "raw_mb": round(it["raw"] / 2 ** 20, 1),
                    "cv2_ms": round(t["cv2"] / REPS * 1e3, 2),
                    "pyzlib_ms": round(t["pyzlib"] / REPS * 1e3, 2),
                    "libdeflate_ms": round(t["libdeflate"] / REPS * 1e3, 2)})

    total_raw = sum(i["raw"] for i in items)
    out = {"files": len(items), "raw_gb": round(total_raw / 2 ** 30, 3), "reps": REPS,
           "totals_s": {k: round(v, 3) for k, v in cvt.items()},
           "mb_s": {k: round(total_raw / v / 2 ** 20, 1) for k, v in cvt.items()},
           "per_image": per,
           "bound_cv2_over_libdeflate_inflate": round(cvt["cv2"] / max(cvt["libdeflate"], 1e-9), 3),
           "deflate_share_implied": round(cvt["pyzlib"] / max(cvt["cv2"], 1e-9), 3),
           "libdeflate_dll": DLL,
           "sha1": hashlib.sha1(open(DLL, "rb").read()).hexdigest()[:12]}
    print("样本 %d 张，解出 %.2f GB，%d 次重复" % (len(items), total_raw / 2 ** 30, REPS))
    print("总量口径：cv2 全解码 %.2f s（%.0f MB/s）｜ Python zlib 纯 inflate %.2f s（%.0f MB/s）"
          "｜ libdeflate 纯 inflate %.2f s（%.0f MB/s）"
          % (cvt["cv2"], total_raw / cvt["cv2"] / 2 ** 20,
             cvt["pyzlib"], total_raw / cvt["pyzlib"] / 2 ** 20,
             cvt["libdeflate"], total_raw / cvt["libdeflate"] / 2 ** 20))
    b = cvt["cv2"] / max(cvt["libdeflate"], 1e-9)
    print("上限：cv2 全解码 / libdeflate 纯 inflate = %.3f → %s"
          % (b, "上限为零（cv2 自带 inflate 已不慢于 libdeflate，换库无意义）" if b <= 1.0
             else "理论最多提速 %.2fx（且必须做到反滤波零成本）" % b))
    f = os.path.join(_HERE, "perf_reports",
                     "deflate_bound_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("JSON:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
