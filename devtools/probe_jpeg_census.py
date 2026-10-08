# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — JPEG 全库普查（只读表头，不读像素）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""JPEG 全库普查：C2（换解码器）的样本加权依据。

只读每个文件的头部（默认前 256KB，SOF/APP1 必在其中），**不解码像素、不写索引**。

统计项：
  * 基线/渐进、精度(8/12bit)、分量数、采样比（4:4:4 / 4:2:0 …）
  * 尺寸分档 —— 直接映射 io_utils._reduced_flag 的 4 个档（全解/1/2/1/4/1/8），
    这决定"换解码器"能不能保住 DCT 域缩放的收益（imagecodecs 不支持缩放解码）
  * EXIF 方向分布 —— cv2 的 imdecode 会自动转正；换成 PyTurboJPEG/imagecodecs
    必须自己用 numpy 转，所以非 1 的占比就是"额外转正开销"的权重

用法: python -E devtools/probe_jpeg_census.py [图库根] [头部读取字节数]
"""
import json
import os
import struct
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

GALLERY_INDEX = os.path.join(GALLERY_ROOT, ".gallery_index")
ROOT = sys.argv[1] if len(sys.argv) > 1 else GALLERY_ROOT
HEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 256 << 10

# 与 io_utils._reduced_flag 完全一致的阈值（改那里必须同步改这里）
_TARGET = 2048
_BUCKETS = (("全解<=2560", _TARGET * 1.25),
            ("1/2<=5120", _TARGET * 2.5),
            ("1/4<=10240", _TARGET * 5))

# SOF 标记 → (名称, 是否渐进)
_SOF = {0xC0: ("baseline", False), 0xC1: ("extended-seq", False),
        0xC2: ("progressive", True), 0xC3: ("lossless", False),
        0xC5: ("diff-seq", False), 0xC6: ("diff-prog", True),
        0xC7: ("diff-lossless", False), 0xC9: ("arith-seq", False),
        0xCA: ("arith-prog", True), 0xCB: ("arith-lossless", False),
        0xCD: ("diff-arith-seq", False), 0xCE: ("diff-arith-prog", True),
        0xCF: ("diff-arith-lossless", False)}


def exif_orient(payload: bytes):
    """从 APP1 载荷（'Exif\\0\\0' 之后）读 TIFF 的 Orientation(0x0112)。"""
    if len(payload) < 14:
        return None
    bo = payload[0:2]
    if bo == b"II":
        u16, u32 = "<H", "<I"
    elif bo == b"MM":
        u16, u32 = ">H", ">I"
    else:
        return None
    try:
        ifd = struct.unpack(u32, payload[4:8])[0]
        if ifd + 2 > len(payload):
            return None
        n = struct.unpack(u16, payload[ifd:ifd + 2])[0]
        for k in range(n):
            e = ifd + 2 + 12 * k
            if e + 12 > len(payload):
                return None
            tag, typ, cnt = struct.unpack(u16 + u16 + u32, payload[e:e + 8])
            if tag == 0x0112 and cnt == 1:
                if typ == 3:
                    return struct.unpack(u16, payload[e + 8:e + 10])[0]
                if typ == 4:
                    return struct.unpack(u32, payload[e + 8:e + 12])[0]
    except Exception:                                   # noqa: BLE001
        return None
    return None


def parse_jpeg(buf: bytes):
    """扫描 JPEG 标记段，返回 (w, h, prec, comp, sof_name, progressive, orient)。"""
    if buf[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(buf)
    out = {"w": 0, "h": 0, "prec": 8, "comp": 0, "sof": "?", "prog": False,
           "orient": 1, "exif": False, "subs": ""}
    while i + 1 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        m = buf[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0x01 or 0xD0 <= m <= 0xD8:
            i += 2
            continue
        if m == 0xDA:                       # SOS：之后是熵编码数据，停止
            break
        if i + 4 > n:
            break
        ln = struct.unpack(">H", buf[i + 2:i + 4])[0]
        seg = buf[i + 4:i + 2 + ln]
        if m in _SOF and len(seg) >= 6:
            name, prog = _SOF[m]
            out["prec"], out["sof"], out["prog"] = seg[0], name, prog
            out["h"], out["w"] = struct.unpack(">HH", seg[1:5])
            out["comp"] = seg[5]
            if len(seg) >= 6 + 3 * out["comp"]:
                # 分量 H/V 采样因子字节按**十六进制**读（0x11=4:4:4、0x22=4:2:0、
                # 0x21=4:2:2、0x12=4:4:0），用十进制打印会看成非法值
                out["subs"] = "/".join("%02x" % seg[7 + 3 * k]
                                       for k in range(out["comp"]))
                out["subs_name"] = {0x11: "4:4:4", 0x22: "4:2:0", 0x21: "4:2:2",
                                    0x12: "4:4:0"}.get(seg[7] if out["comp"] > 1 else 0x11,
                                                       "其它(%02x)" % seg[7])
        elif m == 0xE1 and seg[:6] == b"Exif\x00\x00":
            out["exif"] = True
            o = exif_orient(seg[6:])
            if o:
                out["orient"] = o
        i += 2 + ln
    return out if out["w"] and out["h"] else None


def reduced_bucket(w: int, h: int) -> str:
    """与 io_utils._reduced_flag 同口径：返回 cv2 实际会用的缩放档。"""
    ms = max(w, h)
    for name, lim in _BUCKETS:
        if ms <= lim:
            return name
    return "1/8>10240"


def main() -> int:
    import numpy as np

    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                     allow_pickle=True)]
    jpgs = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    print("图库 %s：路径索引 %d 条，其中 JPEG %d 张" % (ROOT, len(paths), len(jpgs)))

    t0 = time.perf_counter()
    bad, mp_by_bucket, bytes_by_bucket = [], Counter(), Counter()
    cnt = Counter()
    cnt_bucket = Counter()
    orient_c = Counter()
    subs_c = Counter()
    total_bytes = total_mp = 0
    no_exif = 0
    for p in jpgs:
        try:
            with open(p, "rb") as f:
                head = f.read(HEAD)
            sz = os.path.getsize(p)
        except OSError:
            bad.append(p)
            continue
        info = parse_jpeg(head)
        if info is None:
            bad.append(p)
            cnt["解析失败"] += 1
            continue
        total_bytes += sz
        mp = info["w"] * info["h"] / 1e6
        total_mp += mp
        cnt["%s %dbit c%d" % (info["sof"], info["prec"], info["comp"])] += 1
        cnt["渐进" if info["prog"] else "基线/序列"] += 1
        b = reduced_bucket(info["w"], info["h"])
        cnt_bucket[b] += 1
        mp_by_bucket[b] += mp
        bytes_by_bucket[b] += sz
        orient_c[info["orient"]] += 1
        subs_c[info["subs"]] += 1
        if not info["exif"]:
            no_exif += 1

    el = time.perf_counter() - t0
    print("\n== 表头扫描 %d 张（%.1f s，只读前 %d KB/张）—— 解析失败 %d"
          % (len(jpgs) - len(bad), el, HEAD >> 10, len(bad)))
    print("   合计 %.1f GB / %.0f MP" % (total_bytes / 2 ** 30, total_mp))
    print("\n== 编码类别")
    for k, v in cnt.most_common():
        print("   %-22s %6d  %5.1f%%" % (k, v, 100.0 * v / max(len(jpgs) - len(bad), 1)))
    print("\n== 采样比（各分量 H/V 采样因子，十六进制；0x11=4:4:4 0x22=4:2:0）")
    for k, v in subs_c.most_common(8):
        nm = None
        s = k.split("/")
        if len(s) >= 3:
            nm = {0x11: "4:4:4", 0x22: "4:2:0", 0x21: "4:2:2", 0x12: "4:4:0"}.get(int(s[0], 16))
        elif len(s) == 1:
            nm = "灰度"
        print("   %-14s %-8s %6d  %5.1f%%"
              % (k or "(空)", nm or "?", v, 100.0 * v / max(len(jpgs) - len(bad), 1)))
    print("\n== EXIF 方向（1=无需转正；非 1 换库需自己 numpy 转）")
    for k in sorted(orient_c):
        print("   orient=%d  %6d  %5.1f%%" % (k, orient_c[k], 100.0 * orient_c[k] / max(len(jpgs) - len(bad), 1)))
    print("   无 EXIF 段：%d" % no_exif)
    print("\n== 缩放档（决定'换解码器能否保住 DCT 域缩放'）")
    print("   %-12s %7s %9s %9s %9s" % ("档", "张数", "占比", "MP占比", "字节占比"))
    for name, _ in _BUCKETS + (("1/8>10240", None),):
        if not cnt_bucket[name]:
            continue
        print("   %-12s %7d %8.1f%% %8.1f%% %8.1f%%"
              % (name, cnt_bucket[name], 100.0 * cnt_bucket[name] / max(len(jpgs) - len(bad), 1),
                 100.0 * mp_by_bucket[name] / max(total_mp, 1e-9),
                 100.0 * bytes_by_bucket[name] / max(total_bytes, 1)))

    out = {"root": ROOT, "n_jpeg": len(jpgs), "n_parsed": len(jpgs) - len(bad),
           "total_gb": round(total_bytes / 2 ** 30, 2), "total_mp": round(total_mp, 1),
           "class": dict(cnt), "progressive": cnt["渐进"],
           "orient": {str(k): v for k, v in orient_c.items()}, "no_exif": no_exif,
           "subsampling": dict(subs_c), "bucket_n": dict(cnt_bucket),
           "bucket_mp": {k: round(v, 1) for k, v in mp_by_bucket.items()},
           "bucket_gb": {k: round(v / 2 ** 30, 2) for k, v in bytes_by_bucket.items()},
           "scan_s": round(el, 2), "ts": time.strftime("%Y%m%d-%H%M%S")}
    os.makedirs(os.path.join(_HERE, "perf_reports"), exist_ok=True)
    jf = os.path.join(_HERE, "perf_reports", "jpeg_census_%s.json" % out["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nJSON:", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
