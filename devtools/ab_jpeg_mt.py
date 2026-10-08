# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — JPEG 解码器**多线程**对照（隔离建库流水线）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""为什么单张快 1.14x 的解码器，进了 18 路建库反而更慢？—— 把流水线剥掉只测解码。

建库 A/B（devtools/ab_jpeg_turbo.py）出现了反直觉结果：
    JPEG 解码 21.5 → 17.8 核秒（**快了 1.21x**），但同一进程里 **PNG 解码的累计耗时
    从 64.5 涨到 99.6 核秒（+54%）**，总 CPU +20%、墙钟 +24%。
PNG 那条路一行没改，所以怀疑是"同进程并发下的相互影响"（内存/分配器/GIL），
而不是解码器本身。本脚本把建库流水线整个剥掉，只保留"18 路线程池 + decode_rgb"，
分三段定位：

  (1) 纯 JPEG 负载：两个解码器谁快（去掉 PNG 干扰）
  (2) 混合负载：JPEG 换库后，**PNG 的累计耗时是否被拖慢**（复现建库里的现象）
  (3) 缓解尝试：给 TurboJPEG 每线程缓存输出缓冲（`dst=`，省掉每次 np.empty）

每段都交替顺序、多轮取中位，并同时记录 墙钟 / 进程 CPU / 各格式累计耗时。

用法: python -E devtools\ab_jpeg_mt.py [每段张数=240] [线程=18] [轮=3]
"""
import io
import json
import os
import random
import statistics as st
import struct
import sys
import threading
import time
import warnings
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

GALLERY_INDEX = os.path.join(GALLERY_ROOT, ".gallery_index")
OUT_DIR = os.path.join(_HERE, "perf_reports")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 18
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
DLL = os.environ.get("TURBOJPEG_DLL") or os.path.join(
    os.environ.get("TEMP", ""), "c2_jpeg", "turbojpeg.dll")
_TARGET = 2048


def scale_of(w, h):
    ms = max(w, h)
    if ms <= _TARGET * 1.25:
        return (1, 1)
    if ms <= _TARGET * 2.5:
        return (1, 2)
    if ms <= _TARGET * 5:
        return (1, 4)
    return (1, 8)


def jpeg_dims(head: bytes):
    if head[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(head)
    while i + 4 <= n:
        if head[i] != 0xFF:
            i += 1
            continue
        m = head[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0x01 or 0xD0 <= m <= 0xD8:
            i += 2
            continue
        if m == 0xDA:
            break
        ln = struct.unpack(">H", head[i + 2:i + 4])[0]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            seg = head[i + 4:i + 2 + ln]
            if len(seg) >= 5:
                h, w = struct.unpack(">HH", seg[1:5])
                return w, h
            return None
        i += 2 + ln
    return None


# ------------------------------------------------------------------ 解码器
_TL = threading.local()


def _tl():
    if getattr(_TL, "d", None) is None:
        _TL.d = {}
    return _TL.d


def arm_cv2(data, probe):
    from hybrid_search import io_utils as iu
    return iu.decode_rgb(data)


def _tj_ready(probe, data):
    """与真实候选同款准入：JPEG + EXIF 方向 1 + 尾部 EOI；否则回退 cv2。"""
    return (probe is not None and probe[0] == "JPEG" and probe[2] == 1
            and data[-2:] == b"\xff\xd9")


def arm_tj(data, probe):
    from hybrid_search import io_utils as iu
    from turbojpeg import TJPF_RGB, TurboJPEG
    if not _tj_ready(probe, data):
        return iu.decode_rgb(data)                        # PNG / 异常 JPEG 走现状
    d = _tl()
    if "tj" not in d:
        d["tj"] = TurboJPEG(DLL)
        d["rgb"] = TJPF_RGB
    return d["tj"].decode(data, pixel_format=d["rgb"],
                          scaling_factor=scale_of(*probe[1]))


def arm_tj_dst(data, probe):
    """缓解尝试：每线程按输出形状缓存 dst 缓冲，省掉每次 np.empty 的新页。"""
    from hybrid_search import io_utils as iu
    from turbojpeg import TJPF_RGB, TurboJPEG
    if not _tj_ready(probe, data):
        return iu.decode_rgb(data)
    d = _tl()
    if "tj" not in d:
        d["tj"] = TurboJPEG(DLL)
        d["rgb"] = TJPF_RGB
        d["dst"] = {}
    w, h = probe[1]
    sf = scale_of(w, h)
    ow, oh = -(-w // sf[1]), -(-h // sf[1])
    key = (oh, ow)
    buf = d["dst"].get(key)
    if buf is None:
        if len(d["dst"]) > 6:
            d["dst"].clear()
        buf = np.empty((oh, ow, 3), dtype=np.uint8)
        d["dst"][key] = buf
    return d["tj"].decode(data, pixel_format=d["rgb"], scaling_factor=sf, dst=buf)


ARMS = [("cv2", arm_cv2), ("tj", arm_tj), ("tj_dst", arm_tj_dst)]


def run_arm(items, fn, name, threads=THREADS):
    """18 路线程池跑一遍；返回 (wall, cpu, 每张耗时ms, 按格式累计秒)。"""
    lock = threading.Lock()
    idx = [0]
    per = []
    byfmt = defaultdict(float)
    byfmt_n = defaultdict(int)

    def work():
        from hybrid_search import io_utils as iu
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(items):
                return
            it = items[i]
            t0 = time.perf_counter()
            try:
                fn(it["b"], it["probe"])
            except Exception:                             # noqa: BLE001
                pass
            dt = time.perf_counter() - t0
            with lock:
                per.append(dt * 1e3)
                byfmt[it["fmt"]] += dt
                byfmt_n[it["fmt"]] += 1
        del iu

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    ths = [threading.Thread(target=work, daemon=True) for _ in range(threads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    return {"arm": name, "wall_s": round(wall, 3), "cpu_s": round(cpu, 2),
            "cores": round(cpu / max(wall, 1e-9), 2),
            "imgs_per_s": round(len(items) / max(wall, 1e-9), 1),
            "p50_ms": round(st.median(per), 2),
            "fmt_s": {k: round(v, 2) for k, v in byfmt.items()},
            "fmt_n": dict(byfmt_n)}


def main() -> int:
    random.seed(11)
    from hybrid_search import io_utils as iu
    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                     allow_pickle=True)]
    jpgs = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    pngs = [p for p in paths if p.lower().endswith(".png")]
    random.shuffle(jpgs)
    random.shuffle(pngs)

    def load(plist, n):
        out = []
        for p in plist:
            if len(out) >= n:
                break
            try:
                b = open(p, "rb").read()
            except OSError:
                continue
            probe = iu._probe(b)
            if probe is None:
                continue
            if (probe[0] == "JPEG" and (probe[2] != 1 or b[-2:] != b"\xff\xd9")):
                continue
            out.append({"b": b, "probe": probe,
                        "fmt": "PNG" if probe[0] == "PNG" else "JPEG"})
        return out

    j_only = load(jpgs, N)
    mixed = load(jpgs, N * 3 // 4) + load(pngs, N // 4)
    random.shuffle(mixed)
    print("样本：纯 JPEG %d 张 / 混合 %d 张（JPEG %d + PNG %d）；%d 路 × %d 轮"
          % (len(j_only), len(mixed), sum(1 for x in mixed if x["fmt"] == "JPEG"),
             sum(1 for x in mixed if x["fmt"] == "PNG"), THREADS, ROUNDS))

    results = {}
    for tag, items in (("纯JPEG", j_only), ("混合", mixed)):
        print("\n===== 段：%s（%d 张 × %d 路，%d 轮交替）=====" % (tag, len(items), THREADS, ROUNDS))
        acc = {n: [] for n, _ in ARMS}
        for r in range(ROUNDS):
            order = ARMS if r % 2 == 0 else list(reversed(ARMS))
            for name, fn in order:
                res = run_arm(items, fn, name)
                acc[name].append(res)
                print("   轮%d %-7s wall %6.2fs CPU %7.1f 核秒 %5.2f核 %6.1f 张/s p50 %6.1f ms | %s"
                      % (r + 1, name, res["wall_s"], res["cpu_s"], res["cores"],
                         res["imgs_per_s"], res["p50_ms"],
                         " ".join("%s %.1fs/%d" % (k, v, res["fmt_n"][k])
                                  for k, v in sorted(res["fmt_s"].items()))))
        base = acc["cv2"]
        print("   --- 相对 cv2（各轮中位）---")
        for name, _ in ARMS:
            if name == "cv2":
                continue
            dw = st.median([100 * (b["wall_s"] - a["wall_s"]) / a["wall_s"]
                            for a, b in zip(base, acc[name])])
            dc = st.median([100 * (b["cpu_s"] - a["cpu_s"]) / a["cpu_s"]
                            for a, b in zip(base, acc[name])])
            line = "   %-7s Δwall %+6.2f%%  ΔCPU %+6.2f%%" % (name, dw, dc)
            for fmt in ("JPEG", "PNG"):
                xs = [(b["fmt_s"].get(fmt, 0), a["fmt_s"].get(fmt, 0)) for a, b in zip(base, acc[name])]
                if all(x[1] for x in xs):
                    d = st.median([100 * (y - x) / x for y, x in xs])
                    line += "  | %s 累计耗时 %+6.2f%%" % (fmt, d)
            print(line)
        results[tag] = {n: acc[n] for n, _ in ARMS}

    os.makedirs(OUT_DIR, exist_ok=True)
    jf = os.path.join(OUT_DIR, "ab_jpeg_mt_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(jf, "w", encoding="utf-8") as f:
        json.dump({"n": N, "threads": THREADS, "rounds": ROUNDS, "dll": DLL,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\nJSON:", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
