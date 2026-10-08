# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — C4 前置：JPEG 解码耗时随 DCT 缩放档的曲线
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""C4 的前置测量：**只给数，不改任何代码**。

现状（io_utils._reduced_flag，_DECODE_TARGET_SIDE = 2048）：

    最长边 <= 2560 → 全解 ；<= 5120 → 1/2 ；<= 10240 → 1/4 ；更大 → 1/8

候选（C4）：把门槛收到 **>1600 就用 1/2**。也就是把"最长边落在 (1600, 2560] 的图"
从全尺寸解码改成 1/2 解码 —— 解码像素量降到约 1/4。

本脚本回答两个问题（都不动代码）：
  1. **值多少**：那批图占全库多少张/多少像素；同批图 1/1 与 1/2 的实际解码耗时差多少
     → 折算成全库能省多少解码 CPU。
  2. **代价是什么**：工作图最长边从最多 2560 掉到约 1280（少 4 倍像素），
     于是粗筛指纹（64×64）、Hu、ResNet 的 224 中心窗**全部换了输入** →
     必须重建索引并重跑召回评估；此外库内命中框是"解码域坐标"，
     几何口径也会跟着变（这是已知问题，不是本脚本引入的）。

只读：只读图片，不写索引、不改配置。

用法: python -E devtools/probe_jpeg_scale_curve.py [每档张数=12] [轮数=2]
"""
import json
import os
import random
import statistics as st
import struct
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT                                  # noqa: E402

OUT_DIR = os.path.join(_HERE, "perf_reports")
N_EACH = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# 现状门槛与候选门槛（与 io_utils._reduced_flag 同口径）
CUR = [("全解<=2560", 2560, cv2.IMREAD_COLOR_RGB),
       ("1/2<=5120", 5120, cv2.IMREAD_REDUCED_COLOR_2),
       ("1/4<=10240", 10240, cv2.IMREAD_REDUCED_COLOR_4),
       ("1/8>10240", None, cv2.IMREAD_REDUCED_COLOR_8)]
BAND = (1600, 2560)          # C4 会改动的尺寸带（左开右闭）


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


def band_of(ms):
    for name, lim, _f in CUR:
        if lim is None or ms <= lim:
            return name
    return CUR[-1][0]


def main() -> int:
    idx = os.path.join(GALLERY_ROOT, ".gallery_index")
    paths = [str(x) for x in np.load(os.path.join(idx, "gallery.paths.npy"), allow_pickle=True)]
    jpgs = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    random.seed(11)
    random.shuffle(jpgs)

    # ---- 1) 头部扫描：现状分档 + C4 会改动的尺寸带
    print("扫描 JPEG 头部（只读前 32KB）…", flush=True)
    t0 = time.perf_counter()
    by_band, in_c4, in_c4_mp, tot_mp = [], 0, 0.0, 0.0
    band_lists = {name: [] for name, _l, _f in CUR}
    c4_list = []
    for p in jpgs:
        try:
            with open(p, "rb") as f:
                d = jpeg_dims(f.read(32 << 10))
        except OSError:
            continue
        if not d:
            continue
        w, h = d
        ms = max(w, h)
        mp = w * h / 1e6
        tot_mp += mp
        band_lists[band_of(ms)].append((p, ms, mp))
        if BAND[0] < ms <= BAND[1]:
            in_c4 += 1
            in_c4_mp += mp
            c4_list.append((p, ms, mp))
    print("  %.1f s；JPEG %d 张 / %.0f MP" % (time.perf_counter() - t0, len(jpgs), tot_mp))
    print("\n=== 现状分档（io_utils._reduced_flag）===")
    for name, _l, _f in CUR:
        v = band_lists[name]
        if not v:
            continue
        print("  %-12s %6d 张  %7.1f MP  %5.1f%% 张数" % (name, len(v), sum(x[2] for x in v),
                                                        100.0 * len(v) / len(jpgs)))
    print("\n=== C4 会改动的尺寸带：最长边 ∈ (%d, %d] ===" % BAND)
    print("  %d 张（占 JPEG %.1f%%）、%.1f MP（占 %.1f%%）→ 现状是全解，候选改成 1/2"
          % (in_c4, 100.0 * in_c4 / len(jpgs), in_c4_mp, 100.0 * in_c4_mp / max(tot_mp, 1e-9)))

    # ---- 2) 配对计时：同一张图跑 4 个缩放档
    groups = [("C4 目标带(%d,%d]" % BAND, c4_list)]
    for name in ("1/2<=5120", "1/4<=10240"):
        groups.append((name + " 取样", band_lists[name]))
    decs = [("全解 1/1", cv2.IMREAD_COLOR_RGB), ("1/2", cv2.IMREAD_REDUCED_COLOR_2),
            ("1/4", cv2.IMREAD_REDUCED_COLOR_4), ("1/8", cv2.IMREAD_REDUCED_COLOR_8)]
    rows, per_img = [], {}
    for gname, pool in groups:
        if not pool:
            continue
        take = pool if len(pool) <= N_EACH else random.sample(pool, N_EACH)
        data = []
        for p, ms, mp in take:
            try:
                data.append((p, open(p, "rb").read(), ms, mp))
            except OSError:
                pass
        print("\n=== %s：%d 张，%d 轮配对（同一张图顺序轮换跑 4 档）===" % (gname, len(data), ROUNDS))
        acc = {n: 0.0 for n, _ in decs}
        cnt = {n: 0 for n, _ in decs}
        pxs = {n: 0 for n, _ in decs}
        for r in range(ROUNDS):
            for k, (p, b, ms, mp) in enumerate(data):
                order = decs if (k + r) % 2 == 0 else list(reversed(decs))
                for name, flag in order:
                    t = time.perf_counter()
                    a = cv2.imdecode(np.frombuffer(b, np.uint8), flag)
                    dt = time.perf_counter() - t
                    if a is None:
                        continue
                    acc[name] += dt
                    cnt[name] += 1
                    pxs[name] += a.shape[0] * a.shape[1]
        base = acc["全解 1/1"] / max(cnt["全解 1/1"], 1)
        row = {"group": gname, "n": len(data)}
        for name, _f in decs:
            if not cnt[name]:
                continue
            ms_avg = acc[name] / cnt[name] * 1e3
            mp_avg = pxs[name] / cnt[name] / 1e6
            print("   %-8s %8.2f ms/张  %7.2f MP/张  %7.1f MP/s  相对全解 %.2fx"
                  % (name, ms_avg, mp_avg, mp_avg / max(ms_avg, 1e-9) * 1e3,
                     base / max(acc[name] / cnt[name], 1e-12)))
            row[name + "_ms"] = round(ms_avg, 2)
            row[name + "_mp"] = round(mp_avg, 2)
        rows.append(row)
        if "C4" in gname:
            per_img["full_ms"] = row.get("全解 1/1_ms")
            per_img["half_ms"] = row.get("1/2_ms")

    # ---- 3) 折算全库收益
    print("\n=== 折算（只算 C4 目标带那 %d 张）===" % in_c4)
    saving_cpu = None
    if per_img.get("full_ms") and per_img.get("half_ms"):
        d_ms = per_img["full_ms"] - per_img["half_ms"]
        saving_cpu = d_ms * in_c4 / 1000.0
        print("   该带单张：全解 %.2f ms → 1/2 %.2f ms（省 %.2f ms/张）"
              % (per_img["full_ms"], per_img["half_ms"], d_ms))
        print("   全库口径：%d 张 × %.2f ms ≈ **省 %.0f 核秒（%.1f 分钟）解码 CPU**"
              % (in_c4, d_ms, saving_cpu, saving_cpu / 60.0))
        print("   参考：C2 实测 JPEG 解码占建库 CPU 约 1/4；上值只占 JPEG 解码的一部分")
    print("   **代价（必须同时接受）**：")
    print("     - 工作图最长边最多 2560 → 约 1280（像素少 4 倍）")
    print("     - 粗筛指纹 64×64 / Hu / ResNet 224 中心窗**全部换输入** → 必须重建索引")
    print("       并重跑召回评估（37,683 张整图 + 44 万块瓦片）")
    print("     - 库内命中框存的是「解码域坐标」，几何口径随之改变（已知问题，见 OLD_text 记录）")

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"ts": time.strftime("%Y%m%d-%H%M%S"), "n_each": N_EACH, "rounds": ROUNDS,
            "jpeg_total": len(jpgs), "jpeg_mp": round(tot_mp, 1),
            "band": list(BAND), "band_n": in_c4, "band_mp": round(in_c4_mp, 1),
            "band_pct_n": round(100.0 * in_c4 / len(jpgs), 2),
            "band_pct_mp": round(100.0 * in_c4_mp / max(tot_mp, 1e-9), 2),
            "rows": rows, "saving_cpu_s": round(saving_cpu, 1) if saving_cpu else None,
            "bands_cur": {n: len(band_lists[n]) for n, _l, _f in CUR}}
    jf = os.path.join(OUT_DIR, "jpeg_scale_curve_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nJSON: %s" % jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
