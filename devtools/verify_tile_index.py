# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 瓦片索引体检：结构一致性 + 内容复核 + 去重账 + 命中框坐标系判定
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""瓦片(局部)索引核查：结构一致性 + 内容可复核 + 命中框坐标系体检。

只读索引、只读图片，不写任何索引与图库文件；建议每次瓦片建库后跑一次。用法：

    python devtools/verify_tile_index.py "F:\\视频\\.gallery_index\\gallery_tiles" --root F:\\视频
    python devtools/verify_tile_index.py <瓦片前缀> --root <图库目录> --samples 300

检查项：
  1) 结构：各 .npy 行数一致 / 唯一图片数 / 块-图比 / 特征范数 / 框合法性 / md5 唯一
  2) 与整图索引：粗筛参数是否一致（决定 hybrid 能否用）＋ 路径集合差异
  3) 内容复核：抽若干块重算 md5(文件字节+框)，与库内值比对（索引是否与当前文件一致）
  4) 去重账：磁盘上"有图但库里没块"的文件，抽样确认是否都是已入库内容的重复副本
  5) 命中框坐标系：抽图按文件名义尺寸重算几何，判定库内框是"原图像素"还是
     "解码域缩放后的坐标"（后者会让 GUI 红框在大图上按 1/2、1/4 偏位）
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search.io_utils import read_bytes, decode_rgb  # noqa: E402
from hybrid_search.tile_index import (tile_grid, TILE_DEFAULT,  # noqa: E402
                                      OVERLAP_DEFAULT, MIN_SIDE_DEFAULT,
                                      PRE_MAX_SIDE)

OK = "  OK "
BAD = "  !! "
KEYS = ("paths", "fine_paths", "fp", "hu", "hu_stats", "md5s", "boxes", "fine")


def load(prefix, name, mmap=False):
    p = "%s.%s.npy" % (prefix, name)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True, mmap_mode="r" if mmap else None)


def norm(p):
    return os.path.normcase(os.path.normpath(str(p)))


def main() -> int:
    ap = argparse.ArgumentParser(description="瓦片索引核查（只读）")
    ap.add_argument("prefix", help="瓦片索引前缀（不带 .paths.npy 后缀）")
    ap.add_argument("--root", default=None, help="图库根目录（做去重账抽样时需要）")
    ap.add_argument("--gallery", default=None,
                    help="整图索引前缀（默认＝瓦片同目录下的 gallery）")
    ap.add_argument("--samples", type=int, default=200, help="几何/内容抽样数")
    a = ap.parse_args()

    prefix = a.prefix
    arrs = {k: load(prefix, k, mmap=(k == "fine")) for k in KEYS}
    if arrs["paths"] is None:
        print("找不到索引：%s.paths.npy" % prefix)
        return 2
    paths = [str(x) for x in arrs["paths"]]
    n = len(paths)
    md5s = [str(x) for x in arrs["md5s"]]
    idx_by_path = collections.defaultdict(list)
    for i, p in enumerate(paths):
        idx_by_path[norm(p)].append(i)
    uniq = sorted(idx_by_path)
    meta = json.load(open(prefix + ".meta.json", encoding="utf-8"))

    print("=" * 78)
    print("1) 结构一致性")
    for k in KEYS:
        print("   %-11s %s" % (k, "缺失" if arrs[k] is None else arrs[k].shape))
    sizes = {k: len(arrs[k]) for k in KEYS if arrs[k] is not None and k != "hu_stats"}
    same = len(set(sizes.values())) == 1
    print("%s行数一致（%d 行）%s" % (OK if same else BAD, n, "" if same else " → " + str(sizes)))
    if arrs["hu_stats"] is not None:
        print("   hu_stats（Hu 归一化基准，与行数无关）%s" % (arrs["hu_stats"].shape,))
    cnt = collections.Counter(paths)
    print("   唯一图片 %d 张 → 平均 %.2f 块/图（1..%d 块，单块图 %.1f%%）"
          % (len(uniq), n / max(len(uniq), 1), max(cnt.values()),
             100.0 * sum(1 for v in cnt.values() if v == 1) / max(len(cnt), 1)))
    fine = arrs["fine"]
    if fine is not None:
        sub = np.asarray(fine[::max(1, n // 500)], dtype=np.float32)
        nr = np.linalg.norm(sub, axis=1)
        print("%s精排特征 L2 范数 %.5f..%.5f（应≈1.0）"
              % (OK if abs(float(nr.mean()) - 1.0) < 1e-4 else BAD, nr.min(), nr.max()))
    box = arrs["boxes"]
    if box is not None:
        good = bool(np.all(box[:, 2] > box[:, 0]) and np.all(box[:, 3] > box[:, 1]))
        print("%s命中框全部 x1>x0 且 y1>y0" % (OK if good else BAD))
    nmd5 = len(set(md5s))
    print("%s块 md5 唯一（%d 个）" % (OK if nmd5 == n else BAD, nmd5))

    gal = a.gallery or os.path.join(os.path.dirname(os.path.abspath(prefix)), "gallery")
    gp = load(gal, "paths")
    print("\n2) 与整图索引对比（%s）" % gal)
    if gp is None:
        print("   未找到整图索引，跳过")
        wmd5 = None
    else:
        gm = json.load(open(gal + ".meta.json", encoding="utf-8"))
        same_cfg = gm.get("cfg") == meta.get("cfg")
        print("%s粗筛参数一致（hybrid 检索可用）" % (OK if same_cfg else BAD))
        gset = set(norm(x) for x in gp)
        tset = set(uniq)
        print("   整图 %d 张 / 瓦片 %d 张 / 交集 %d / 仅瓦片 %d / 仅整图 %d"
              % (len(gset), len(tset), len(gset & tset), len(tset - gset), len(gset - tset)))
        _gm = load(gal, "md5s")
        wmd5 = set(str(x) for x in _gm) if _gm is not None else None
    print("   meta.tiles = %s" % (meta.get("tiles"),))

    print("\n3) 内容复核：重算 md5(文件字节+框) 与库内比对")
    random.seed(20260912)
    ok = bad = miss = 0
    for key in random.sample(uniq, min(a.samples, len(uniq))):
        raw = paths[idx_by_path[key][0]]
        data = read_bytes(raw)
        if data is None:
            miss += 1
            continue
        hits = 0
        for i in idx_by_path[key][:3]:
            b = tuple(int(round(v)) for v in box[i])
            h = hashlib.md5(data + ("|%d,%d,%d,%d" % b).encode()).hexdigest()
            hits += int(h == md5s[i])
        ok, bad = (ok + 1, bad) if hits == len(idx_by_path[key][:3]) else (ok, bad + 1)
    print("%s一致 %d / 不一致 %d / 读失败 %d" % (OK if bad == 0 else BAD, ok, bad, miss))

    if a.root and os.path.isdir(a.root):
        print("\n4) 去重账：磁盘上有图、库里没块的文件")
        exts = set((meta.get("cfg") or {}).get("extensions")
                   or [".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"])
        files = []
        for root, dirs, fs in os.walk(a.root):
            dirs[:] = [d for d in dirs if not d.startswith(".gallery")]
            files += [os.path.join(root, f) for f in fs
                      if os.path.splitext(f)[1].lower() in exts]
        tset = set(uniq)
        excluded = [p for p in files if norm(p) not in tset]
        dup = fail = other = 0
        for p in (random.sample(excluded, min(40, len(excluded))) if excluded else []):
            data = read_bytes(p)
            if data is None:
                fail += 1
                continue
            m = hashlib.md5(data).hexdigest()
            if wmd5 is not None and m in wmd5:
                dup += 1
            elif decode_rgb(data) is None:
                fail += 1
            else:
                other += 1
        print("   磁盘 %d 张 / 未进瓦片索引 %d 张；抽检 40：内容重复 %d、解码失败 %d、其他 %d"
              % (len(files), len(excluded), dup, fail, other))
        print("   → " + ("重复内容被内容去重属正常（与旧库块数同量级）"
                         if other == 0 else "出现『其他』：既非重复也没入库，需要排查"))
    else:
        print("\n4) 未给 --root，跳过去重账")

    print("\n5) 命中框坐标系（GUI 红框按名义尺寸换算，故这里判定库内框空间）")
    t = meta.get("tiles") or {}
    tile = int(t.get("tile", TILE_DEFAULT))
    ov = float(t.get("overlap", OVERLAP_DEFAULT))
    mn = int(t.get("min_side", MIN_SIDE_DEFAULT))
    pm = int(t.get("pre_max", PRE_MAX_SIDE))

    def geometry(w, h):
        """按 tile_index.tiles_of_rgb 的几何（含 min_side/pre_max）重算框列表。"""
        if max(w, h) < mn:
            return [(0, 0, w, h)]
        work_w, work_h, scale = w, h, 1.0
        if max(w, h) > pm:
            scale = pm / float(max(w, h))
            work_w, work_h = int(w * scale), int(h * scale)
        inv = 1.0 / scale
        out = []
        for (x0, y0, x1, y1) in tile_grid(work_w, work_h, tile, ov):
            sx0, sy0 = max(0, int(x0 * inv)), max(0, int(y0 * inv))
            sx1, sy1 = min(w, int(round(x1 * inv))), min(h, int(round(y1 * inv)))
            if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                continue
            out.append((sx0, sy0, sx1, sy1))
        return out

    same_ok = nominal_ok = reduced_bad = unknown = 0
    worst = None
    from PIL import Image
    for key in random.sample(uniq, min(a.samples, len(uniq))):
        raw = paths[idx_by_path[key][0]]
        stored = sorted(tuple(int(round(v)) for v in box[i]) for i in idx_by_path[key])
        try:
            with Image.open(raw) as im:
                w0, h0 = im.size
        except Exception:                          # noqa: BLE001
            unknown += 1
            continue
        data = read_bytes(raw)
        rgb = decode_rgb(data) if data else None
        if rgb is None:
            unknown += 1
            continue
        dh, dw = rgb.shape[:2]
        if sorted(geometry(dw, dh)) == stored:
            if (dw, dh) == (w0, h0):
                nominal_ok += 1                        # 未域缩放 → 框=原图像素
            else:
                reduced_bad += 1
                worst = worst or (os.path.basename(raw), (w0, h0), (dw, dh))
        elif sorted(geometry(w0, h0)) == stored:
            same_ok += 1
        else:
            unknown += 1
    tot = nominal_ok + reduced_bad + same_ok + unknown
    print("   抽样 %d 张：框=原图像素（解码未缩放）%d 张；框=名义尺寸几何 %d 张；"
          "框=解码缩放后坐标 %d 张；无法判定 %d 张"
          % (tot, nominal_ok, same_ok, reduced_bad, unknown))
    if reduced_bad:
        print("%s结论：约 %.0f%% 的图命中框落在『解码缩放后』坐标系；GUI 用名义尺寸换算"
              % (BAD, 100.0 * reduced_bad / max(tot - unknown, 1)))
        print("      → 红框位置/大小会按 1/2 或 1/4 偏位（README「已知问题」已记录）")
        if worst:
            print("      样例：%s 名义 %s×%s，实际解出 %s×%s"
                  % (worst[0], worst[1][0], worst[1][1], worst[2][0], worst[2][1]))
    else:
        print("%s结论：命中框与名义尺寸一致" % OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
