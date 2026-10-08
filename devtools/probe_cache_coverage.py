# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 预处理缓存覆盖率探针（只读）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""prep 缓存覆盖率：**先确认"磁盘上的缓存对我们当前配置还算不算数"，再谈命中率**。

缓存键 = `sha1(规范化绝对路径 | st_size | st_mtime_ns)`，落在
`<索引目录>/prep_cache/<键前两位>/<键>.bin`。除此之外 `head.sig` 还编码了预处理契约
（模型/resize/crop/pre_side/normgpu-normcpu/codec），**sig 不符即视为未命中**，
所以"缓存文件存在"≠"能用" —— 两者必须分开统计：

  A. 契约一致性：抽样读 `head.sig`，与当前配置算出的 sig 比对（不符 → 全库 0 覆盖）
  B. 键命中率：逐张索引图片算键、看磁盘上有没有对应文件（这是真正的覆盖率）
  C. 未命中归因：文件已消失 / 文件在但键变了（size 或 mtime 变过 = "改文件即失效"）
  D. 孤儿条目：缓存里有、索引里没有的键（旧版本残留，白占磁盘）

只读：不写缓存、不改索引、不删文件。

用法: python -E devtools/probe_cache_coverage.py [索引目录] [--sample 400]
"""
import collections
import json
import os
import struct
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np                                             # noqa: E402

from paths import GALLERY_ROOT                                  # noqa: E402

OUT_DIR = os.path.join(_HERE, "perf_reports")
SAVED_MS_PER_HIT = 80.0        # 与 prep_cache.get() 里 saved_ms 的经验值一致


def read_head(path):
    """读缓存条目的 JSON 头（失败返回 None）。"""
    from hybrid_search import prep_cache as pc
    try:
        with open(path, "rb") as f:
            b = f.read(9)
            if b[:5] != pc.MAGIC:
                return None
            (hlen,) = struct.unpack("<I", b[5:9])
            return json.loads(f.read(hlen).decode("utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sample = 400
    for a in sys.argv[1:]:
        if a.startswith("--sample"):
            sample = int(a.split("=", 1)[1]) if "=" in a else sample
    from hybrid_search import prep_cache as pc
    from hybrid_search.config import Config

    idx = args[0] if args else os.path.join(GALLERY_ROOT, ".gallery_index")
    cdir = os.path.join(idx, "prep_cache")
    cfg = Config()
    codec_now = pc._CODEC if pc._CODEC_OK[0] else "png1"
    sig_now = pc._sig_of(cfg.model, 2048, cfg.norm_on_gpu, codec_now)
    print("索引目录 : %s" % idx)
    print("缓存目录 : %s（存在=%s）" % (cdir, os.path.isdir(cdir)))
    print("当前契约 : sig=%s（model=%s pre_side=2048 norm_on_gpu=%s codec=%s）"
          % (sig_now, cfg.model, cfg.norm_on_gpu, codec_now))
    if not os.path.isdir(cdir):
        print("**没有磁盘缓存** → 覆盖率 0，首次建库必然全 miss（符合预期）")
        return 0

    # ---- 全量枚举缓存条目
    t0 = time.perf_counter()
    entries = []
    total_bytes = 0
    for dp, _d, fs in os.walk(cdir):
        for f in fs:
            p = os.path.join(dp, f)
            entries.append(p)
            try:
                total_bytes += os.path.getsize(p)
            except OSError:
                pass
    print("\n磁盘缓存条目 %d 个 / %.2f GB（枚举 %.1f s）"
          % (len(entries), total_bytes / 2 ** 30, time.perf_counter() - t0))

    # ---- A. 契约一致性（抽样读头）
    step = max(1, len(entries) // max(sample, 1))
    sigs, codecs, shapes, bad = collections.Counter(), collections.Counter(), collections.Counter(), 0
    for p in entries[::step][:sample]:
        h = read_head(p)
        if h is None:
            bad += 1
            continue
        sigs[h.get("sig")] += 1
        codecs[h.get("codec", "png")] += 1
        shapes[tuple(h.get("shape", []))] += 1
    n_s = sum(sigs.values())
    print("\n=== A. 契约一致性（抽样 %d 条）===" % n_s)
    for s, n in sigs.most_common():
        print("   sig=%-14s %5d 条  %s" % (s, n, "<== 与当前配置一致" if s == sig_now else ""))
    print("   codec 分布: %s" % dict(codecs))
    print("   shape 分布: %s" % dict(shapes))
    print("   头解析失败: %d" % bad)
    usable = sigs.get(sig_now, 0) / max(n_s, 1)
    print("   → 抽样中契约一致占比 **%.1f%%**" % (100 * usable))

    # ---- B/C. 键命中率（逐张索引图片）
    pfile = os.path.join(idx, "gallery.paths.npy")
    paths = [str(x) for x in np.load(pfile, allow_pickle=True)]
    print("\n=== B. 键命中率（索引内 %d 张）===" % len(paths))
    t0 = time.perf_counter()
    hit = miss_key = miss_gone = 0
    keyset = set()
    by_ext = collections.defaultdict(lambda: [0, 0])
    for i, p in enumerate(paths):
        k = pc.PrepCache.key_for(p)
        if k is None:
            miss_gone += 1
            continue
        ok = os.path.exists(os.path.join(cdir, k[:2], k + ".bin"))
        hit += ok
        miss_key += (not ok)
        if ok:
            keyset.add(k)
        e = os.path.splitext(p)[1].lower()
        by_ext[e][0 if ok else 1] += 1
        if (i + 1) % 10000 == 0:
            print("   ... %d/%d（%.0f s）命中 %d" % (i + 1, len(paths), time.perf_counter() - t0, hit),
                  flush=True)
    n = max(len(paths), 1)
    print("   命中（键存在）    : %6d  **%.1f%%**" % (hit, 100.0 * hit / n))
    print("   未命中-键不存在   : %6d  %.1f%%（文件在，但 size 或 mtime 变过）"
          % (miss_key, 100.0 * miss_key / n))
    print("   未命中-文件已消失 : %6d  %.1f%%" % (miss_gone, 100.0 * miss_gone / n))
    print("   按扩展名（命中 / 未命中）:")
    for e, (h_, m_) in sorted(by_ext.items(), key=lambda x: -x[1][0]):
        print("     %-8s %6d / %6d" % (e or "(无)", h_, m_))

    # ---- D. 孤儿条目
    orphan = len(entries) - len(keyset)
    print("\n=== D. 孤儿条目（缓存有、索引无）===")
    print("   %d 条（%.1f%%；旧版本/被替换文件的残留，白占 %.2f GB 量级）"
          % (orphan, 100.0 * orphan / max(len(entries), 1),
             total_bytes / 2 ** 30 * orphan / max(len(entries), 1)))

    saved_s = hit * SAVED_MS_PER_HIT / 1000.0
    print("\n=== 结论 ===")
    if usable < 0.99:
        print("   ⚠ 抽样中只有 %.1f%% 的条目契约与当前配置一致 → 实际可用命中率上限 "
              "≈ %d 张（%.1f%%）" % (100 * usable, int(hit * usable), 100.0 * hit * usable / n))
    else:
        print("   ✓ 抽样中契约全部一致 → 键命中即可用；预计下次建库可少花 **%.0f s CPU**"
              "（%.1f ms/张 × %d 张）" % (saved_s, SAVED_MS_PER_HIT, hit))

    meta = {"ts": time.strftime("%Y%m%d-%H%M%S"), "index": idx, "cache_dir": cdir,
            "entries": len(entries), "bytes": total_bytes, "sig_now": sig_now,
            "sig_dist": dict(sigs), "codec_dist": dict(codecs),
            "contract_match_ratio": round(usable, 4),
            "n_images": len(paths), "hit": hit, "miss_key": miss_key,
            "miss_gone": miss_gone, "orphan": orphan,
            "hit_ratio": round(hit / n, 4),
            "saved_ms_per_hit": SAVED_MS_PER_HIT, "saved_cpu_s": round(saved_s, 1),
            "by_ext": {k: v for k, v in by_ext.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    jf = os.path.join(OUT_DIR, "cache_coverage_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("   JSON: %s" % jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
