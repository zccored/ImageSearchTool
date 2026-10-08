# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — P0/P1 优化步骤的可复现 A/B 基准台
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""固定样本建库基准（P0/P1 每一步优化都用它出性能统计 + 位级一致性）。

设计要点：
  * 样本固定且可复现：从真实索引抽 N 张 + 注入 D 张"内容重复但不在索引里"的文件
    （用 --dup 才能体现"重复文件预过滤"的效果）；
  * 索引写在独立工作目录，**绝不触碰图库的真实索引**；
  * 默认关闭预处理缓存（要测的是解码/特征流水线本身，不是缓存命中）；
  * 记录 wall / CPU 核 / GPU 均值峰值 / 吞吐 / RSS 峰值，并把索引数组的
    sha256 落盘，供 --compare 做"索引语义是否改变"的位级比对。

用法:
  python devtools/ab_build_bench.py --mode tiles --label base --n 540 --dup 60
  python devtools/ab_build_bench.py --mode whole --label base --n 540 --dup 60
  python devtools/ab_build_bench.py --compare base p0a        # 位级比对
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time

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
DEFAULT_WORK = os.environ.get("BENCH_WORK",
                              os.path.join(tempfile.gettempdir(), "bench_imgsearch"))
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def norm(p):
    return os.path.normcase(os.path.normpath(str(p)))


def build_sample(n, dup, seed):
    """样本 = 索引内文件 + **与其中若干张内容完全相同**的索引外文件（成对注入）。

    成对才有意义：只有"孪生兄弟已在本轮处理过"，重复文件预过滤/去重才会生效；
    重复文件统一排在样本末尾（保证孪生兄弟先被处理）。
    """
    idx_paths = [str(x) for x in np.load(
        os.path.join(GALLERY_INDEX, "gallery.paths.npy"), allow_pickle=True)]
    idx_md5 = [str(x) for x in np.load(
        os.path.join(GALLERY_INDEX, "gallery.md5s.npy"), allow_pickle=True)]
    md5_to_path = dict(zip(idx_md5, idx_paths))
    have = set(norm(p) for p in idx_paths)
    rng = random.Random(seed)

    twins_in, twins_out, scanned = [], [], 0
    if dup:
        for base, dirs, files in os.walk(GALLERY_ROOT):
            dirs[:] = [d for d in dirs if not d.startswith(".gallery")]
            for f in files:
                if os.path.splitext(f)[1].lower() not in EXTS:
                    continue
                p = os.path.join(base, f)
                if norm(p) in have:
                    continue
                try:
                    m = hashlib.md5(open(p, "rb").read()).hexdigest()
                except OSError:
                    continue
                scanned += 1
                if m in md5_to_path:
                    twins_out.append(p)
                    twins_in.append(md5_to_path[m])
                    if len(twins_out) >= dup:
                        break
            if len(twins_out) >= dup:
                break
    print("  成对注入: 找到 %d 对内容相同的文件（扫描 %d 张索引外文件）"
          % (len(twins_out), scanned))
    chosen = set(twins_in)
    rest = [p for p in idx_paths if p not in chosen]
    rng.shuffle(rest)
    head = twins_in + rest[:max(0, n - len(twins_in))]
    return head + twins_out


class GpuWatch(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.stop_flag = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:                     # noqa: BLE001 —— 无显卡/无 pynvml 则只测 CPU
            self.nv = None

    def run(self):
        if self.nv is None:
            return
        while not self.stop_flag.is_set():
            try:
                self.samples.append(
                    self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            except Exception:                 # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)
        if self.nv is not None:
            try:
                self.nv.nvmlShutdown()
            except Exception:                 # noqa: BLE001
                pass


def sha_arrays(prefix):
    """索引数组指纹：只看数据文件（跳过 meta.json 里的时间戳）。"""
    out = {}
    for base in (prefix,):
        d = os.path.dirname(os.path.abspath(base)) or "."
        stem = os.path.basename(base)
        for f in sorted(os.listdir(d)):
            if f.startswith(stem + ".") and f.endswith((".npy", ".npz")):
                p = os.path.join(d, f)
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                out[f] = {"sha256": h.hexdigest(),
                          "bytes": os.path.getsize(p)}
    return out


def run_one(mode, label, sample, work, prep_cache, dedup_prefilter, tile_flush_ms,
            md5_reuse=True, cv2_rgb_direct=True, norm_on_gpu=True,
            batch=0, png_decoder="cv2"):
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    from hybrid_search import tile_index as TI

    root = os.path.join(work, "%s_%s" % (mode, label))
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    prefix = os.path.join(root, "idx_tiles" if mode == "tiles" else "idx")

    cfg = Config()
    cfg.prep_cache = bool(prep_cache)
    cfg.store_fine = True
    cfg.silence_png_warnings = True
    cfg.png_decoder = png_decoder
    # 基准台钉死索引存储格式为 npz：比对函数按 .npz 读取，若跟随 config 的
    # fast_load=True（侧车 .npy）会导致 --compare-tiles/--compare-arrays 找不到文件。
    # 这里显式固定，保证 A/B 两侧与历史基线都是同一格式、可直接比对。
    cfg.fast_load = False
    if tile_flush_ms:
        cfg.tile_flush_ms = int(tile_flush_ms)
    if hasattr(cfg, "dedup_prefilter"):
        cfg.dedup_prefilter = bool(dedup_prefilter)
    if hasattr(cfg, "tile_md5_reuse"):
        cfg.tile_md5_reuse = bool(md5_reuse)
    if hasattr(cfg, "cv2_rgb_direct"):
        cfg.cv2_rgb_direct = bool(cv2_rgb_direct)
    if hasattr(cfg, "norm_on_gpu"):
        cfg.norm_on_gpu = bool(norm_on_gpu)
    if batch:
        cfg.batch = int(batch)

    eng = HybridEngine(cfg)
    # 模型/解码器预热与计时区间分离：build_s 只含真正的建库流水线
    t_load0 = time.perf_counter()
    try:
        eng._get_extractor()                      # noqa: SLF001 —— 预热，避免把加载算进建库
    except Exception:                             # noqa: BLE001
        pass
    load_s = time.perf_counter() - t_load0

    gpu = GpuWatch()
    gpu.start()
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    if mode == "tiles":
        n_items = TI.build_tiles(eng, prefix, paths=sample)
    else:
        n_items = eng.build(prefix, paths=sample)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    gpu.stop()

    res = {
        "label": label, "mode": mode, "images": len(sample), "items": int(n_items),
        "wall_s": round(wall, 3), "load_s": round(load_s, 3), "cpu_s": round(cpu, 1),
        "cores": round(cpu / max(wall, 1e-9), 2),
        "img_per_s": round(len(sample) / max(wall, 1e-9), 1),
        "items_per_s": round(n_items / max(wall, 1e-9), 1),
        "gpu_mean": round(sum(gpu.samples) / len(gpu.samples), 1) if gpu.samples else None,
        "gpu_peak": max(gpu.samples) if gpu.samples else None,
        "prep_cache": bool(prep_cache), "dedup_prefilter": bool(dedup_prefilter),
        "md5_reuse": bool(md5_reuse), "cv2_rgb_direct": bool(cv2_rgb_direct),
        "norm_on_gpu": bool(norm_on_gpu), "batch": int(batch),
        "png_decoder": png_decoder,
        "tile_flush_ms": tile_flush_ms,
        "arrays": sha_arrays(prefix),
    }
    try:
        import psutil
        res["rss_peak_mb"] = round(
            psutil.Process().memory_info().peak_wset / 2 ** 20, 0)
    except Exception:                         # noqa: BLE001
        pass
    rdir = os.path.join(work, "results")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "%s.json" % label), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    return res


def _npz_arrays(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def compare_arrays(work, a, b):
    """数组级比对（npz 是 zip 容器，含时间戳，字节比对无意义）。

    逐数组成对比较：对象数组（paths/md5s）逐元素、数值数组逐位；
    精排特征额外给最大余弦偏差。
    """
    ra = json.load(open(os.path.join(work, "results", "%s.json" % a), encoding="utf-8"))
    rb = json.load(open(os.path.join(work, "results", "%s.json" % b), encoding="utf-8"))
    da = os.path.join(work, "%s_%s" % (ra["mode"], a))
    db = os.path.join(work, "%s_%s" % (rb["mode"], b))
    for fn in sorted(os.listdir(da)):
        if not fn.endswith(".npz"):
            continue
        pa, pb = os.path.join(da, fn), os.path.join(db, fn)
        if not os.path.exists(pb):
            print("  %-28s 基线有、新版本缺 → ✗" % fn)
            continue
        A, B = _npz_arrays(pa), _npz_arrays(pb)
        print("  --- %s ---" % fn)
        for k in sorted(set(A) | set(B)):
            if k not in A or k not in B:
                print("    %-14s 键缺失 ✗" % k)
                continue
            x, y = A[k], B[k]
            if x.shape != y.shape:
                print("    %-14s shape %s vs %s → ✗" % (k, x.shape, y.shape))
                continue
            if x.dtype.kind in "OU" or y.dtype.kind in "OU":
                same = list(map(str, x)) == list(map(str, y))
                print("    %-14s 对象数组 %d 项 → %s" % (k, len(x), "一致 ✓" if same else "不一致 ✗"))
            elif x.dtype.kind == "f":
                eq = bool(np.array_equal(x, y))
                if k == "fine" and x.ndim == 2:
                    na = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
                    nb = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-9)
                    cos = float((na * nb).sum(axis=1).min())
                    print("    %-14s 逐位一致 %s | 最小余弦 %.6f（1.000000 = 完全一致）"
                          % (k, "✓" if eq else "✗", cos))
                else:
                    print("    %-14s 逐位一致 %s | 最大绝对差 %.3g"
                          % (k, "✓" if eq else "✗", float(np.abs(x - y).max())))
            else:
                eq = bool(np.array_equal(x, y))
                print("    %-14s 逐位一致 %s | 不同元素 %d"
                      % (k, "✓" if eq else "✗", int((x != y).sum())))


def compare_tile_arrays(work, a, b):
    """瓦片索引的顺序无关比对。

    瓦片落盘顺序 = 线程完成顺序（每次运行不同），所以只能按
    「(原图路径, 框)」建键来比较集合内容：块 md5 / 指纹 / Hu / 精排特征。
    """
    ra = json.load(open(os.path.join(work, "results", "%s.json" % a), encoding="utf-8"))
    rb = json.load(open(os.path.join(work, "results", "%s.json" % b), encoding="utf-8"))
    da = os.path.join(work, "tiles_%s" % a)
    db = os.path.join(work, "tiles_%s" % b)

    def snap(root):
        c = _npz_arrays(os.path.join(root, "idx_tiles.coarse.npz"))
        f = _npz_arrays(os.path.join(root, "idx_tiles.fine.npz"))
        out = {}
        for i in range(len(c["paths"])):
            key = (str(c["paths"][i]), tuple(int(v) for v in c["boxes"][i]))
            out[key] = (str(c["md5s"][i]),
                        np.asarray(c["fp"][i]).tobytes(),
                        np.asarray(c["hu"][i]).tobytes(),
                        np.asarray(f["features"][i], dtype=np.float32))
        return out

    A, B = snap(da), snap(db)
    print("=== 瓦片索引顺序无关比对 %s vs %s ===" % (a, b))
    print("  块数 %d vs %d；键集合相同 %s"
          % (len(A), len(B), "✓" if set(A) == set(B) else "✗"))
    only_a = list(set(A) - set(B))[:3]
    only_b = list(set(B) - set(A))[:3]
    if only_a or only_b:
        print("    仅 %s 有: %s" % (a, only_a))
        print("    仅 %s 有: %s" % (b, only_b))
    bad_md5 = bad_fp = bad_hu = bad_fine = 0
    min_cos = 1.0
    for k in set(A) & set(B):
        va, vb = A[k], B[k]
        bad_md5 += int(va[0] != vb[0])
        bad_fp += int(va[1] != vb[1])
        bad_hu += int(va[2] != vb[2])
        if not np.array_equal(va[3], vb[3]):
            bad_fine += 1
            na = va[3] / max(float(np.linalg.norm(va[3])), 1e-9)
            nb = vb[3] / max(float(np.linalg.norm(vb[3])), 1e-9)
            min_cos = min(min_cos, float(na @ nb))
    print("  块 md5 不同 %d ；指纹不同 %d ；Hu 不同 %d ；精排特征逐位不同 %d"
          % (bad_md5, bad_fp, bad_hu, bad_fine))
    if bad_fine:
        print("    精排特征最小余弦 %.9f（偏差 %.2e；GPU FP16 前向本身非逐位可复现）"
              % (min_cos, 1.0 - min_cos))
    ok = (set(A) == set(B)) and not (bad_md5 or bad_fp or bad_hu) and min_cos > 0.99999
    print("  结论: %s" % ("索引内容一致（块 md5/指纹/Hu 逐位一致；精排差异仅 FP16 量级）✓"
                        if ok else "存在实质差异 ✗"))
    print("  wall: %s %.2f s → %s %.2f s（%+.1f%%）；CPU 秒 %.1f → %.1f（%+.1f%%）"
          % (a, ra["wall_s"], b, rb["wall_s"],
             100 * (rb["wall_s"] - ra["wall_s"]) / ra["wall_s"],
             ra["cpu_s"], rb["cpu_s"],
             100 * (rb["cpu_s"] - ra["cpu_s"]) / max(ra["cpu_s"], 1e-9)))
    print("  吞吐: %.1f → %.1f 张/s ；CPU 核 %.2f → %.2f ；GPU 均值 %s%% → %s%%"
          % (ra["img_per_s"], rb["img_per_s"], ra["cores"], rb["cores"],
             ra["gpu_mean"], rb["gpu_mean"]))
    return ok


def compare(work, a, b):
    rdir = os.path.join(work, "results")
    ra = json.load(open(os.path.join(rdir, "%s.json" % a), encoding="utf-8"))
    rb = json.load(open(os.path.join(rdir, "%s.json" % b), encoding="utf-8"))
    print("=== 位级比对 %s vs %s（mode=%s）===" % (a, b, ra["mode"]))
    print("  %-26s %-10s %-10s %s" % ("数组", a, b, "一致"))
    keys = sorted(set(ra["arrays"]) | set(rb["arrays"]))
    allsame = True
    for k in keys:
        ha = ra["arrays"].get(k, {}).get("sha256", "-")
        hb = rb["arrays"].get(k, {}).get("sha256", "-")
        same = ha == hb
        allsame &= same
        print("  %-26s %-10s %-10s %s" % (k, ha[:8], hb[:8], "✓" if same else "✗"))
    print("  %-26s %-10s %-10s" % ("wall(s)", ra["wall_s"], rb["wall_s"]))
    print("  %-26s %-10s %-10s" % ("img/s", ra["img_per_s"], rb["img_per_s"]))
    print("  %-26s %-10s %-10s" % ("cores", ra["cores"], rb["cores"]))
    if ra["wall_s"]:
        print("  结论: %s；wall %.2f s → %.2f s（%+.1f%%）"
              % ("索引逐位一致" if allsame else "索引有差异，需排查",
                 ra["wall_s"], rb["wall_s"],
                 100 * (rb["wall_s"] - ra["wall_s"]) / ra["wall_s"]))
    return allsame


def main() -> int:
    ap = argparse.ArgumentParser(description="P0/P1 优化步骤的建库 A/B 基准")
    ap.add_argument("--mode", choices=["whole", "tiles"], default="tiles")
    ap.add_argument("--label", default=None)
    ap.add_argument("--n", type=int, default=540)
    ap.add_argument("--dup", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--prep-cache", action="store_true", help="开启预处理缓存")
    ap.add_argument("--no-dedup-prefilter", action="store_true")
    ap.add_argument("--tile-flush-ms", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--compare-arrays", nargs=2, metavar=("A", "B"))
    ap.add_argument("--compare-tiles", nargs=2, metavar=("A", "B"))
    ap.add_argument("--png-decoder", default="cv2",
                    choices=["cv2", "imagecodecs", "pillow", "libdeflate"],
                    help="PNG 解码器实现（cv2 / imagecodecs / pillow / libdeflate）")
    ap.add_argument("--batch", type=int, default=0,
                    help="前向/分块批大小（0=自动 64/16）；验证分块屏障影响用")
    ap.add_argument("--no-norm-gpu", action="store_true",
                    help="归一化留在 CPU（回退旧行为），用于 A/B 对照")
    ap.add_argument("--no-cv2-rgb", action="store_true",
                    help="关闭 cv2 RGB 直出（回退 BGR+cvtColor），用于 A/B 对照")
    ap.add_argument("--no-md5-reuse", action="store_true",
                    help="瓦片块 md5 退回旧行为（逐块整文件哈希），用于 A/B 对照")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    if a.compare_tiles:
        return 0 if compare_tile_arrays(a.work, a.compare_tiles[0],
                                       a.compare_tiles[1]) else 1
    if a.compare_arrays:
        compare_arrays(a.work, a.compare_arrays[0], a.compare_arrays[1])
        return 0
    if a.compare:
        return 0 if compare(a.work, a.compare[0], a.compare[1]) else 1
    label = a.label or ("%s_%d" % (a.mode, int(time.time())))
    sample = build_sample(a.n, a.dup, a.seed)
    print("样本: %d 张（其中 %d 张为内容重复、索引外文件）；工作目录 %s"
          % (len(sample), a.dup, a.work))
    res = run_one(a.mode, label, sample, a.work, a.prep_cache,
                  not a.no_dedup_prefilter, a.tile_flush_ms, not a.no_md5_reuse,
                  not a.no_cv2_rgb, not a.no_norm_gpu, a.batch, a.png_decoder)
    print("  [%s/%s] %d 张 → %.2f s | %.1f 张/s | %.1f 项/s | CPU %.2f 核 | "
          "GPU 均值 %s%% 峰值 %s%% | CPU 秒 %.1f"
          % (res["mode"], res["label"], res["images"], res["wall_s"],
             res["img_per_s"], res["items_per_s"], res["cores"],
             res["gpu_mean"], res["gpu_peak"], res["cpu_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
