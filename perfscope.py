# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器（二值法粗筛 + ResNet 精排）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
perfscope —— 图库只读观测仪（档案 / 效能画像 / CPU·GPU 时间轴 / 建议）

用法：
  python perfscope.py F:\\视频                       # 扫描档案 + 分层抽样解码效能 + 小型融合建库画像
  python perfscope.py F:\\视频 --scan-only          # 只做档案（最快）
  python perfscope.py F:\\视频 --no-fused           # 跳过融合建库时间轴
  python perfscope.py F:\\视频 --max-scan 5000      # 扫描上限（大库调试用）

严格只读：本工具不会创建/修改/移动/删除图库里的任何文件；
全部中间产物（扫描缓存、临时索引、HTML 报告）只写当前工作目录。

报告内容：
  1) 档案表：扩展名×真实格式、分辨率档分布、大小档分布、EXIF 方向、解码档位预估；
  2) 效能矩阵：按 (格式×分辨率档×EXIF) 分层抽样，实测 decode_gray/decode_rgb 耗时，
     折算单库 16 解码线程吞吐，标出拖后腿的类别；
  3) CPU/GPU 时间轴：小样本真实融合建库期间的 CPU%、GPU SM%、显存、img/s，
     直观看出“解码(CPU) 与 ResNet(GPU) 是否同步并行、谁在等谁”；
  4) 建议清单：只出方案，不自动改动。
"""
from __future__ import annotations

import argparse
import collections
import csv
import io as _io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp",
           ".tif", ".tiff", ".gif", ".jpe"}
VID_EXT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
           ".ts", ".m2ts", ".mpg", ".mpeg", ".3gp", ".rmvb", ".f4v",
           ".m4v", ".ogv", ".vob"}

# 分辨率档（像素数）
SIZE_BANDS = [
    ("<1MP", 1_000_000), ("1-4MP", 4_000_000), ("4-12MP", 12_000_000),
    ("12-24MP", 24_000_000), (">24MP", 2 ** 63),
]
# 大小档（字节）
BYTE_BANDS = [("<0.5MB", 0.5 * 2 ** 20), ("0.5-2MB", 2 * 2 ** 20),
              ("2-8MB", 8 * 2 ** 20), ("8-32MB", 32 * 2 ** 20),
              (">32MB", 2 ** 63)]
# 解码档位预估（与 io_utils._reduced_flag 同规则，用于报告“会走哪种解码”）
DECODE_LEVELS = [("全尺寸解码", 2560), ("1/2 域缩放", 5120),
                 ("1/4 域缩放", 10240), ("1/8 域缩放", 2 ** 63)]


def band_of(n: int, bands) -> str:
    for name, hi in bands:
        if n <= hi:
            return name
    return bands[-1][0]


def decode_level_of(max_side: int) -> str:
    for name, hi in DECODE_LEVELS:
        if max_side <= hi:
            return name
    return DECODE_LEVELS[-1][0]


# ---------------------------------------------------------------------------
# A. 扫描档案（只读头部探测，不解码像素）
# ---------------------------------------------------------------------------
def probe_image(path: str):
    """PIL 惰性读头部：返回 (fmt, w, h, orient) 或 None（不支持/损坏）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            fmt = im.format
            w, h = im.size
            try:
                orient = int(im.getexif().get(0x0112, 1))
            except Exception:  # noqa: BLE001
                orient = 1
            return fmt, w, h, orient
    except Exception:  # noqa: BLE001
        return None


def scan_gallery(root: str, ext_filter: Optional[set] = None,
                 max_files: Optional[int] = None, workers: int = 8):
    """递归统计 + 逐图片文件头部探测。绝不改动文件。"""
    entries = []                     # (path, size)
    counter = collections.Counter()
    n_video = 0
    video_bytes = 0
    other_counter = collections.Counter()
    other_bytes = 0
    n_total = 0
    t0 = time.time()
    stop = False
    for dp, _dn, fn in os.walk(root):
        if stop:
            break
        for f in fn:
            n_total += 1
            ext = os.path.splitext(f)[1].lower()
            p = os.path.join(dp, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if ext in IMG_EXT:
                if ext_filter and ext not in ext_filter:
                    continue
                entries.append((p, st.st_size))
            elif ext in VID_EXT:
                n_video += 1
                video_bytes += st.st_size
            else:
                other_counter[ext or "(无扩展名)"] += 1
                other_bytes += st.st_size
            if max_files and len(entries) >= max_files:
                stop = True
                break

    # 并行头部探测
    results: List = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(probe_image, p) for p, _s in entries]
        for (p, size), fut in zip(entries, futs):
            r = fut.result()
            if r is None:
                counter["探测失败/不支持的图片"] += 1
                continue
            fmt, w, h, orient = r
            counter["图片总张数"] += 1
            if fmt:
                counter["真实格式:" + fmt] += 1
            counter["大小档:" + band_of(size, BYTE_BANDS)] += 1
            counter["分辨率档:" + band_of(w * h, SIZE_BANDS)] += 1
            counter["解码档位:" + decode_level_of(max(w, h))] += 1
            if orient != 1:
                counter["EXIF需转正(orient!=1)"] += 1
            results.append({
                "path": p, "size": size, "fmt": fmt, "w": w, "h": h,
                "orient": orient,
            })
    counter["视频文件"] = n_video
    counter["其他文件"] = sum(other_counter.values())
    return results, dict(counter), dict(other_counter), video_bytes, time.time() - t0


# ---------------------------------------------------------------------------
# B. 分层抽样解码效能（与索引同一套 decode 代码路径，真实计时）
# ---------------------------------------------------------------------------
def class_of(item: dict) -> Tuple[str, str, bool]:
    return (item["fmt"] or "?", band_of(item["w"] * item["h"], SIZE_BANDS),
            item["orient"] != 1)


def stratified_sample(items: List[dict], per_class: int, total_cap: int,
                      seed: int = 7) -> List[dict]:
    rnd = random.Random(seed)
    groups = collections.defaultdict(list)
    for it in items:
        groups[class_of(it)].append(it)
    out = []
    for _k, v in groups.items():
        rnd.shuffle(v)
        out.extend(v[:per_class])
    rnd.shuffle(out)
    if len(out) > total_cap:
        out = out[:total_cap]
    return out


def profile_samples(samples: List[dict], quiet: bool = False):
    """对样本逐张做与索引一致的解码计时：decode_gray + decode_rgb。"""
    from hybrid_search.io_utils import decode_gray, decode_rgb, read_bytes
    rows = []
    for i, it in enumerate(samples, 1):
        data = read_bytes(it["path"])
        g = r = None
        t0 = time.time()
        if data is not None:
            g = decode_gray(data)
            t_gray = time.time() - t0
            t0 = time.time()
            r = decode_rgb(data)
            t_rgb = time.time() - t0
        else:
            t_gray = t_rgb = float("nan")
        if not quiet and (i % 50 == 0 or i == len(samples)):
            print(f"profile {i}/{len(samples)}", flush=True)
        rows.append({
            **it,
            "ms_gray": t_gray * 1000 if g is not None else None,
            "ms_rgb": t_rgb * 1000 if r is not None else None,
            "out_gray": None if g is None else list(g.shape),
            "out_rgb": None if r is None else list(r.shape),
        })
    return rows


def aggregate_profile(rows: List[dict]) -> List[dict]:
    agg = collections.defaultdict(lambda: {"n": 0, "ms_g": 0.0, "ms_r": 0.0,
                                           "mb": 0.0, "reduced": 0})
    for r in rows:
        if r["ms_rgb"] is None:
            continue
        key = (r["fmt"], band_of(r["w"] * r["h"], SIZE_BANDS),
               "EXIF" if r["orient"] != 1 else "正向")
        a = agg[key]
        a["n"] += 1
        a["ms_g"] += r["ms_gray"] or 0
        a["ms_r"] += r["ms_rgb"] or 0
        a["mb"] += r["size"] / 2 ** 20
        if r["out_rgb"]:
            ow, oh = r["out_rgb"][1], r["out_rgb"][0]
            if max(r["w"], r["h"]) / max(ow, oh) >= 1.9:
                a["reduced"] += 1
    table = []
    for (fmt, band, orient), a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        ms_avg = a["ms_r"] / a["n"]
        table.append({
            "format": fmt, "band": band, "orient": orient, "n": a["n"],
            "ms_gray": a["ms_g"] / a["n"], "ms_rgb": ms_avg,
            "mb": a["mb"] / a["n"],
            "reduced": a["reduced"],
            "imgps16": 16000.0 / max(ms_avg, 0.01),   # 16 线程折算：16000ms/s ÷ 单张ms
        })
    return table


# ---------------------------------------------------------------------------
# C. 小型融合建库 + CPU/GPU 时间轴采样
# ---------------------------------------------------------------------------
class HardwareSampler:
    """后台线程：每 0.4s 采样 CPU%(系统/进程)、GPU SM%、显存、编解码引擎、img/s。"""

    def __init__(self, img_per_s: Optional[callable] = None):
        import psutil
        self.cpu = psutil.cpu_percent
        self.proc = psutil.Process()
        _ = self.cpu(None)          # 预热首次采样
        _ = self.proc.cpu_percent(None)
        self._img_per_s = img_per_s or (lambda: 0.0)
        self.rows: List[dict] = []
        self._stop = threading.Event()
        self._io0 = None            # (读字节, 写字节) 基准
        self._last_io_t = time.time()
        try:
            self._io0 = self.proc.io_counters()
        except Exception:            # noqa: BLE001
            self._io0 = None

        self._nv = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nv = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:           # noqa: BLE001 —— 无 NVIDIA 时跳过
            self._nv = None

    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> List[dict]:
        self._stop.set()
        self._th.join(timeout=3)
        return self.rows

    def _loop(self):
        t_last = time.time()
        while not self._stop.is_set():
            time.sleep(0.4)
            now = time.time()
            row = {"t": now - t_last}
            t_last = now
            try:
                row["cpu_sys"] = self.cpu(None)
                row["cpu_proc"] = self.proc.cpu_percent(None)
            except Exception:       # noqa: BLE001
                row["cpu_sys"] = row["cpu_proc"] = None
            if self._nv is not None:
                try:
                    u = self._nv.nvmlDeviceGetUtilizationRates(self._handle)
                    row["gpu"] = u.gpu
                    row["mem%"] = u.memory
                    m = self._nv.nvmlDeviceGetMemoryInfo(self._handle)
                    row["memMB"] = m.used / 2 ** 20
                    try:
                        enc, _ = self._nv.nvmlDeviceGetEncoderUtilization(
                            self._handle)
                        row["enc%"] = enc
                    except Exception:       # noqa: BLE001
                        row["enc%"] = None
                    try:
                        dec, _ = self._nv.nvmlDeviceGetDecoderUtilization(
                            self._handle)
                        row["dec%"] = dec
                    except Exception:       # noqa: BLE001
                        row["dec%"] = None
                except Exception:           # noqa: BLE001
                    row["gpu"] = None
            row["imgps"] = self._img_per_s()
            # 进程内存（RSS）：用于判断“任务结束后是否回落/是否存在泄漏”
            try:
                mi = self.proc.memory_info()
                row["rssMB"] = mi.rss / 2 ** 20
                row["peakMB"] = getattr(mi, "peak_wset", 0) / 2 ** 20
            except Exception:       # noqa: BLE001
                row["rssMB"] = row["peakMB"] = None
            # 进程级磁盘读写速率（验证“IO 等待”假说：SSD 上应远高于瓶颈值）
            try:
                io = self.proc.io_counters()
                if self._io0 is not None:
                    dt = max(now - self._last_io_t, 1e-6)
                    row["rdMBps"] = (io.read_bytes - self._io0[0]) / 2 ** 20 / dt
                    row["wrMBps"] = (io.write_bytes - self._io0[1]) / 2 ** 20 / dt
                    self._io0 = (io.read_bytes, io.write_bytes)
                    self._last_io_t = now
                else:
                    self._last_io_t = now
            except Exception:       # noqa: BLE001
                row["rdMBps"] = row["wrMBps"] = None
            self.rows.append(row)


def run_fused_bench(samples: List[dict], max_imgs: int = 60):
    """抽样图（原路径只读）在临时目录建小型索引，期间采样 CPU/GPU。"""
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    cfg = Config()
    cfg.device = "auto"
    tmp = tempfile.mkdtemp(prefix="perfscope_fused_")
    paths = [it["path"] for it in samples[:max_imgs]]
    state = {"done": 0, "last_t": None, "last_done": 0}

    def img_per_s():
        now = time.time()
        if state["last_t"] is None:
            state["last_t"] = now
            state["last_done"] = state["done"]
            return 0.0
        dt = now - state["last_t"]
        d = state["done"] - state["last_done"]
        state["last_t"] = now
        state["last_done"] = state["done"]
        return d / dt if dt > 0 else 0.0

    def cb(done, _total, _phase):
        state["done"] = done

    sampler = HardwareSampler(img_per_s=img_per_s)
    sampler.start()
    eng = HybridEngine(cfg)
    t0 = time.time()
    try:
        n = eng.build(os.path.join(tmp, "g"), paths=paths, progress=cb)
        elapsed = time.time() - t0
    finally:
        rows = sampler.stop()
        try:
            for f in ("meta.json", "coarse.npz", "fine.npz"):
                p = os.path.join(tmp, "g." + f)
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(tmp)
        except OSError:
            pass
    return rows, n, elapsed


# ---------------------------------------------------------------------------
# 瓦片（局部）索引性能档：合成图集 建库+检索 全链路硬件画像
# ---------------------------------------------------------------------------
def _make_tiles_dataset(work: str, n_big: int, n_small: int,
                        big_w: int = 1600, big_h: int = 1000, seed: int = 7):
    """生成可区分内容的大图/小图，返回 (big_paths, small_paths, origins)。
    每张大图内容 = 纯色底 + 若干几何 + 大字编号，保证瓦片可判别。"""
    import cv2
    os.makedirs(os.path.join(work, "db", "big"), exist_ok=True)
    os.makedirs(os.path.join(work, "db", "small"), exist_ok=True)
    rng = np.random.RandomState(seed)
    big_paths, small_paths, big_anchors = [], [], []

    def enc(path, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        assert ok
        with open(path, "wb") as f:
            f.write(buf.tobytes())

    for i in range(n_big):
        import colorsys
        # 24° 色相步长(15 档一轮)；同色相轮次用左右分区避免同构混淆。
        # 背景用中高饱和/亮度：24° 色相在 RGB 上有 ~60/255 差异，保证可判别。
        round_no = i // 15
        hue = ((i % 15) * 24.0) % 360.0
        bg = colorsys.hsv_to_rgb(hue / 360.0, 0.72, 0.55)
        bg = tuple(int(v * 255) for v in bg)
        fg = colorsys.hsv_to_rgb(((hue + 150.0) % 360.0) / 360.0, 0.95, 0.97)
        fg = tuple(int(v * 255) for v in fg)
        img = np.zeros((big_h, big_w, 3), dtype=np.uint8)
        img[:] = bg
        anchor = None
        zone = (round_no % 4) * 360           # 同色相轮次分 4 个横向区域
        for k in range(4 + i % 3):
            if k == 0:
                cx = int(rng.randint(120, 260)) + zone
                cy = int(rng.randint(200, 500))
            else:
                cx, cy = rng.randint(150, big_w - 150), rng.randint(150, big_h - 150)
            cv2.circle(img, (cx, cy),
                       int(rng.randint(200, 300)) if k == 0
                       else int(rng.randint(90, 240)), fg, -1)
            cv2.rectangle(img, (cx - 70, cy + 60), (cx + 70, cy + 150),
                          (12, 12, 12), -1)
            if k == 0:
                anchor = (cx, cy)
                # 锚点圆内画超大唯一编号：为裁切查询提供强判别细节
                cv2.putText(img, f"{i:03d}", (cx - 150, cy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.6, (255, 255, 255), 14)
                cv2.putText(img, f"{i:03d}", (cx - 150, cy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.6, (10, 10, 10), 3)
        cv2.putText(img, f"B{i:03d}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 8)
        p = os.path.join(work, "db", "big", f"b{i:03d}.jpg")
        enc(p, img)
        big_paths.append(p)
        big_anchors.append(anchor)
    for i in range(n_small):
        img = rng.randint(0, 256, (420, 420, 3), dtype=np.uint8)
        cv2.putText(img, f"S{i:03d}", (60, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 7)
        p = os.path.join(work, "db", "small", f"s{i:03d}.jpg")
        enc(p, img)
        small_paths.append(p)
    return big_paths, small_paths, big_anchors


def run_tiles_bench(work: str, n_big: int = 80, n_small: int = 80,
                    seed: int = 7, tile: int = 512, overlap: float = 0.25,
                    min_side: int = 768, lsh_bits: int = 12,
                    lsh_tables: int = 8, n_queries: int = 20,
                    cand_methods=("lsh", "coarse"),
                    real_big: Optional[List[str]] = None,
                    real_small: Optional[List[str]] = None,
                    source: str = "synthetic",
                    order: str = "mixed"):
    """瓦片索引建库（全程 CPU/GPU 采样 + 文件/批次 trace）+ 检索延迟分解/召回率。
    real_big/real_small：真实图库抽样（只读）。
    order: mixed=大小图混合随机(默认)；layer=大图在前小图在后(旧行为,对照)。"""
    import random as _random
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    from hybrid_search import tile_index as T
    from hybrid_search.io_utils import LOGGER as _LOG
    import logging as _logging
    _logging.getLogger("hybrid_search").setLevel(_logging.WARNING)

    if real_big is None:
        big_paths, small_paths, big_anchors = _make_tiles_dataset(
            work, n_big, n_small, seed=seed)
        n_big = len(big_paths)
    else:
        big_paths, small_paths, big_anchors = \
            list(real_big), list(real_small or []), []
        for p in real_big:
            try:
                import cv2 as _cv2
                im = _imread_unicode(p)
                h, w = im.shape[:2]
                big_anchors.append((w // 2, h // 2))
            except Exception:            # noqa: BLE001
                big_anchors.append((0, 0))
    all_paths = big_paths + small_paths
    if order == "mixed":
        _random.Random(seed + 11).shuffle(all_paths)   # 消除“大前小后”结构
    prefix = os.path.join(work, "gallery_tiles")

    # ---- 查询集：每张大图取一个 512x512 裁切（真实局部检索） -------------
    import cv2
    qdir = os.path.join(work, "queries")
    os.makedirs(qdir, exist_ok=True)
    queries = []
    rng = np.random.RandomState(seed + 1)
    n_q = min(n_queries, len(big_paths))
    for qi in range(n_q):
        img = _imread_unicode(big_paths[qi % len(big_paths)])
        if img is None:
            continue
        h, w = img.shape[:2]
        ax, ay = big_anchors[qi % len(big_paths)]
        ax += int(rng.randint(-120, 121))
        ay += int(rng.randint(-120, 121))
        x0 = min(max(0, ax - 256), max(0, w - 512))
        y0 = min(max(0, ay - 256), max(0, h - 512))
        crop = img[y0:y0 + 512, x0:x0 + 512]
        qp = os.path.join(qdir, f"q{qi:03d}.jpg")
        cv2.imwrite(qp, crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        queries.append((qp, big_paths[qi % len(big_paths)]))

    cfg = Config()
    cfg.device = "auto"
    cfg.exclude_self = False
    eng = HybridEngine(cfg)

    # ============ A. 建库（采样 CPU/GPU/显存/瓦片吞吐 + 处理轨迹） ========
    state = {"done": 0, "last_t": None, "last_done": 0}

    def tiles_per_s():
        now = time.time()
        if state["last_t"] is None:
            state["last_t"], state["last_done"] = now, state["done"]
            return 0.0
        dt = now - state["last_t"]
        d = state["done"] - state["last_done"]
        state["last_t"], state["last_done"] = now, state["done"]
        return d / dt if dt > 0 else 0.0

    def cb(done, _total):
        state["done"] = done

    trace = T.BuildTrace()
    sampler = HardwareSampler(img_per_s=tiles_per_s)
    sampler.start()
    t0 = time.time()
    n_tiles = T.build_tiles(eng, prefix, paths=all_paths, progress=cb,
                            tile=tile, overlap=overlap, min_side=min_side,
                            trace=trace)
    build_sec = time.time() - t0
    build_rows = sampler.stop()
    trace_snap = trace.snapshot()
    _LOG.info("tiles build done")            # noqa: E800
    n_images = len(all_paths)

    # ============ B. 检索：候选法 × 查询集 ==============================
    eng2 = HybridEngine(cfg)
    eng2.open(prefix)
    # 首次查询（含 LSH 建表 / 模型加载）单独计时
    warm = T.search_tiles(eng2, queries[0][0], top_k=5, coarse_k=200,
                          method="lsh", lsh_bits=lsh_bits,
                          lsh_tables=lsh_tables)
    warm_sec = warm.times.get("total", 0.0)
    results = {}
    for method in cand_methods:
        lat = []
        hits_top1 = 0
        hits_top5 = 0
        stage_agg: dict = {}
        for qp, origin in queries:
            o = T.search_tiles(eng2, qp, top_k=5, coarse_k=200,
                               method=method, lsh_bits=lsh_bits,
                               lsh_tables=lsh_tables)
            paths5 = [os.path.normcase(h.path) for h in o.hits[:5]]
            if o.hits and os.path.normcase(o.hits[0].path) == \
                    os.path.normcase(origin):
                hits_top1 += 1
            if os.path.normcase(origin) in paths5:
                hits_top5 += 1
            for k, v in o.times.items():
                if k != "total":
                    stage_agg[k] = stage_agg.get(k, 0.0) + v
            lat.append(o.times.get("total", 0.0))
        results[method] = {
            "lat": lat,
            "stage": {k: v / max(len(queries), 1)
                      for k, v in stage_agg.items()},
            "top1": hits_top1, "top5": hits_top5,
            "n_q": len(queries),
        }
    decoders = None
    if real_big is not None:
        pool = (big_paths[:7] + small_paths[:7]
                if small_paths else big_paths[:12])
        try:
            decoders = _bench_decoders(pool)
        except Exception as e:            # noqa: BLE001 —— 对照失败不阻断主报告
            decoders = {"error": repr(e)}
    return {
        "n_images": n_images, "n_tiles": n_tiles, "build_sec": build_sec,
        "build_rows": build_rows,
        "avg_tiles_per_img": n_tiles / max(n_images, 1),
        "warm_sec": warm_sec, "results": results,
        "trace": trace_snap, "source": source, "decoders": decoders,
        "params": {"tile": tile, "overlap": overlap, "min_side": min_side,
                   "lsh_bits": lsh_bits, "lsh_tables": lsh_tables,
                   "n_big": n_big, "n_small": n_small, "order": order},
    }


# ---------------------------------------------------------------------------
# D. HTML 报告（纯内嵌 CSS/SVG，零依赖）
# ---------------------------------------------------------------------------
def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_cell(v: float, vmax: float, color: str = "#3d8fd1") -> str:
    w = 0 if vmax <= 0 else max(2.0, v / vmax * 100)
    return (f'<td style="min-width:140px"><div style="background:{color};'
            f'height:14px;width:{w:.1f}%;border-radius:2px"></div></td>')


def svg_line(series: List[dict], keys: List[Tuple[str, str, str]],
             w: int = 900, h: int = 240) -> str:
    """keys: (key, 颜色, 名称)。各序列独立归一化到 0..100 后再同图显示，
    图例标注各自峰值；用于把 %（CPU/GPU）与 img/s（不同量纲）画在一起。
    注意：series[].t 是采样间隔(dt)，这里先做累计 -> x 轴为真实时间轴，
    点从左到右按时间均匀分布。"""
    if not series:
        return "<p>（无采样数据）</p>"
    # t 为每行间隔：累计成真实时间轴（修复：不能拿单个 dt 当总时长，
    # 否则所有点被除成同一坐标，折线全部缩到一侧/出界）
    cum = 0.0
    ts = []
    for r in series:
        cum += max(float(r.get("t") or 0.0), 0.0)
        ts.append(cum)
    total_t = max(ts[-1], 1e-6)
    pad_l, pad_b, pad_t, pad_r = 46, 24, 12, 10
    inner_w, inner_h = w - pad_l - pad_r, h - pad_t - pad_b
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" '
             f'style="background:#0d1117;border-radius:6px">']
    for i in range(6):
        y = pad_t + inner_h * i / 5
        parts.append(f'<line x1="{pad_l}" y1="{y:.0f}" x2="{w - pad_r}" '
                     f'y2="{y:.0f}" stroke="#1f2730" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 3:.0f}" '
                     f'text-anchor="end" fill="#7d8b96" font-size="10">'
                     f"{100 - i * 20}</text>")
    parts.append(f'<text x="{pad_l}" y="{h - 4}" fill="#7d8b96" '
                 f'font-size="10">时间 → 总 {total_t:.1f}s</text>')

    def t_of(idx):
        return pad_l + inner_w * (ts[idx] / total_t)

    legend_x = pad_l + 110
    for key, color, name in keys:
        vals = [r.get(key) for r in series if r.get(key) is not None]
        if len(vals) < 2:
            continue
        peak = max(max(float(v) for v in vals), 1e-6)
        pts = []
        for ri, r in enumerate(series):
            v = r.get(key)
            if v is None:
                continue
            y = pad_t + inner_h * (1 - min(max(float(v) / peak, 0), 1.0))
            pts.append(f"{t_of(ri):.1f},{y:.1f}")
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                     f'stroke="{color}" stroke-width="1.6" opacity="0.9"/>')
        parts.append(f'<rect x="{legend_x}" y="{pad_t + 4}" width="10" '
                     f'height="10" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 14}" y="{pad_t + 13}" '
                     f'fill="#c8d0d6" font-size="10">{esc(name)} '
                     f'(峰值 {peak:.0f})</text>')
        legend_x += 165
    parts.append("</svg>")
    return "".join(parts)


def build_html(report: dict) -> str:
    c = report["counter"]
    prof = report["profile"]
    n_img = c.get("图片总张数", 0)
    # 柱状辅助
    size_rows = [(k.replace("分辨率档:", ""), v)
                 for k, v in c.items() if k.startswith("分辨率档:")]
    fmt_rows = [(k.replace("真实格式:", ""), v)
                for k, v in c.items() if k.startswith("真实格式:")]
    fmt_max = max([v for _k, v in fmt_rows] or [1])
    size_max = max([v for _k, v in size_rows] or [1])

    html = ["""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>图库效能观测报告</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;
margin:0;padding:20px}h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;
margin-top:28px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12.5px}
th,td{border:1px solid #26323d;padding:4px 8px;text-align:left}th{background:#1a222b}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96}.warn{color:#ffd28f}code{background:#1a222b;padding:1px 5px}
</style></head><body>"""]

    html.append(f"<h1>图库效能观测 · 只读报告</h1>")
    html.append(f"<p class='muted'>图库: <code>{esc(report['root'])}</code> · "
                f"扫描耗时 {report['scan_sec']:.1f}s · "
                f"生成 {report['ts']}</p>")

    # 1) 档案总览
    html.append("<h2>1. 档案总览</h2><table><tr><th>项</th><th>数值</th></tr>")
    for k in ("图片总张数", "视频文件", "其他文件", "探测失败/不支持的图片"):
        html.append(f"<tr><td>{esc(k)}</td><td class='num'>{c.get(k, 0):,}</td></tr>")
    html.append(f"<tr><td>EXIF 需转正(orient!=1)</td>"
                f"<td class='num'>{c.get('EXIF需转正(orient!=1)', 0):,} "
                f"({c.get('EXIF需转正(orient!=1)', 0) / max(n_img, 1) * 100:.1f}%)"
                f"</td></tr></table>")

    html.append("<h2>2. 真实格式分布（PIL 头部识别，非扩展名）</h2><table>"
                "<tr><th>格式</th><th>张数</th><th>占比</th><th></th></tr>")
    for name, v in fmt_rows:
        html.append(f"<tr><td>{esc(name)}</td><td class='num'>{v:,}</td>"
                    f"<td class='num'>{v / max(n_img, 1) * 100:.1f}%</td>"
                    f"{bar_cell(v, fmt_max, '#3d8fd1')}</tr>")

    html.append("<h2>3. 分辨率档 / 大小档 / 预计解码档位</h2>")
    html.append("<table><tr><th>分辨率档</th><th>张数</th><th></th></tr>")
    for name, v in size_rows:
        html.append(f"<tr><td>{esc(name)}</td><td class='num'>{v:,}</td>"
                    f"{bar_cell(v, size_max, '#5aa469')}</tr>")
    html.append("</table><table><tr><th>大小档</th><th>张数</th></tr>")
    for k, v in c.items():
        if k.startswith("大小档:"):
            html.append(f"<tr><td>{esc(k[4:])}</td><td class='num'>{v:,}</td></tr>")
    html.append("</table><table><tr><th>解码方式（按边长预估）</th>"
                "<th>张数</th><th>说明</th></tr>")
    notes = {"全尺寸解码": "小图直接解码", "1/2 域缩放": "解码器输出一半像素",
             "1/4 域缩放": "更省", "1/8 域缩放": "最大压缩档"}
    for k, v in c.items():
        if k.startswith("解码档位:"):
            name = k[5:]
            html.append(f"<tr><td>{esc(name)}</td><td class='num'>{v:,}</td>"
                        f"<td class='muted'>{esc(notes.get(name, ''))}</td></tr>")
    html.append("</table>")

    # 4) 解码效能矩阵
    html.append("<h2>4. 抽样解码效能矩阵（真实 decode 计时）</h2>")
    if prof:
        p_max = max([p["ms_rgb"] for p in prof] or [1])
        html.append("<table><tr><th>格式</th><th>分辨率档</th><th>方向</th>"
                    "<th>样本</th><th>平均MB</th><th>灰度ms/张</th>"
                    "<th>RGB ms/张</th><th>折16线程估 张/秒</th>"
                    "<th>走域缩放样本</th><th></th></tr>")
        for p in prof:
            html.append(
                f"<tr><td>{esc(p['format'])}</td><td>{esc(p['band'])}</td>"
                f"<td>{esc(p['orient'])}</td><td class='num'>{p['n']}</td>"
                f"<td class='num'>{p['mb']:.1f}</td>"
                f"<td class='num'>{p['ms_gray']:.0f}</td>"
                f"<td class='num'>{p['ms_rgb']:.0f}</td>"
                f"<td class='num'>{p['imgps16']:.0f}</td>"
                f"<td class='num'>{p['reduced']}/{p['n']}</td>"
                f"{bar_cell(p['ms_rgb'], p_max, '#d17d3d')}</tr>")
        html.append("</table>")
        slow = sorted(prof, key=lambda x: -x["ms_rgb"])[:3]
        html.append("<p class='muted'>最耗时类别（可能拖累整体建库）："
                    + "、".join(f"{esc(p['format'])} {esc(p['band'])}"
                                f"({p['ms_rgb']:.0f}ms/张)" for p in slow)
                    + "</p>")
    else:
        html.append("<p class='muted'>无样本数据</p>")

    # 5) CPU/GPU 时间轴
    if report.get("fused_rows"):
        rows = report["fused_rows"]
        html.append("<h2>5. 融合建库 CPU/GPU 时间轴</h2>")
        html.append(f"<p class='muted'>小样本 {report.get('fused_n', 0)} 张 "
                    f"实际融合建库：耗时 {report.get('fused_sec', 0):.1f}s，"
                    f"其中 CPU%(系统/进程)、GPU%(SM 利用率)、显存、img/s "
                    f"每 0.4s 采样一次</p>")
        html.append(svg_line(rows, [("cpu_sys", "#5aa469", "CPU 总%"),
                                    ("gpu", "#3d8fd1", "GPU SM%"),
                                    ("mem%", "#b48ad9", "显存%"),
                                    ("imgps", "#d17d3d", "img/s×1")]))
        # 附加 img/s 放缩到 % 坐标系的说明与同步判定
        html.append("<p class='muted'>“CPU 绿线高 + GPU 蓝线高且并存”= "
                    "解码(CPU) 与 ResNet(GPU) 同步并行；若蓝线长期 0 而绿线忙"
                    " = 纯 CPU 阶段；若两者都低 = 磁盘/等待瓶颈。</p>")

    # 6) 建议
    html.append("<h2>6. 观察与建议（方案，未自动改动）</h2><ul>")
    for s in report["suggestions"]:
        html.append(f"<li>{s}</li>")
    html.append("</ul>")
    html.append("</body></html>")
    return "".join(html)


# ---------------------------------------------------------------------------
# 建议生成（数据驱动）
# ---------------------------------------------------------------------------
def make_suggestions(report: dict) -> List[str]:
    c = report["counter"]
    prof = report["profile"]
    sug = []
    n_img = c.get("图片总张数", 0)
    ext_dist = report.get("ext_dist")
    if n_img == 0:
        sug.append("未发现可索引图片：请确认目录内为图片文件或调整扩展名配置。")
        return sug

    png = c.get("真实格式:PNG", 0)
    if png and png / n_img > 0.25:
        sug.append(
            f"PNG 占比 {png / n_img * 100:.0f}%（{png:,} 张）。PNG 解码走 Pillow "
            "全尺寸（无域缩放），大分辨率 PNG 是全库最慢类。方案：a) 图库本体改为"
            "无损 JPEG2000/WebP 或另建 WebP 代理图；b) 需要像素级原图时可评估 "
            "OpenCV PNG 域缩放解码（代价：放弃 Pillow 的静默通道）；"
            "c) 若大量为截图/长图，考虑按需分块解码（改动较大）。")
    big = sum(v for k, v in c.items()
              if k.startswith("分辨率档:>24MP"))
    if big:
        sug.append(
            f">24MP 超清图 {big:,} 张：已在解码器侧域缩放档位覆盖；若效能矩阵显示"
            "这些图仍慢，瓶颈多为 JPEG 熵解码（CPU 硬成本），下一步选项："
            "nvJPEG/DALI 走 GPU 批量解码（需引入二进制依赖）。")
    exif_n = c.get("EXIF需转正(orient!=1)", 0)
    if exif_n and exif_n / n_img > 0.05:
        sug.append(
            f"{exif_n:,} 张（{exif_n / n_img * 100:.1f}%）带非正 EXIF 方向："
            "当前 OpenCV 自动转正（全尺寸与域缩放档均支持），开销已并入解码计时；"
            "若手机照片多建议用 exiftool 类工具离线转正（可选优化，非必需）。")
    if prof:
        slow = sorted(prof, key=lambda x: -x["ms_rgb"])[:3]
        top = slow[0]
        sug.append(
            f"效能矩阵最慢类别 {top['format']} {top['band']} 平均 "
            f"{top['ms_rgb']:.0f}ms/张。若该类占比高，整体建库时间由它主导；"
            "可用 --no-store-fine 或 decode-workers 调优前先跑一次真实小样本确认。")
    gif = c.get("真实格式:GIF", 0)
    if gif:
        sug.append(f"GIF {gif:,} 张仅取首帧特征（当前行为）。")
    return sug


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _cache_path(out: str, root: str) -> str:
    import hashlib
    h = hashlib.md5(root.encode("utf-8")).hexdigest()[:10]
    return os.path.join(out, f"scan_cache_{h}.json")


def _load_cache(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data["items"], data["counter"], data["other"],
                data["video_bytes"])
    except Exception:            # noqa: BLE001 —— 缓存缺失/损坏即重扫
        return None


def _save_cache(path: str, items, counter, other, video_bytes) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"items": items, "counter": counter,
                       "other": other, "video_bytes": video_bytes}, f)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 瓦片（局部）索引性能图纸（HTML）
# ---------------------------------------------------------------------------
def _tiles_html(report: dict) -> str:
    import statistics
    p = report["params"]
    rows = report["build_rows"]
    out = ["<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
           "<title>局部(瓦片)索引性能报告</title><style>"
           "body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;"
           "color:#d7dee4;margin:0;padding:20px}h1{font-size:20px}"
           "h2{font-size:15px;color:#9fd0ff;margin-top:26px;"
           "border-bottom:1px solid #26323d;padding-bottom:6px}"
           "table{border-collapse:collapse;width:100%;margin:8px 0;"
           "font-size:12.5px}th,td{border:1px solid #26323d;padding:4px 8px;"
           "text-align:left}th{background:#1a222b}td.num{text-align:right;"
           "font-variant-numeric:tabular-nums}.muted{color:#7d8b96}"
           ".warn{color:#ffd28f}.good{color:#7fdb9a}code{background:#1a222b;"
           "padding:1px 5px}svg{background:#121820;border:1px solid #26323d}"
           "</style></head><body>"]
    out.append(f"<h1>局部(瓦片)索引性能 · 图纸报告</h1>")
    out.append(f"<p class='muted'>生成 {report['ts']} · 工作目录 "
               f"<code>{esc(report['work'])}</code></p>")

    out.append("<h2>1. 建库规模与参数</h2><table>"
               "<tr><th>原图数</th><th>瓦片总数</th><th>平均瓦片/图</th>"
               "<th>tile</th><th>overlap</th><th>min_side</th>"
               "<th>LSH bits/表</th></tr>")
    out.append(f"<tr><td class='num'>{report['n_images']}</td>"
               f"<td class='num'>{report['n_tiles']:,}</td>"
               f"<td class='num'>{report['avg_tiles_per_img']:.1f}</td>"
               f"<td class='num'>{p['tile']}</td><td class='num'>{p['overlap']}</td>"
               f"<td class='num'>{p['min_side']}</td>"
               f"<td class='num'>{p['lsh_bits']}/{p['lsh_tables']}</td></tr></table>")

    out.append("<h2>2. 建库全程硬件时间轴（0.4s 采样）</h2>")
    if rows:
        keys = [("cpu_sys", "#e0a458", "CPU 系统%"),
                ("cpu_proc", "#d17a6f", "CPU 进程%"),
                ("gpu", "#4da3d6", "GPU SM%"),
                ("rdMBps", "#63c9a2", "读MB/s"),
                ("wrMBps", "#e08a5a", "写MB/s"),
                ("mem%", "#7fdb9a", "显存%")]
        out.append(svg_line(rows, keys, w=960, h=260))
        out.append("<table><tr><th>指标</th><th>均值</th><th>峰值</th></tr>")
        for k, name in (("cpu_sys", "CPU 系统%"), ("cpu_proc", "CPU 进程%"),
                        ("gpu", "GPU SM%"), ("rdMBps", "读 MB/s"),
                        ("wrMBps", "写 MB/s"), ("mem%", "显存%")):
            vals = [r.get(k) for r in rows if r.get(k) is not None]
            if vals:
                out.append(f"<tr><td>{name}</td>"
                           f"<td class='num'>{statistics.mean(vals):.0f}</td>"
                           f"<td class='num'>{max(vals):.0f}</td></tr>")
        tps = [r.get("imgps") for r in rows if r.get("imgps")]
        if tps:
            out.append(f"<tr><td>瓦片/秒(采样)</td>"
                       f"<td class='num'>{statistics.mean(tps):.0f}</td>"
                       f"<td class='num'>{max(tps):.0f}</td></tr>")
        out.append("</table>")
    order_txt = {"mixed": "大小图混合随机顺序", "layer": "大图在前小图在后"}
    out.append(f"<p>建库总耗时 <b>{report['build_sec']:.1f}s</b>（"
               f"{report['n_images']} 张原图 → {report['n_tiles']:,} 瓦片，"
               f"吞吐 {report['n_tiles'] / max(report['build_sec'], 1e-6):.0f} 瓦片/s）"
               f" · 抽样顺序：{order_txt.get(report.get('params', {}).get('order'), '?')}"
               f" · 首查暖机(建LSH表+模型加载) {report['warm_sec'] * 1000:.0f}ms</p>")

    out.append("<h2>2.5 文件处理过程 × 利用率（锯齿诊断）</h2>")
    out.append(_trace_chart(report))
    out.append(_segment_trend(report))
    if report.get("decoders"):
        out.append("<h2>2.6 解码工具对照（同批真实文件，CPU vs GPU）</h2>")
        out.append(_decoder_table(report["decoders"]))

    out.append("<h2>3. 检索延迟分解与召回（Top1 命中率）</h2>")
    for method, r in report["results"].items():
        lat = sorted(r["lat"])
        n = len(lat)
        p50 = lat[n // 2] if n else 0
        p95 = lat[min(n - 1, int(n * 0.95))] if n else 0
        tag = "LSH 近似候选+指纹复核" if method == "lsh" else "全库指纹线性扫描(对照)"
        out.append(f"<h2 style='margin-top:14px'>3.{1 if method == 'lsh' else 2} "
                   f"{method} —— {tag}</h2>")
        out.append(
            f"<p>Top1 命中 <b>{r['top1']}/{r['n_q']}</b> "
            f"({r['top1'] / max(r['n_q'], 1) * 100:.0f}%) · "
            f"<b>Recall@5</b> <b>{r['top5']}/{r['n_q']}</b> "
            f"({r['top5'] / max(r['n_q'], 1) * 100:.0f}%) · "
            f"单次检索 min {min(lat) * 1000:.1f}ms / "
            f"中位 {p50 * 1000:.1f}ms / p95 {p95 * 1000:.1f}ms "
            f"<span class='muted'>（Top1 受近邻歧义影响，Recall@5 反映"
            f"正确项是否被管线保留）</span></p>")
        if r["stage"]:
            out.append("<table><tr><th>阶段(平均)</th><th>耗时</th></tr>")
            for k, v in sorted(r["stage"].items(), key=lambda kv: -kv[1]):
                out.append(f"<tr><td>{esc(k)}</td>"
                           f"<td class='num'>{v * 1000:.2f} ms</td></tr>")
            out.append("</table>")
    return "".join(out) + "</body></html>"


def _trace_chart(report: dict) -> str:
    """文件/批次处理过程甘特 + GPU 利用率对齐 + 空窗统计（锯齿诊断）。
    时间轴统一为 0..T 秒；蓝条=每张原图的解码窗口(高 2px,完成序排列)，
    顶部红/橙=每次 GPU 前向批(宽=耗时,高=行数/峰值)，与 GPU% 曲线同框。"""
    import statistics
    trace = report.get("trace") or {}
    files = trace.get("files") or []
    batches = trace.get("batches") or []
    build_sec = max(report.get("build_sec", 0.0), 1e-6)
    parts = []
    if not files:
        return "<p class='muted'>（无轨迹数据）</p>"
    w, h = 960, 340
    pad_l, pad_r, pad_t, pad_b = 52, 12, 14, 20
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    # 三区：上 55% GPU 利用率曲线；中 30% 文件解码条；下 12% 批条
    y_gpu = pad_t
    h_gpu = ih * 0.40
    y_files = pad_t + h_gpu + 10
    h_files = ih * 0.34
    y_batch = pad_t + h_gpu + h_files + 18
    h_batch = ih * 0.16

    def X(sec):
        return pad_l + iw * (sec / build_sec)

    parts.append(f'<svg viewBox="0 0 {w} {h}" width="100%" '
                 f'style="background:#0d1117;border-radius:6px">')
    # 网格（时间刻度，按时长自动分 6~12 段）
    n_ticks = min(12, max(6, int(build_sec // 1) + 2))
    for i in range(n_ticks + 1):
        sec = build_sec * i / n_ticks
        x = X(sec)
        parts.append(f'<line x1="{x:.0f}" y1="{pad_t}" x2="{x:.0f}" '
                     f'y2="{h - pad_b}" stroke="#1c242e" stroke-width="1"/>')
        parts.append(f'<text x="{x:.0f}" y="{h - 6}" fill="#7d8b96" '
                     f'font-size="9" text-anchor="middle">{sec:.1f}s</text>')
    # GPU% 曲线（合并 0.4s 采样）
    rows = report.get("build_rows") or []
    if rows:
        cum = 0.0
        pts = []
        for r in rows:
            cum += max(float(r.get("t") or 0), 0.0)
            g = r.get("gpu")
            if g is not None:
                pts.append((pad_l + iw * (cum / build_sec),
                            y_gpu + h_gpu * (1 - min(max(g / 100.0, 0), 1))))
        if len(pts) > 1:
            parts.append('<polyline points="' + " ".join(
                f"{x:.1f},{y:.1f}" for x, y in pts) + '" fill="none" '
                'stroke="#4da3d6" stroke-width="1.8" opacity="0.95"/>')
        parts.append(f'<text x="{pad_l}" y="{y_gpu + 8}" fill="#4da3d6" '
                     f'font-size="10">GPU SM% (0.4s 采样)</text>')
    # 文件解码窗（按完成顺序排行）
    fmax_tiles = max([f["tiles"] for f in files] + [1])
    y_row = y_files
    row_step = min(h_files / max(len(files), 1), 3.0)
    for i, f in enumerate(sorted(files, key=lambda x: x["t1"])):
        x0 = X(max(0.0, f["t0"]))
        x1 = X(min(max(f["t1"], f["t0"] + 1e-4), build_sec))
        if x1 - x0 < 0.6:
            x1 = x0 + 0.6
        frac = f["tiles"] / fmax_tiles
        col = f"rgb({int(70 + frac * 150)},{int(140 + frac * 60)},{255})"
        yy = y_files + i * row_step
        parts.append(f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{x1 - x0:.1f}" '
                     f'height="{max(1.0, row_step - 0.6):.1f}" fill="{col}" '
                     f'opacity="0.85"><title>{esc(f["path"])} '
                     f'{f["tiles"]} 块</title></rect>')
    parts.append(f'<text x="{pad_l}" y="{y_files + 8}" fill="#8ab4ff" '
                 f'font-size="10">逐文件解码窗(每张 1 条; 色越亮瓦片越多)</text>')
    # GPU 前向批
    brow_max = max([b["rows"] for b in batches] + [1])
    for b in batches:
        x0 = X(max(0.0, b["t0"]))
        x1 = X(min(max(b["t1"], b["t0"] + 1e-4), build_sec))
        if x1 - x0 < 0.6:
            x1 = x0 + 0.6
        bh = max(1.5, h_batch * b["rows"] / brow_max)
        parts.append(f'<rect x="{x0:.1f}" y="{y_batch + h_batch - bh:.1f}" '
                     f'width="{x1 - x0:.1f}" height="{bh:.1f}" fill="#ff7b4a" '
                     f'opacity="0.9"><title>批 {b["rows"]} 行</title></rect>')
    parts.append(f'<text x="{pad_l}" y="{y_batch + 8}" fill="#ff7b4a" '
                 f'font-size="10">GPU 前向批(橙, 高=行数/峰值)</text>')
    parts.append("</svg>")

    # ---- 空窗统计：批间 gap、GPU 低占用占比 -----------------------------
    gaps = []
    if len(batches) > 1:
        prev_end = None
        for b in sorted(batches, key=lambda x: x["t0"]):
            if prev_end is not None:
                g = b["t0"] - prev_end
                if g > 0.001:
                    gaps.append(g)
            prev_end = max(prev_end or 0.0, b["t1"])
    gpu_low = [r.get("gpu") for r in rows
               if r.get("gpu") is not None and r["gpu"] < 25]
    gpu_all = [r.get("gpu") for r in rows if r.get("gpu") is not None]
    dec_dur = [f["t1"] - f["t0"] for f in files]
    parts.append("<table><tr><th>指标</th><th>数值</th><th>说明</th></tr>")
    parts.append(f"<tr><td>文件数 / 解码窗口</td><td class='num'>{len(files)} 张"
                 f"（中位 {statistics.median(dec_dur) * 1000:.0f} ms，"
                 f"最长 {max(dec_dur) * 1000:.0f} ms）</td>"
                 f"<td class='muted'>解码长尾图会拖慢瓦片供给</td></tr>")
    parts.append(f"<tr><td>GPU 前向批次数</td><td class='num'>{len(batches)}"
                 f"（平均 {sum(b['rows'] for b in batches) / max(len(batches), 1):.0f}"
                 f" 行/批）</td><td class='muted'>流式批：不足批时 10ms tick 发出</td></tr>")
    if gaps:
        parts.append(f"<tr><td>批间空窗(gap&gt;1ms)</td><td class='num'>"
                     f"{len(gaps)} 次，中位 {statistics.median(gaps) * 1000:.1f} ms，"
                     f"最大 {max(gaps) * 1000:.1f} ms，累计 "
                     f"{sum(gaps):.2f} s</td>"
                     f"<td class='muted'>GPU 无活可干的累计时长（锯齿谷）</td></tr>")
    else:
        parts.append("<tr><td>批间空窗</td><td class='num'>0</td>"
                     "<td class='muted'>无（GPU 恒有活干）</td></tr>")
    if gpu_all:
        parts.append(f"<tr><td>GPU 利用率分布</td><td class='num'>均值 "
                     f"{statistics.mean(gpu_all):.0f}%，峰值 {max(gpu_all):.0f}%，"
                     f"&lt;25% 采样 {len(gpu_low)}/"
                     f"{len(gpu_all)} ({len(gpu_low) / len(gpu_all) * 100:.0f}%)</td>"
                     f"<td class='muted'>低占用占比高=空窗/启动占主导</td></tr>")
    parts.append("</table>")
    return "".join(parts)


def _segment_trend(report: dict) -> str:
    """把建库时间轴均分 4 段，统计各段 CPU/GPU/显存/吞吐均值，
    暴露“后半段性能下降”（缓存/IO/资源退化），并给出 GPU 过闲判定。"""
    import statistics
    rows = report.get("build_rows") or []
    if len(rows) < 8:
        return ""
    parts = ["<table><tr><th>时间分段</th><th>CPU 系统%</th><th>GPU SM%</th>"
             "<th>显存%</th><th>瓦片/s</th></tr>"]
    n = len(rows)
    segs = 4
    stats_rows = []
    for s in range(segs):
        seg = rows[s * n // segs:(s + 1) * n // segs] or rows[-1:]
        def avg(key):
            vs = [r.get(key) for r in seg if r.get(key) is not None]
            return statistics.mean(vs) if vs else float("nan")
        stats_rows.append((s, avg("cpu_sys"), avg("gpu"),
                           avg("mem%"), avg("imgps")))
        lab = ["前 25%", "25-50%", "50-75%", "后 25%"][s]
        parts.append(f"<tr><td>{lab}</td><td class='num'>"
                     f"{stats_rows[-1][1]:.0f}</td><td class='num'>"
                     f"{stats_rows[-1][2]:.0f}</td><td class='num'>"
                     f"{stats_rows[-1][3]:.0f}</td><td class='num'>"
                     f"{stats_rows[-1][4]:.0f}</td></tr>")
    parts.append("</table>")
    # 趋势诊断
    g_first, g_last = stats_rows[0][2], stats_rows[-1][2]
    c_first, c_last = stats_rows[0][1], stats_rows[-1][1]
    t_first, t_last = stats_rows[0][4], stats_rows[-1][4]
    notes = []
    if g_last < 20 and c_last > 55 and g_first > 30:
        notes.append(f"尾段 GPU 过闲({g_last:.0f}%)而 CPU 满载"
                     f"({c_last:.0f}%)：解码/预处理供给退化——若可用 GPU 解码器"
                     f"(nvImageCodec/nvJPEG)可把熵解码卸载到 GPU；本机实测 JPEG "
                     f"nvJPEG 因全尺寸输出反而更慢、PNG 无可用 GPU 硬解包，"
                     f"优先排查解码/IO 队列")
    if g_last < 20 and c_last < 30 and g_first > 30:
        notes.append(f"尾段 CPU/GPU 双低({c_last:.0f}%/{g_last:.0f}%)且吞吐 "
                     f"{t_first:.0f}→{t_last:.0f} 瓦片/s：读写/等待占主导"
                     f"（磁盘缓存耗尽/页交换/锁），先查 IO 层")
    if t_first > t_last * 1.5:
        notes.append(f"吞吐从 {t_first:.0f} 降至 {t_last:.0f} 瓦片/s"
                     f"（-{(1 - t_last / max(t_first, 1e-6)) * 100:.0f}%）")
    if notes:
        parts.append("<p class='warn'>趋势诊断：<br>· "
                     + "<br>· ".join(notes) + "</p>")
    return "".join(parts)


def _bench_decoders(paths: List[str]) -> dict:
    """同批真实文件 × 各解码工具（当前可用）的逐样本耗时：
      cpu_cv2   : io_utils.decode_rgb（JPEG 域缩放 / PNG libpng，建库现状）
      gpu_nvjpeg: torchvision nvJPEG（仅 JPEG，GPU 全尺寸）
      pillow    : PIL 全尺寸（对照）
    返回 {samples: [{name, fmt, w, h, ms:{tool:..}}]} 供 HTML 与决策。"""
    import cv2 as _cv2
    from hybrid_search.io_utils import decode_rgb, set_png_decoder
    set_png_decoder("cv2")
    jpeg_ok = False
    try:
        from torchvision.io import decode_jpeg  # noqa: PLC0415
        _ = decode_jpeg
        jpeg_ok = True
    except Exception:            # noqa: BLE001
        jpeg_ok = False
    import torch
    samples = []
    for p in paths[:14]:
        data = open(p, "rb").read()
        fmt = "PNG" if p.lower().endswith(".png") else "JPEG"
        ms = {}
        for r in range(2):       # 预热
            decode_rgb(data)
        ts = []
        for r in range(3):
            t0 = time.perf_counter()
            arr = decode_rgb(data)
            ts.append(time.perf_counter() - t0)
        ms["cpu_cv2(现状)"] = sorted(ts)[1] * 1000
        if fmt == "JPEG" and jpeg_ok and torch.cuda.is_available():
            ts = []
            for r in range(3):
                t0 = time.perf_counter()
                j = torch.tensor(np.frombuffer(data, dtype=np.uint8))
                out = decode_jpeg(j, device="cuda")
                _ = out.cpu()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ms["gpu_nvJPEG"] = sorted(ts)[1] * 1000
        ts = []
        for r in range(3):
            from PIL import Image as _Im
            import io as _io
            t0 = time.perf_counter()
            with _Im.open(_io.BytesIO(data)) as im:
                _ = im.convert("RGB")
            ts.append(time.perf_counter() - t0)
        ms["pillow"] = sorted(ts)[1] * 1000
        h, w = arr.shape[:2]
        samples.append({"name": os.path.basename(p)[:26], "fmt": fmt,
                        "w": w, "h": h, "mb": len(data) / 1e6, "ms": ms})
    return {"samples": samples,
            "gpu_jpeg_available": bool(jpeg_ok and torch.cuda.is_available()),
            "png_gpu_available": False,
            "png_gpu_note": "nvImageCodec 无 Windows wheel 可装(上游 v0.9.0 无资产)，"
                            "PNG GPU 硬解码本机不可用"}


def _decoder_table(dec: dict) -> str:
    samples = dec["samples"]
    tools = ["cpu_cv2(现状)", "gpu_nvJPEG", "pillow"]
    parts = ["<table><tr><th>样本</th><th>格式</th><th>尺寸</th><th>MB</th>"]
    for t in tools:
        parts.append(f"<th>{t}</th>")
    parts.append("</tr>")
    for s in samples:
        parts.append(f"<tr><td>{esc(s['name'])}</td><td>{s['fmt']}</td>"
                     f"<td class='num'>{s['w']}x{s['h']}</td>"
                     f"<td class='num'>{s['mb']:.1f}</td>")
        for t in tools:
            v = s["ms"].get(t)
            parts.append(f"<td class='num'>{v:.1f} ms"
                         f"</td>" if v is not None else "<td>n/a</td>")
        parts.append("</tr>")
    parts.append("</table>")
    note = ("<p class='muted'>结论：JPEG 硬解码 GPU 需全尺寸输出+回传"
            "（12MP≈40MB），实测不优于 CPU 域缩放(输出≤2048 省 3× 带宽)；"
            + ("PNG：" + dec["png_gpu_note"] if dec.get("png_gpu_note")
               else "") + "</p>")
    parts.append(note)
    return "".join(parts)


def _make_tiles_suggestions(report: dict) -> List[str]:
    import statistics
    out: List[str] = []
    rows = report["build_rows"]
    cpu = [r.get("cpu_sys") for r in rows if r.get("cpu_sys") is not None]
    gpu = [r.get("gpu") for r in rows if r.get("gpu") is not None]
    if gpu:
        g_avg = statistics.mean(gpu)
        c_avg = statistics.mean(cpu) if cpu else 0
        tr = report.get("trace") or {}
        tfiles = tr.get("files") or []
        if tfiles:
            durs = sorted(f["t1"] - f["t0"] for f in tfiles)
            med = durs[len(durs) // 2] if durs else 0
            if med > 0.3:
                out.append(f"CPU/GPU 未饱和（均值 {c_avg:.0f}%/{g_avg:.0f}%）："
                           f"供给受大图解码长尾约束（单图窗中位 "
                           f"{med * 1000:.0f}ms），见 2.5 节空窗统计")
            else:
                out.append(f"解码供给正常（单图窗中位 {med * 1000:.0f}ms），"
                           f"CPU/GPU 均值 {c_avg:.0f}%/{g_avg:.0f}%："
                           "图集规模小或 IO 等待，增大规模再测")
        elif g_avg < 55 and c_avg < 40:
            out.append("建库期间 CPU/GPU 均空闲：图太小或 IO 等待，检查磁盘；"
                       "增大图集规模再测")
        out.append(f"建库平均 CPU {c_avg:.0f}% / GPU {g_avg:.0f}%"
                   f"（峰值 {max(gpu):.0f}%）")
    res = report["results"]
    if "lsh" in res and "coarse" in res:
        ml = min(res["lsh"]["lat"]) if res["lsh"]["lat"] else 0
        mc = min(res["coarse"]["lat"]) if res["coarse"]["lat"] else 0
        if mc > 0 and ml > 0:
            ratio = mc / ml
            if ratio > 3:
                out.append(f"LSH 相对全库扫描快 {ratio:.0f}×")
            elif ratio < 1.2:
                out.append("本库规模下 LSH 与线性扫描同速：库较小，可直用"
                           "线性(--cand coarse)简化，规模扩大后再启用 LSH")
        # 漏召判断：LSH 相对精确扫描的 Top5 缺口才是 LSH 漏召
        d5 = res["coarse"]["top5"] - res["lsh"]["top5"]
        if d5 > 0:
            out.append(f"LSH 相对精确扫描漏召 {d5} 个(Top5 缺口)：提高表数"
                       f"(--lsh-tables 12~16)、桶位数或 recheck-hits")
        miss = res["coarse"]["n_q"] - res["coarse"]["top5"]
        if miss > 0:
            if str(report.get("source", "")).startswith("real"):
                out.append(f"精确扫描 Top5 也缺失 {miss} 个：裁切到低判别区/"
                           f"构图近邻时 ResNet 会判相近（真实库命中已高于"
                           f"合成压力集），可调大 top_k 观察")
            else:
                out.append(f"精确扫描 Top5 也缺失 {miss} 个：属合成压力集近邻"
                           f"歧义（ResNet 层面判相近），非 LSH 引入；真实内容"
                           f"差异大时命中率会更高（几何图形 smoke 集为 100%）")
    # 解码/特征构成提示（真实模式）：单图处理窗偏大时给出分段建议
    tr = report.get("trace") or {}
    files = tr.get("files") or []
    if files and str(report.get("source", "")).startswith("real"):
        import statistics as _st
        durs = sorted(f["t1"] - f["t0"] for f in files)
        med = _st.median(durs)
        if med > 0.4:
            out.append(f"单图解码窗中位 {med * 1000:.0f}ms：大 PNG 全尺寸解码"
                       f"与瓦片特征(≈4ms/块)是供给瓶颈；JPEG 已走域缩放。"
                       f"下一步可做瓦片特征 cv2 直通 transform 或 PNG 抽样降档")
    return out


def _cmd_tiles_profile(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="局部(瓦片)索引性能画像："
                                             "合成图集或真实图库(只读抽样)"
                                             "建库+检索全链路")
    ap.add_argument("--work", default=None,
                    help="工作目录（默认系统临时目录；含生成的合成图与索引）")
    ap.add_argument("--root", default=None,
                    help="真实图库根目录（只读抽样；不写图库任何内容）")
    ap.add_argument("--root-big", type=int, default=40,
                    help="真实模式：抽取的大图(>1.5MB)张数")
    ap.add_argument("--root-small", type=int, default=20,
                    help="真实模式：抽取的小图(<=1.5MB)张数")
    ap.add_argument("--n-big", type=int, default=80, help="合成模式：大图张数")
    ap.add_argument("--n-small", type=int, default=80, help="合成模式：小图张数")
    ap.add_argument("--queries", type=int, default=20, help="裁切查询数")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--min-side", type=int, default=768)
    ap.add_argument("--lsh-bits", type=int, default=12)
    ap.add_argument("--lsh-tables", type=int, default=8)
    ap.add_argument("--cands", default="lsh,coarse",
                    help="候选法列表(逗号分隔)：lsh,coarse")
    ap.add_argument("--order", choices=["mixed", "layer"], default="mixed",
                    help="真实抽样建库顺序：mixed=大小图混合随机(默认)；"
                         "layer=大图在前小图在后(旧行为，用于对照尾段下降)")
    ap.add_argument("--out", default="perf_reports")
    ap.add_argument("--keep", action="store_true", help="保留工作目录不清理")
    a = ap.parse_args(argv)

    work = a.work or tempfile.mkdtemp(prefix="perfscope_tiles_")
    os.makedirs(a.out, exist_ok=True)
    try:
        kwargs = dict(seed=7, tile=a.tile, overlap=a.overlap,
                      min_side=a.min_side, lsh_bits=a.lsh_bits,
                      lsh_tables=a.lsh_tables, n_queries=a.queries,
                      order=a.order,
                      cand_methods=tuple(x.strip()
                                         for x in a.cands.split(",")))
        if a.root:
            root = os.path.abspath(a.root)
            if not os.path.isdir(root):
                print(f"目录不存在: {root}")
                return 2
            big, small = _sample_real_files(root, a.root_big, a.root_small,
                                            seed=7)
            if not big:
                print("真实模式未抽到大图（>1.5MB），请检查目录")
                return 2
            print(f"真实图库抽样：{root} -> 大图 {len(big)} 张、"
                  f"小图 {len(small)} 张（原图只读）", flush=True)
            report = run_tiles_bench(work, real_big=big, real_small=small,
                                     source=f"real:{root}", **kwargs)
        else:
            report = run_tiles_bench(
                work, n_big=a.n_big, n_small=a.n_small, **kwargs)
    except Exception as e:            # noqa: BLE001
        import traceback as _tb
        print(f"tiles-profile 失败：{e}\n{_tb.format_exc()}", flush=True)
        return 1
    report["ts"] = time.strftime("%Y%m%d-%H%M%S")
    report["work"] = work
    report["suggestions"] = _make_tiles_suggestions(report)
    ts = report["ts"]
    html_path = os.path.join(a.out, f"perf_tiles_{ts}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_tiles_html(report))

    print("\n========== 局部索引性能摘要 ==========", flush=True)
    print(f"原图 {report['n_images']} 张 -> 瓦片 {report['n_tiles']:,} 块 "
          f"({report['avg_tiles_per_img']:.1f} 块/图) "
          f"建库 {report['build_sec']:.1f}s "
          f"({report['n_tiles'] / max(report['build_sec'], 1e-6):.0f} 瓦片/s)",
          flush=True)
    for method, r in report["results"].items():
        lat = sorted(r["lat"])
        n = len(lat)
        print(f"  [{method}] Top1 {r['top1']}/{r['n_q']} · "
              f"Recall@5 {r['top5']}/{r['n_q']} · "
              f"单查中位 {lat[n // 2] * 1000:.1f}ms · "
              f"p95 {lat[min(n - 1, int(n * 0.95))] * 1000:.1f}ms",
              flush=True)
    print("\n调控建议：", flush=True)
    for s in report["suggestions"]:
        print(f"  - {s}", flush=True)
    print(f"\nHTML 图纸报告：{os.path.abspath(html_path)}", flush=True)
    if not a.keep and work != a.work:
        shutil.rmtree(work, ignore_errors=True)
    return 0


def _imread_unicode(path: str):
    """cv2.imread 不支持中文路径 -> np.fromfile + imdecode（BGR）。"""
    import cv2 as _cv2
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
    except Exception:            # noqa: BLE001
        return None


def _sample_real_files(root: str, n_big: int, n_small: int,
                       seed: int = 7) -> Tuple[List[str], List[str]]:
    """真实图库只读抽样：按文件大小分大(>1.5MB)/小(<=1.5MB)两层随机抽，
    只读 stat + 路径（不做任何解码/写入）。返回 (big, small) 路径列表。"""
    import random
    big_pool, small_pool = [], []
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in fnames:
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz > 1_500_000:
                big_pool.append(p)
            else:
                small_pool.append(p)
    rng = random.Random(seed)
    rng.shuffle(big_pool)
    rng.shuffle(small_pool)
    return big_pool[:n_big], small_pool[:n_small]


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "tiles-profile":
        return _cmd_tiles_profile(sys.argv[2:])
    ap = argparse.ArgumentParser(description="图库只读效能观测（不写图库）")
    ap.add_argument("root", help="图库根目录（只读）")
    ap.add_argument("--scan-only", action="store_true", help="只做档案扫描")
    ap.add_argument("--no-fused", action="store_true", help="跳过融合建库时间轴")
    ap.add_argument("--per-class", type=int, default=3, help="每类抽样张数")
    ap.add_argument("--max-sample", type=int, default=350, help="解码抽样总上限")
    ap.add_argument("--fused-max", type=int, default=60, help="融合建库样本上限")
    ap.add_argument("--out", default="perf_reports", help="报告输出目录")
    ap.add_argument("--rescan", action="store_true", help="忽略缓存强制重扫")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        print(f"目录不存在: {root}")
        return 2
    ts = time.strftime("%Y%m%d-%H%M%S")
    report = {"root": root, "ts": ts, "counter": {}, "profile": [],
              "fused_rows": [], "suggestions": []}

    # A. scan（带缓存复用）
    cache_path = _cache_path(a.out, root)
    cached = None if a.rescan else _load_cache(cache_path)
    if cached:
        items, counter, other_counter, vid_bytes = cached
        scan_sec = 0.0
        print(f"[1/4] 使用扫描缓存：{cache_path}", flush=True)
        print(f"      档案共 {len(items):,} 张可索引图片", flush=True)
    else:
        print(f"[1/4] 扫描档案：{root}（只读头部探测，不改动任何文件）…",
              flush=True)
        items, counter, other_counter, vid_bytes, scan_sec = scan_gallery(root)
        _save_cache(cache_path, items, counter, other_counter, vid_bytes)
        print(f"      图片头部探测完成：{counter.get('图片总张数', 0):,} 张，"
              f"耗时 {scan_sec:.1f}s，缓存 -> {cache_path}", flush=True)
    report["counter"] = counter
    report["ext_dist"] = other_counter
    report["scan_sec"] = scan_sec
    if a.scan_only:
        report["suggestions"] = make_suggestions(report)
        return write_out(a, report, html=True)
    if not items:
        report["suggestions"] = make_suggestions(report)
        return write_out(a, report, html=True)

    # B. profile
    print(f"[2/4] 分层抽样解码计时（每类≤{a.per_class} 张，共≤{a.max_sample} 张）…",
          flush=True)
    sample = stratified_sample(items, a.per_class, a.max_sample, a.seed)
    prof_rows = profile_samples(sample)
    report["profile"] = aggregate_profile(prof_rows)

    # C. fused 时间轴
    fused_rows: List[dict] = []
    if not a.no_fused:
        print(f"[3/4] 小样本融合建库（≤{a.fused_max} 张，索引写临时目录，"
              f"CPU/GPU 0.4s 采样）…", flush=True)
        try:
            fused_rows, n, sec = run_fused_bench(sample, a.fused_max)
            report["fused_rows"] = fused_rows
            report["fused_n"] = n
            report["fused_sec"] = sec
        except Exception as e:      # noqa: BLE001
            print(f"      融合建库画像跳过：{e}", flush=True)

    report["suggestions"] = make_suggestions(report)
    return write_out(a, report, html=True)


def write_out(a: argparse.Namespace, report: dict, html: bool = True) -> int:
    ts = report["ts"]
    html_path = os.path.join(a.out, f"perf_{ts}.html")
    if html:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(build_html(report))
    # 控制台摘要
    c = report["counter"]
    print("\n========== 摘要 ==========", flush=True)
    for k in ("图片总张数", "真实格式:JPEG", "真实格式:PNG", "真实格式:WEBP",
              "真实格式:GIF", "真实格式:TIFF", "真实格式:BMP",
              "探测失败/不支持的图片", "视频文件", "其他文件"):
        if k in c:
            print(f"{k}: {c[k]:,}", flush=True)
    exif = c.get("EXIF需转正(orient!=1)", 0)
    n_img = c.get("图片总张数", 0)
    print(f"EXIF需转正: {exif:,} ({exif / max(n_img, 1) * 100:.1f}%)", flush=True)
    if report["profile"]:
        print("\n解码效能 Top 慢（RGB ms/张 | 类 | 样本数）：", flush=True)
        for p in sorted(report["profile"], key=lambda x: -x["ms_rgb"])[:6]:
            print(f"  {p['ms_rgb']:7.0f}  {p['format']} {p['band']} "
                  f"{p['orient']}  n={p['n']}", flush=True)
    if report["fused_rows"]:
        rows = report["fused_rows"]
        cpu = [r.get("cpu_sys") for r in rows if r.get("cpu_sys") is not None]
        gpu = [r.get("gpu") for r in rows if r.get("gpu") is not None]
        import statistics
        if cpu and gpu:
            print(f"\n融合建库 {report.get('fused_n', 0)} 张 "
                  f"{report.get('fused_sec', 0):.1f}s | 平均 CPU {statistics.mean(cpu):.0f}% "
                  f"| 平均 GPU(SM) {statistics.mean(gpu):.0f}% | "
                  f"峰值 GPU {max(gpu):.0f}%", flush=True)
    print(f"\nHTML 图纸报告：{os.path.abspath(html_path)}", flush=True)
    print("（图库未被写入/修改任何内容）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
