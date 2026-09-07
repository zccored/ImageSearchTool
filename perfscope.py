# -*- coding: utf-8 -*-
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
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

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
    图例标注各自峰值；用于把 %（CPU/GPU）与 img/s（不同量纲）画在一起。"""
    if not series:
        return "<p>（无采样数据）</p>"
    xs = [r["t"] for r in series]
    total_t = max(xs[-1], 1e-6)
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

    def t_of(sec):
        return pad_l + inner_w * (sec / total_t)

    legend_x = pad_l + 110
    for key, color, name in keys:
        vals = [r.get(key) for r in series if r.get(key) is not None]
        if len(vals) < 2:
            continue
        peak = max(max(float(v) for v in vals), 1e-6)
        pts = []
        for r in series:
            v = r.get(key)
            if v is None:
                continue
            y = pad_t + inner_h * (1 - min(max(float(v) / peak, 0), 1.0))
            pts.append(f"{t_of(r['t']):.1f},{y:.1f}")
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


def main() -> int:
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
