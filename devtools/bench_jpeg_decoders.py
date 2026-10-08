# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — C2 · JPEG 解码器横向对照（靶子图库，只读）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""C2：cv2（现状 libjpeg-turbo）vs PyTurboJPEG vs imagecodecs(含 mozjpeg) 的 JPEG 解码对照。

口径（与 PNG 那轮同形态，便于横向比较）：
  * 分层抽样：按 io_utils._reduced_flag 的**缩放档**分层（全解 / 1/2 / 1/4 / 1/8）。
    普查（probe_jpeg_census.py）显示：50.1% 张数、90.8% 像素落在三个缩放档上，
    所以不支持 DCT 域缩放的库（imagecodecs）会**结构性**吃亏，必须分档看，
    只给一个总数会把"大图上被拉开 4~8 倍像素量"的差距平均掉；
  * **配对 A/B**：每张图在同一轮内跑全部解码器，顺序按 (图号+轮号) 轮转，消时钟漂移；
    先算每张图的逐张比值，再取中位数（配对统计，比整轮总时长抗离群）；
  * **逐位校验**：一律与"当前部署路径"（io_utils.decode_rgb）的输出比对，形状不一致单独计数；
  * 逐张处理、不缓存参考数组（1/4 档有大到 100MP 的图，全缓存会吃光内存）；
  * 只读：只读图片、不写索引、不删文件。

libjpeg-turbo 3.2.0 的 DLL 由 devtools/fetch_turbojpeg.py 从 conda-forge 取
（纯 Python 解包，不跑安装器、不动注册表/PATH）。

用法: python -E devtools/bench_jpeg_decoders.py [每档张数=15] [轮数=3]
"""
import json
import os
import random
import statistics as st
import struct
import sys
import time
from collections import defaultdict

import cv2
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
PER_BUCKET = int(sys.argv[1]) if len(sys.argv) > 1 else 15
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DLL = os.environ.get("TURBOJPEG_DLL") or os.path.join(
    os.environ.get("TEMP", ""), "c2_jpeg", "turbojpeg.dll")

# 缩放档定义必须与 io_utils._reduced_flag / probe_jpeg_census 同步
_TARGET = 2048
SCALES = [("全解<=2560", _TARGET * 1.25), ("1/2<=5120", _TARGET * 2.5),
          ("1/4<=10240", _TARGET * 5.0), ("1/8>10240", float("inf"))]
# 1/8 档全库只有几十张，全取；其余按 PER_BUCKET 随机抽
_PICK = {SCALES[0][0]: PER_BUCKET, SCALES[1][0]: PER_BUCKET,
         SCALES[2][0]: max(4, int(PER_BUCKET * 0.8)), SCALES[3][0]: 4}


def bucket_of(w, h):
    ms = max(w, h)
    for name, lim in SCALES:
        if ms <= lim:
            return name
    return SCALES[-1][0]


def scale_of(w, h):
    """与 bucket_of 同口径，返回 TurboJPEG 的 (num, denom) 缩放因子。"""
    return {"全解<=2560": (1, 1), "1/2<=5120": (1, 2),
            "1/4<=10240": (1, 4), "1/8>10240": (1, 8)}[bucket_of(w, h)]


def jpeg_dims(head: bytes):
    """极简 SOF 扫描（只用于抽样分档，不解码像素）。"""
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


# ---------------------------------------------------------------- 各解码器
def _to_rgb8(a):
    if a is None:
        return None
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    elif a.ndim == 3 and a.shape[2] == 2:
        a = np.repeat(a[:, :, :1], 3, axis=2)
    return np.ascontiguousarray(a)


_TJ = {"tl": None}


def tj_obj():
    """TurboJPEG 实例不可跨线程共享 → 每线程一个（建库是 18 路线程池）。"""
    import threading
    if _TJ["tl"] is None:
        _TJ["tl"] = threading.local()
    if getattr(_TJ["tl"], "obj", None) is None:
        from turbojpeg import TJPF_RGB, TurboJPEG
        _TJ["tl"].obj = TurboJPEG(DLL)
        _TJ["tl"].rgb = TJPF_RGB
    return _TJ["tl"].obj, _TJ["tl"].rgb


def make_decoders(iu):
    """返回 [(名称, 函数(data, probe)->ndarray|None), ...]。"""
    decs = []

    def d_cv2_cur(data, probe):
        return iu.decode_rgb(data)                 # 当前部署路径（内部自带 Pillow probe）

    def d_cv2_same(data, probe):
        """现状的解码段：复用外部 probe，按同一缩放档 imdecode（苹果对苹果）。"""
        w, h = probe[1]
        flag, swap = iu._flag_maybe_rgb(iu._reduced_flag(w, h, False), False)
        a = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        if a is None:
            return None
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB) if swap else a

    def d_cv2_full(data, probe):
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR_RGB)

    def d_tj_reuse(data, probe):
        tj, TJPF_RGB = tj_obj()
        return _to_rgb8(tj.decode(data, pixel_format=TJPF_RGB,
                                  scaling_factor=scale_of(*probe[1])))

    def d_tj_header(data, probe):
        tj, TJPF_RGB = tj_obj()
        w, h = tj.decode_header(data)[:2]
        return _to_rgb8(tj.decode(data, pixel_format=TJPF_RGB, scaling_factor=scale_of(w, h)))

    def d_tj_full(data, probe):
        tj, TJPF_RGB = tj_obj()
        return _to_rgb8(tj.decode(data, pixel_format=TJPF_RGB))

    def d_ic_jpeg(data, probe):
        import imagecodecs
        return _to_rgb8(imagecodecs.jpeg_decode(data, outcolorspace="RGB"))

    def d_ic_moz(data, probe):
        import imagecodecs
        return _to_rgb8(imagecodecs.mozjpeg_decode(data, outcolorspace="RGB"))

    decs += [("cv2 现状 decode_rgb(全路径)", d_cv2_cur),
             ("cv2 仅 imdecode(同缩放档)", d_cv2_same),
             ("cv2 全尺寸(不缩放)", d_cv2_full)]
    if os.path.exists(DLL):
        try:
            tj_obj()
            decs += [("TurboJPEG 3.2.0 同档(复用probe)", d_tj_reuse),
                     ("TurboJPEG 3.2.0 同档(自读头)", d_tj_header),
                     ("TurboJPEG 3.2.0 全尺寸", d_tj_full)]
        except Exception as e:                            # noqa: BLE001
            print("  [warn] PyTurboJPEG 不可用:", e)
    else:
        print("  [warn] 缺 turbojpeg.dll（%s），跳过 TurboJPEG 组" % DLL)
    try:
        import imagecodecs
        print("  imagecodecs %s / libjpeg-turbo %s / mozjpeg %s"
              % (imagecodecs.__version__, imagecodecs.jpeg_version(),
                 imagecodecs.mozjpeg_version()))
        decs += [("imagecodecs jpeg 3.2.0 全尺寸", d_ic_jpeg),
                 ("imagecodecs mozjpeg 全尺寸", d_ic_moz)]
    except Exception as e:                                # noqa: BLE001
        print("  [warn] imagecodecs 不可用:", e)
    return decs


def probe_cost(iu, blobs, reps=3):
    """单列：Pillow probe / 纯 SOF 扫描 / TurboJPEG decode_header 的单张开销。"""
    out = {}
    for tag, fn in (("io_utils._probe (Pillow)", iu._probe),
                    ("纯 SOF 头扫描 (Python)", jpeg_dims)):
        t0 = time.perf_counter()
        for _ in range(reps):
            for b in blobs:
                fn(b)
        out[tag] = (time.perf_counter() - t0) / (reps * len(blobs)) * 1e3
    if os.path.exists(DLL):
        try:
            import turbojpeg
            tj = turbojpeg.TurboJPEG(DLL)
            t0 = time.perf_counter()
            for _ in range(reps):
                for b in blobs:
                    tj.decode_header(b)
            out["TurboJPEG decode_header"] = (time.perf_counter() - t0) / (reps * len(blobs)) * 1e3
        except Exception as e:                            # noqa: BLE001
            print("  [warn] decode_header 计时失败:", e)
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    from hybrid_search import io_utils as iu

    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                     allow_pickle=True)]
    jpgs = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    random.seed(11)
    random.shuffle(jpgs)

    # ---- 全库头扫描分档（32KB/张），再按档抽样
    print("扫描 %d 张 JPEG 头部分档（只读前 32KB）…" % len(jpgs))
    t0 = time.perf_counter()
    by_bucket = defaultdict(list)
    for p in jpgs:
        try:
            with open(p, "rb") as f:
                d = jpeg_dims(f.read(32 << 10))
        except OSError:
            continue
        if d:
            by_bucket[bucket_of(*d)].append(p)
    print("  %.1f s；全库分档 %s"
          % (time.perf_counter() - t0, {k: len(v) for k, v in by_bucket.items() if v}))

    items, picked = [], {}
    for name, _ in SCALES:
        pool = by_bucket.get(name, [])
        if not pool:
            picked[name] = 0
            continue
        take = pool if len(pool) <= _PICK[name] else random.sample(pool, _PICK[name])
        picked[name] = len(take)
        for p in take:
            try:
                with open(p, "rb") as f:
                    b = f.read()
            except OSError:
                continue
            d = jpeg_dims(b[:32 << 10])
            if d:
                items.append({"p": p, "b": b, "w": d[0], "h": d[1], "bucket": name})
    print("样本 %d 张；分档 %s（%d 轮配对）" % (len(items), picked, ROUNDS))
    print("合计 %.1f MB" % (sum(len(it["b"]) for it in items) / 2 ** 20))

    decs = make_decoders(iu)
    names = [n for n, _ in decs]
    base = names[0]

    # ---- 逐位校验（逐张处理、不缓存参考数组）
    print("\n=== 逐位校验（基准 = %s）" % base)
    exact = {n: {"same": 0, "diff": 0, "err": 0, "shape": 0} for n in names}
    for it in items:
        b = it["b"]
        try:
            probe = iu._probe(b)
            r = iu.decode_rgb(b)
        except Exception:                                 # noqa: BLE001
            continue
        if r is None:
            continue
        for name, fn in decs:
            try:
                a = fn(b, probe)
            except Exception:                             # noqa: BLE001
                exact[name]["err"] += 1
                continue
            if a is None:
                exact[name]["err"] += 1
            elif a.shape != r.shape:
                exact[name]["shape"] += 1
            elif np.array_equal(a, r):
                exact[name]["same"] += 1
            else:
                exact[name]["diff"] += 1
            del a
        del r
        it["probe"] = probe
    nvalid = sum(exact[base].values())
    for name in names:
        e = exact[name]
        print("   %-30s 逐位一致 %3d/%d  像素不一致 %d  形状不同 %d  失败 %d"
              % (name, e["same"], nvalid, e["diff"], e["shape"], e["err"]))

    # ---- 配对计时
    print("\n=== 配对计时（%d 轮 × %d 张，顺序按 图号+轮号 轮转）" % (ROUNDS, len(items)))
    per_img = defaultdict(dict)
    acc = defaultdict(float)
    nok = defaultdict(int)
    t_all = time.perf_counter()
    for rnd in range(ROUNDS):
        for k, it in enumerate(items):
            b, probe = it["b"], it.get("probe") or iu._probe(it["b"])
            order = decs if (k + rnd) % 2 == 0 else list(reversed(decs))
            for name, fn in order:
                t0 = time.perf_counter()
                try:
                    fn(b, probe)
                except Exception:                         # noqa: BLE001
                    continue
                dt = time.perf_counter() - t0
                acc[name] += dt
                nok[name] += 1
                per_img[name].setdefault(it["p"], []).append(dt * 1e3)
        print("   轮 %d/%d 完成（累计 %.1f s）" % (rnd + 1, ROUNDS, time.perf_counter() - t_all))

    # ---- 汇总
    paths_ok = [it["p"] for it in items
                if all(it["p"] in per_img[n] for n in names)]
    med = {n: {p: st.median(v) for p, v in per_img[n].items()} for n in names}
    rows = []
    for name in names:
        if not nok[name]:
            continue
        paired = [med[base][p] / med[name][p] for p in paths_ok if med[name][p] > 0]
        rows.append({"name": name, "ms": round(acc[name] / nok[name] * 1e3, 2),
                     "paired_median": round(st.median(paired), 3) if paired else 0.0,
                     "paired_mean": round(st.mean(paired), 3) if paired else 0.0,
                     "total_ratio": round(acc[base] / acc[name], 3),
                     "n": nok[name], "exact": exact.get(name, {})})

    print("\n=== 汇总（配对中位比 = 每张图 基准ms ÷ 该库ms；>1 = 比现状快）")
    print("   %-30s %9s %10s %10s" % ("解码器", "ms/张", "配对中位比", "总时长比"))
    for r in rows:
        print("   %-30s %9.2f %9.3fx %9.3fx" % (r["name"], r["ms"], r["paired_median"],
                                                r["total_ratio"]))

    # ---- 分档
    print("\n=== 分档 ms/张（配对中位数）")
    bucket_rows = {}
    print("   %-12s %5s " % ("档", "n") + "".join("%14s" % n[:13] for n in names))
    for name, _ in SCALES:
        ps = [it["p"] for it in items if it["bucket"] == name and it["p"] in paths_ok]
        if not ps:
            continue
        bucket_rows[name] = {"n": len(ps)}
        line = "   %-12s %5d " % (name, len(ps))
        for n in names:
            v = [med[n][p] for p in ps]
            bucket_rows[name][n] = round(st.median(v), 2)
            line += "%14.2f" % st.median(v)
        print(line)

    # ---- probe 开销
    pc = probe_cost(iu, [it["b"] for it in items])
    print("\n=== 头部读取开销（单张 ms，纯 CPU、不解码）")
    for k, v in pc.items():
        print("   %-28s %7.3f ms" % (k, v))

    meta = {"tool": "bench-jpeg-decoders", "ts": time.strftime("%Y%m%d-%H%M%S"),
            "files": len(items), "rounds": ROUNDS, "buckets_sampled": picked,
            "rows": rows, "by_bucket": bucket_rows, "probe_cost_ms": pc,
            "total_mb": round(sum(len(it["b"]) for it in items) / 2 ** 20, 1),
            "dll": DLL, "dll_exists": os.path.exists(DLL),
            "env": {"cv2": cv2.__version__}}
    for mod, key in ((("imagecodecs"), "imagecodecs"), (("turbojpeg"), "PyTurboJPEG")):
        try:
            meta["env"][key] = __import__(mod).__version__
        except Exception:                                 # noqa: BLE001
            pass
    try:
        import imagecodecs
        meta["env"]["libjpeg_turbo_imagecodecs"] = imagecodecs.jpeg_version()
        meta["env"]["mozjpeg"] = imagecodecs.mozjpeg_version()
    except Exception:                                     # noqa: BLE001
        pass
    jf = os.path.join(OUT_DIR, "jpeg_decoders_%s.json" % meta["ts"])
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    hf = os.path.join(OUT_DIR, "jpeg_decoders_%s.html" % meta["ts"])
    write_html(hf, meta)
    print("\nJSON:", jf)
    print("HTML:", hf)
    return 0


def write_html(path, m):
    rows, names = m["rows"], [r["name"] for r in m["rows"]]
    base = rows[0]
    mx = max([r["ms"] for r in rows] + [1e-9])
    bars, tbl = [], []
    for i, r in enumerate(rows):
        w = 400.0 * r["ms"] / mx
        col = "#7fdb9a" if r["paired_median"] >= 1.05 else (
            "#ffd479" if r["paired_median"] >= 0.98 else "#ff9b8a")
        bars.append('<text x="10" y="%d" fill="#d7dee4" font-size="12">%s</text>'
                    '<rect x="320" y="%d" width="%.1f" height="16" fill="%s" opacity=".85"/>'
                    '<text x="%d" y="%d" fill="#9fd0ff" font-size="12">%.2f ms · 配对 %.3fx</text>'
                    % (40 + i * 30, r["name"], 26 + i * 30, w, col, 326 + w, 39 + i * 30,
                       r["ms"], r["paired_median"]))
    for r in rows:
        e = r["exact"]
        tbl.append("<tr><td>%s</td><td class='num'>%.2f</td><td class='num'>%.3f</td>"
                   "<td class='num'>%.3f</td><td class='num'>%d/%d</td>"
                   "<td class='num'>%d</td><td class='num'>%d</td></tr>"
                   % (r["name"], r["ms"], r["paired_median"], r["total_ratio"],
                      e.get("same", 0), m["files"], e.get("diff", 0), e.get("shape", 0)))
    bk = [k for k, _ in SCALES if k in m["by_bucket"]]
    btbl = "".join("<tr><td>%s</td><td class='num'>%d</td>%s</tr>"
                   % (k, m["by_bucket"][k]["n"],
                      "".join("<td class='num'>%.2f</td>" % m["by_bucket"][k].get(n, 0)
                              for n in names)) for k in bk)
    cp = "".join("<tr><td>%s</td><td class='num'>%.3f</td></tr>" % (k, v)
                 for k, v in m["probe_cost_ms"].items())
    h = 70 + 30 * len(rows) + 20
    best = min(rows, key=lambda r: r["ms"])
    verdict = [
        "基准（当前部署路径）<b>%.2f ms/张</b>；样本内最快 <b>%s</b>（%.2f ms/张，配对中位比 <b>%.3fx</b>）"
        % (base["ms"], best["name"], best["ms"], best["paired_median"]),
        "判据（用户口径）：<b>&lt;5%% 即不采纳</b>。JPEG 单张便宜、96.3%% 非渐进，"
        "预计换库只剩个位数百分比",
        "environment：%s" % json.dumps(m["env"], ensure_ascii=False),
    ]
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>JPEG 解码器对照 性能图（C2）</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;margin-top:24px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%%;margin:8px 0;font-size:13px}
th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
th{background:#1a222b}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96;font-size:13px}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}</style></head><body>
<h1>JPEG 解码器对照（C2）· cv2 vs PyTurboJPEG vs imagecodecs / mozjpeg</h1>
<p class="muted">样本 %d 张 / %.1f MB（按 DCT 缩放档分层：%s），%d 轮配对、顺序轮转。
候选取与现状相同的缩放档；imagecodecs / mozjpeg 不支持缩放解码，只能列全尺寸作**结构性**对照。</p>
<h2>1. 单张耗时（越短越好；绿=快于现状≥5%%，黄=持平，红=更慢）</h2>
<svg viewBox="0 0 960 %d" height="%d">%s</svg>
<h2>2. 汇总与逐位校验</h2>
<table><tr><th>解码器</th><th>ms/张</th><th>配对中位比</th><th>总时长比</th>
<th>逐位一致</th><th>像素不一致</th><th>形状不同</th></tr>%s</table>
<h2>3. 分档 ms/张</h2>
<table><tr><th>缩放档</th><th>n</th>%s</tr>%s</table>
<h2>4. 头部读取开销（换库可能顺带省掉 Pillow probe）</h2>
<table><tr><th>方式</th><th>ms/张</th></tr>%s</table>
<h2>5. 判读</h2><ul>%s</ul>
<p class="muted">只读测试。生成器：<code>devtools/bench_jpeg_decoders.py</code>；
普查：<code>devtools/probe_jpeg_census.py</code>；DLL 获取：<code>devtools/fetch_turbojpeg.py</code></p>
</body></html>""" % (
        m["files"], m["total_mb"], json.dumps(m["buckets_sampled"], ensure_ascii=False),
        m["rounds"], h, h, "".join(bars), "".join(tbl),
        "".join("<th>%s</th>" % n for n in names), btbl, cp,
        "".join("<li>%s</li>" % v for v in verdict))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    raise SystemExit(main())
