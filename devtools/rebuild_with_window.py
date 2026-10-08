# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 静默重建 + 存活窗口 + 性能图
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""把建库交给后台子进程跑，前台只留一个**小窗表示进程存活**，并把采样写成性能图。

用途：长时间重建整图 / 瓦片索引时，既不想盯着满屏日志，又要能一眼确认"它还活着"、
"进度到哪了"、"CPU/GPU/内存是什么样"。结束时产出 `perf_reports/rebuild_*.html`
（整页性能图）与同名 `.json`（逐 0.4s 采样），供后续调优做参考。

设计要点：
  * 建库本体仍是**原样的 CLI**（`python -E main.py build / build-tiles`），
    本脚本只在外面套一层"启动 + 采样 + 画图"，不改任何建库逻辑；
  * 进度从 CLI 的进度输出里解析（CLI 用 `\\r` 原地刷新，读日志尾部取最后一段即可），
    因此**不要**给子进程加 `--no-progress`；
  * 窗口置顶、可隐藏；任务结束后窗口**保留一段**（默认 300 s，`LINGER` 环境变量可改），
    便于回来看结果；
  * 两个模式可以串跑（`--mode both`）后续接。

用法:
  python -E devtools/rebuild_with_window.py --mode whole  --gallery "<图库根>" --prefix "<图库根>\\.gallery_index\\gallery"
  python -E devtools/rebuild_with_window.py --mode tiles  --gallery "<图库根>" --prefix "<图库根>\\.gallery_index\\gallery_tiles"
  python -E devtools/rebuild_with_window.py --mode both   --gallery "<图库根>" --prefix "<图库根>\\.gallery_index\\gallery"
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT                                  # noqa: E402

OUT_DIR = os.path.join(_HERE, "perf_reports")
SAMPLE_S = 0.4
LINGER_S = float(os.environ.get("REBUILD_LINGER", "300"))
PROG_RE = re.compile(r"([\d,]+)\s*/\s*([\d,]+)\s*张")

STATE = {"stage": "准备中", "detail": "", "done": 0, "total": 0, "rate": 0.0,
         "elapsed": 0.0, "rss_mb": 0.0, "cpu": 0.0, "gpu": None, "peak_mb": 0.0,
         "finished": False, "result": "", "read_mb": 0.0, "write_mb": 0.0}


class Sampler(threading.Thread):
    """0.4s 一次抓 进程 CPU / RSS / 磁盘 IO / GPU 利用率。"""

    def __init__(self, pid, log_path):
        super().__init__(daemon=True)
        self.pid = pid
        self.log_path = log_path
        self.stop_flag = threading.Event()
        self.rows = []
        self.t0 = time.perf_counter()
        self.nv = None
        self.proc = None
        try:
            import psutil
            self.proc = psutil.Process(pid)
            self.proc.cpu_percent(None)                 # 预热：第一次调用恒为 0
        except Exception:                               # noqa: BLE001
            self.proc = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:                               # noqa: BLE001
            self.nv = None

    def _tail_progress(self):
        try:
            size = os.path.getsize(self.log_path)
            with open(self.log_path, "rb") as f:
                f.seek(max(0, size - 8192))
                blob = f.read().decode("utf-8", "replace")
            seg = re.split(r"[\r\n]", blob)
            best = None
            for line in seg:
                m = PROG_RE.search(line)
                if m:
                    best = (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")),
                            line.strip())
            return best
        except Exception:                               # noqa: BLE001
            return None

    def run(self):
        while not self.stop_flag.is_set():
            now = time.perf_counter()
            row = {"t": round(now - self.t0, 2)}
            if self.proc is not None:
                try:
                    with self.proc.oneshot():
                        row["cpu"] = round(self.proc.cpu_percent(None), 1)
                        mi = self.proc.memory_info()
                        row["rss_mb"] = round(mi.rss / 2 ** 20, 0)
                        io = self.proc.io_counters()
                        row["read_mb"] = round(io.read_bytes / 2 ** 20, 1)
                        row["write_mb"] = round(io.write_bytes / 2 ** 20, 1)
                    STATE["cpu"] = row["cpu"]
                    STATE["rss_mb"] = row["rss_mb"]
                    STATE["peak_mb"] = max(STATE["peak_mb"], row["rss_mb"])
                    STATE["read_mb"] = row["read_mb"]
                    STATE["write_mb"] = row["write_mb"]
                except Exception:                       # noqa: BLE001
                    pass
            if self.nv is not None:
                try:
                    row["gpu"] = self.nv[0].nvmlDeviceGetUtilizationRates(self.nv[1]).gpu
                    STATE["gpu"] = row["gpu"]
                except Exception:                       # noqa: BLE001
                    pass
            pr = self._tail_progress()
            if pr:
                STATE["done"], STATE["total"] = pr[0], pr[1]
                STATE["elapsed"] = round(now - self.t0, 1)
                if STATE["done"] and STATE["elapsed"]:
                    STATE["rate"] = round(STATE["done"] / STATE["elapsed"], 1)
                row["done"] = pr[0]
            self.rows.append(row)
            time.sleep(SAMPLE_S)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=3)


def run_one(mode, gallery, prefix, extra):
    cmd = [sys.executable, "-E", "-u", os.path.join(_HERE, "main.py"), mode, gallery,
           "--prefix", prefix] + list(extra)
    log = os.path.join(OUT_DIR, "rebuild_%s_%s.log" % (mode, time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(OUT_DIR, exist_ok=True)
    STATE.update({"stage": ("整图建库" if mode == "build" else "瓦片建库"),
                  "detail": prefix, "done": 0, "total": 0, "rate": 0.0, "elapsed": 0.0,
                  "finished": False})
    t0 = time.perf_counter()
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, cwd=_HERE, stdout=fh, stderr=subprocess.STDOUT)
        smp = Sampler(proc.pid, log)
        smp.start()
        rc = proc.wait()
    wall = time.perf_counter() - t0
    smp.stop()
    STATE["finished"] = True
    STATE["result"] = "完成（rc=%d，%.1f s）" % (rc, wall)
    return {"mode": mode, "rc": rc, "wall_s": round(wall, 2), "log": log,
            "cmd": " ".join(cmd), "samples": smp.rows,
            "peak_rss_mb": STATE["peak_mb"], "read_mb": STATE["read_mb"],
            "write_mb": STATE["write_mb"]}


def write_report(res):
    ts = time.strftime("%Y%m%d-%H%M%S")
    cpu = [r.get("cpu", 0) for r in res["samples"]]
    gpu = [r.get("gpu") for r in res["samples"] if r.get("gpu") is not None]
    rss = [r.get("rss_mb", 0) for r in res["samples"]]
    summary = {
        "mode": res["mode"], "rc": res["rc"], "wall_s": res["wall_s"],
        "cpu_mean": round(sum(cpu) / max(len(cpu), 1), 1),
        "cpu_peak": round(max(cpu) if cpu else 0, 1),
        "gpu_mean": round(sum(gpu) / max(len(gpu), 1), 1) if gpu else None,
        "gpu_peak": max(gpu) if gpu else None,
        "rss_peak_mb": res["peak_rss_mb"], "read_mb": res["read_mb"],
        "write_mb": res["write_mb"], "samples": len(res["samples"]),
        "cmd": res["cmd"], "log": res["log"],
    }
    jf = os.path.join(OUT_DIR, "rebuild_%s_%s.json" % (res["mode"], ts))
    with open(jf, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": res["samples"]}, f, ensure_ascii=False, indent=2)

    def poly(vals, color, scale=1.0, y0=60, h=110):
        if not vals:
            return ""
        mx = max(vals) or 1
        n = len(vals)
        pts = " ".join("%.1f,%.1f" % (10 + 900.0 * i / max(n - 1, 1),
                                      y0 + h - h * (v / mx) * scale)
                       for i, v in enumerate(vals))
        return ('<polyline fill="none" stroke="%s" stroke-width="1.6" points="%s"/>'
                '<text x="914" y="%d" fill="%s" font-size="11" text-anchor="end">峰值 %.0f</text>'
                % (color, pts, y0 + 12, color, mx))

    svg = ('<svg viewBox="0 0 960 320">'
           + poly(cpu, "#7fdb9a", y0=20)
           + poly(rss, "#9fd0ff", y0=160)
           + poly(gpu or [], "#ffd479", y0=160)
           + '<text x="10" y="14" fill="#d7dee4" font-size="12">CPU%（绿，上）/ RSS MB（蓝，下）/ GPU%（黄，下）</text>'
           + '</svg>')
    rows = "".join("<tr><td>%s</td><td class='num'>%s</td></tr>" % (k, v)
                   for k, v in summary.items())
    html = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>建库性能图 · %s</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;padding:22px;line-height:1.6}
table{border-collapse:collapse;width:100%%;font-size:13px}th,td{border:1px solid #26323d;padding:5px 8px;text-align:left}
td.num{text-align:right;font-variant-numeric:tabular-nums}svg{background:#121820;border:1px solid #26323d;width:100%%}
code{background:#1a222b;padding:1px 5px;border-radius:3px}</style></head><body>
<h1>建库性能图 · %s</h1>
<p><code>%s</code></p>
%s
<table><tr><th>项</th><th>值</th></tr>%s</table>
</body></html>""" % (res["mode"], res["mode"], summary["cmd"], svg, rows)
    hf = os.path.join(OUT_DIR, "rebuild_%s_%s.html" % (res["mode"], ts))
    with open(hf, "w", encoding="utf-8") as f:
        f.write(html)
    return jf, hf, summary


def start_window(mode_list):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title("建库进行中（进程存活）")
    root.attributes("-topmost", True)
    root.geometry("+%d+%d" % (max(root.winfo_screenwidth() - 520, 0), 70))
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    head = ttk.Label(frm, text="准备中…", font=("Microsoft YaHei UI", 11, "bold"))
    head.pack(anchor="w")
    sub = ttk.Label(frm, text="；".join(mode_list), font=("Consolas", 9),
                    wraplength=470, justify="left")
    sub.pack(anchor="w", pady=(4, 2))
    bar = ttk.Progressbar(frm, length=470, mode="determinate", maximum=100)
    bar.pack(fill="x", pady=4)
    foot = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=470, justify="left")
    foot.pack(anchor="w")
    ttk.Button(frm, text="隐藏窗口（建库不受影响）",
               command=root.withdraw).pack(anchor="e", pady=(8, 0))
    root.update()

    def tick():
        try:
            st = STATE
            head.config(text="%s%s" % (st["stage"], " · " + st["result"] if st["result"] else "（进行中）"))
            bar.config(value=100.0 * st["done"] / max(st["total"], 1))
            eta = ("%.0f s" % ((st["total"] - st["done"]) / st["rate"]) if st["rate"] else "--")
            foot.config(text="进度 %d/%d | %.1f 张/秒 | 已用 %.0f s | ETA %s\n"
                             "CPU %.0f%% | GPU %s | RSS %.0f MB（峰值 %.0f）| 读 %.0f MB / 写 %.0f MB"
                             % (st["done"], st["total"], st["rate"], st["elapsed"], eta,
                                st["cpu"], ("%d%%" % st["gpu"]) if st["gpu"] is not None else "n/a",
                                st["rss_mb"], st["peak_mb"], st["read_mb"], st["write_mb"]))
            root.update()
        except Exception:                               # noqa: BLE001
            pass
    return root, tick


def main() -> int:
    ap = argparse.ArgumentParser(description="静默重建 + 存活窗口 + 性能图")
    ap.add_argument("--mode", choices=["whole", "tiles", "both"], default="both")
    ap.add_argument("--gallery", default=GALLERY_ROOT)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--extra", nargs="*", default=[], help="透传给 main.py 的参数")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的索引（索引已在时重建必须加，等价于给 main.py 传 --force）")
    a = ap.parse_args()
    extra = list(a.extra) + (["--force"] if a.force else [])
    idx_dir = os.path.join(a.gallery, ".gallery_index")
    jobs = []
    if a.mode in ("whole", "both"):
        jobs.append(("build", a.prefix or os.path.join(idx_dir, "gallery")))
    if a.mode in ("tiles", "both"):
        jobs.append(("build-tiles", os.path.join(idx_dir, "gallery_tiles")))

    print("图库 : %s" % a.gallery)
    for m, p in jobs:
        print("  %-12s -> %s" % (m, p))
    root, tick = start_window(["%s -> %s" % (m, p) for m, p in jobs])

    results = []
    done_flag = {"v": False, "ok": False}

    def worker():
        """建库放在后台线程，主线程留给 tkinter 刷新 —— 否则窗口会卡成"无响应"。"""
        for mode, prefix in jobs:
            print("\n===== %s 开始（%s）=====" % (mode, time.strftime("%H:%M:%S")), flush=True)
            STATE.update({"stage": ("整图建库" if mode == "build" else "瓦片建库"),
                          "result": "", "finished": False})
            res = run_one(mode, a.gallery, prefix, extra)
            jf, hf, sm = write_report(res)
            res["report"] = hf
            results.append(res)
            print("  rc=%d 墙钟 %.1f s | CPU 均值 %.1f%% | GPU 均值 %s | RSS 峰值 %.0f MB"
                  % (res["rc"], res["wall_s"], sm["cpu_mean"], sm["gpu_mean"], sm["rss_peak_mb"]),
                  flush=True)
            print("  性能图: %s" % hf, flush=True)
            print("  采样   : %s" % jf, flush=True)
        ok = all(r["rc"] == 0 for r in results)
        STATE["stage"] = "全部完成"
        STATE["result"] = "✓ 成功" if ok else "✗ 有失败"
        STATE["done"] = STATE["total"] = 1
        done_flag["v"], done_flag["ok"] = True, ok
        print("\n===== 全部完成，窗口保留 %.0f s 便于查看 =====" % LINGER_S, flush=True)

    threading.Thread(target=worker, daemon=True, name="rebuild-runner").start()
    linger_until = [None]

    def pump():
        tick()
        if done_flag["v"] and linger_until[0] is None:
            linger_until[0] = time.time() + LINGER_S
        if linger_until[0] is not None and time.time() > linger_until[0]:
            try:
                root.destroy()
            except Exception:                           # noqa: BLE001
                pass
            return
        root.after(200, pump)

    root.after(200, pump)
    root.mainloop()
    return 0 if done_flag["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
