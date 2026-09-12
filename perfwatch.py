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

"""perfwatch —— GUI 内置的阶段性能画像（建库/索引 与 搜图）。

用途：在 GUI 里勾选「导出性能图」后，对应任务在后台被采样，任务结束即产出
一份 HTML（+ 原始采样 JSON）到 perf_reports/，用于回答：
  * 这段耗时到底花在哪（阶段打点）；
  * CPU/GPU/内存/磁盘IO 时间轴（谁在等谁、是否真并行）；
  * 进程 RSS 是否在任务结束后回落（排查内存泄漏）。

开销：采样线程每 0.4s 读一次 psutil/nvml（微秒级），HTML 只在结束时写一次；
对建库吞吐的影响 <1%（远小于一个瓦片批次的抖动）。但开启后报告会占磁盘，
且长时间建库的 JSON 采样行会随任务时长线性增长（0.4s 一行，1 小时约 9000 行）。
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(_HERE, "perf_reports")

_STYLE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>阶段性能画像</title><style>
body{font-family:'Microsoft YaHei UI',sans-serif;background:#10141a;color:#d7dee4;
margin:0;padding:20px}h1{font-size:20px}h2{font-size:15px;color:#9fd0ff;
margin-top:28px;border-bottom:1px solid #26323d;padding-bottom:6px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12.5px}
th,td{border:1px solid #26323d;padding:4px 8px;text-align:left}th{background:#1a222b}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#7d8b96}.warn{color:#ffd28f}code{background:#1a222b;padding:1px 5px}
.kpi{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
.kpi div{background:#161d25;border:1px solid #26323d;border-radius:6px;
padding:8px 14px;min-width:130px}
.kpi b{display:block;font-size:17px;color:#9fd0ff}
.kpi span{font-size:11.5px;color:#7d8b96}
</style></head><body>"""


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return _esc(v)


class StageProfiler:
    """单次任务（建库/增量/搜图）的阶段画像。

    用法::

        prof = StageProfiler("search", "以图搜图", cfg=cfg, prefix=prefix)
        prof.start()
        prof.mark("加载索引")
        ...
        prof.mark("粗筛+扫描")
        path = prof.stop(extra=[("耗时明细", [("查询切块", 0.33), ...])])
    """

    def __init__(self, stage: str, title: str, *, cfg=None, prefix: str = "",
                 interval: float = 0.4, out_dir: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        self.stage = stage
        self.title = title
        self.cfg = cfg
        self.prefix = prefix or ""
        self.interval = max(0.1, float(interval))
        self.out_dir = out_dir or REPORT_DIR
        self.meta = dict(meta or {})
        self.counter: Dict[str, float] = {"done": 0.0, "total": 0.0}
        self.phases: List[dict] = []
        self.rows: List[dict] = []
        self._sampler = None
        self._t0 = 0.0
        self._rss0 = 0.0
        self._lock = threading.Lock()
        self._last_done = 0.0
        self._last_done_t = 0.0
        self._started = False
        self._stopped = False

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def start(self) -> "StageProfiler":
        import perfscope as PS
        self._t0 = time.time()
        self._rss0 = self._rss()
        self._last_done_t = self._t0
        self._sampler = PS.HardwareSampler(img_per_s=self._img_per_s)
        self._sampler.start()
        self._started = True
        self.mark("开始")
        return self

    def _img_per_s(self) -> float:
        now = time.time()
        done = float(self.counter.get("done") or 0.0)
        dt = now - self._last_done_t
        rate = 0.0
        if dt > 1e-6:
            rate = max(0.0, (done - self._last_done) / dt)
        self._last_done = done
        self._last_done_t = now
        return rate

    @staticmethod
    def _rss() -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 2 ** 20
        except Exception:  # noqa: BLE001
            return 0.0

    # ------------------------------------------------------------------
    # 打点
    # ------------------------------------------------------------------
    def mark(self, name: str, **info) -> None:
        with self._lock:
            self.phases.append({"name": name, "t": time.time() - self._t0,
                                "rss": self._rss(), "info": info})

    def bump(self, done: Optional[float] = None,
             total: Optional[float] = None) -> None:
        """进度回调里调用，用于换算 张/s（不参与打点，开销为一次赋值）。"""
        if done is not None:
            self.counter["done"] = float(done)
        if total is not None:
            self.counter["total"] = float(total)

    # ------------------------------------------------------------------
    # 结束 + 出报告
    # ------------------------------------------------------------------
    def stop(self, status: str = "ok",
             extra: Optional[List[Tuple[str, List[Tuple[str, Any]]]]] = None,
             note: str = "") -> str:
        if not self._started or self._stopped:
            return ""
        self._stopped = True
        self.mark("结束")
        rows = self._sampler.stop() if self._sampler is not None else []
        self.rows = rows
        total = max(time.time() - self._t0, 1e-6)
        report = {
            "stage": self.stage, "title": self.title, "status": status,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "total": total,
            "prefix": self.prefix, "interval": self.interval,
            "phases": self.phases, "rows": rows, "meta": self.meta,
            "counter": dict(self.counter), "note": note,
            "extra": extra or [],
            "cfg": self._cfg_snapshot(),
        }
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(self.out_dir, f"gui_{self.stage}_{stamp}")
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        with open(base + ".html", "w", encoding="utf-8") as f:
            f.write(self._html(report))
        return base + ".html"

    def _cfg_snapshot(self) -> Dict[str, Any]:
        if self.cfg is None:
            return {}
        keys = ("model", "device", "fp16", "batch", "workers", "decode_workers",
                "big_decode_conc", "tile_decode_slots", "tile_flush_ms",
                "torch_threads", "coarse_k", "top_k", "png_decoder")
        return {k: getattr(self.cfg, k, None) for k in keys}

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------
    def _html(self, rep: dict) -> str:
        import perfscope as PS
        rows = rep["rows"]
        out = [_STYLE]
        stage_txt = {"index": "索引(建库/增量)", "search": "搜图"}.get(
            rep["stage"], rep["stage"])
        out.append(f"<h1>{_esc(rep['title'])} · 性能画像</h1>")
        out.append(
            f"<p class='muted'>阶段 <b>{_esc(stage_txt)}</b> · "
            f"索引 <code>{_esc(rep['prefix'] or '-')}</code> · "
            f"生成 {_esc(rep['ts'])} · 状态 {_esc(rep['status'])}"
            + (f" · {_esc(rep['note'])}" if rep.get("note") else "")
            + "</p>")

        # ---- KPI ----
        def _col(key: str) -> List[float]:
            return [float(r[key]) for r in rows
                    if r.get(key) is not None]

        cpu_sys, cpu_proc = _col("cpu_sys"), _col("cpu_proc")
        gpu = _col("gpu")
        rss = _col("rssMB")
        rd = sum((r.get("rdMBps") or 0.0) * (r.get("t") or 0.0) for r in rows)
        wr = sum((r.get("wrMBps") or 0.0) * (r.get("t") or 0.0) for r in rows)
        avg = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        kpis = [
            ("总耗时", f"{rep['total']:.2f} s"),
            ("进程CPU 均值/峰值",
             f"{_fmt(avg(cpu_proc))}% / {_fmt(max(cpu_proc) if cpu_proc else None)}%"),
            ("系统CPU 均值/峰值",
             f"{_fmt(avg(cpu_sys))}% / {_fmt(max(cpu_sys) if cpu_sys else None)}%"),
            ("GPU 均值/峰值",
             f"{_fmt(avg(gpu))}% / {_fmt(max(gpu) if gpu else None)}%"),
            ("进程内存 起→终",
             f"{_fmt(rep['phases'][0]['rss'] if rep['phases'] else None, 0)} → "
             f"{_fmt(rss[-1] if rss else None, 0)} MB"),
            ("进程内存 峰值", f"{_fmt(max(rss) if rss else None, 0)} MB"),
            ("磁盘读/写", f"{rd:,.0f} / {wr:,.0f} MB"),
        ]
        if rep["counter"].get("done"):
            kpis.append(("处理量", f"{int(rep['counter']['done']):,} 张 / "
                                    f"{int(rep['counter'].get('total') or 0):,}"))
        out.append("<div class='kpi'>")
        for k, v in kpis:
            out.append(f"<div><b>{_esc(v)}</b><span>{_esc(k)}</span></div>")
        out.append("</div>")

        # ---- 时间轴 ----
        out.append("<h2>1. 时间轴（CPU / GPU / 内存 / 磁盘）</h2>")
        out.append(PS.svg_line(rows, [
            ("cpu_proc", "#3d8fd1", "进程CPU%"),
            ("cpu_sys", "#8fd13d", "系统CPU%"),
            ("gpu", "#ffb454", "GPU SM%"),
            ("rssMB", "#d13d8f", "进程内存MB"),
            ("rdMBps", "#5fd3c4", "读MB/s"),
        ]))
        out.append("<p class='muted'>各序列独立归一化，纵轴 0-100 为各自峰值的占比；"
                   "图例括号内为该序列峰值。</p>")

        # ---- 阶段表 ----
        out.append("<h2>2. 阶段耗时（打点）</h2><table>"
                   "<tr><th>阶段</th><th>起始(s)</th><th>耗时(s)</th>"
                   "<th>占比</th><th>进程内存(MB)</th><th>备注</th></tr>")
        ph = rep["phases"]
        for i, p in enumerate(ph):
            nxt = ph[i + 1] if i + 1 < len(ph) else None
            dur = (nxt["t"] - p["t"]) if nxt else (rep["total"] - p["t"])
            pct = dur / max(rep["total"], 1e-6) * 100
            info = " · ".join(f"{k}={_fmt(v, 2)}" for k, v in p["info"].items())
            out.append(f"<tr><td>{_esc(p['name'])}</td>"
                       f"<td class='num'>{p['t']:.2f}</td>"
                       f"<td class='num'>{dur:.2f}</td>"
                       f"<td class='num'>{pct:.1f}%</td>"
                       f"<td class='num'>{p['rss']:.0f}</td>"
                       f"<td>{_esc(info)}</td></tr>")
        out.append("</table>")

        # ---- 附加表 ----
        for idx, (tbl_title, kv) in enumerate(rep.get("extra") or [], start=3):
            out.append(f"<h2>{idx}. {_esc(tbl_title)}</h2>"
                       "<table><tr><th>项</th><th>数值</th></tr>")
            for k, v in kv:
                out.append(f"<tr><td>{_esc(k)}</td>"
                           f"<td class='num'>{_fmt(v, 4)}</td></tr>")
            out.append("</table>")

        # ---- 参数快照 ----
        if rep.get("cfg") or rep.get("meta"):
            out.append("<h2>参数快照</h2><table><tr><th>项</th><th>值</th></tr>")
            for k, v in list((rep.get("cfg") or {}).items()) + \
                    list((rep.get("meta") or {}).items()):
                out.append(f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt(v)}</td></tr>")
            out.append("</table>")
        out.append(f"<p class='muted'>采样间隔 {rep['interval']:.2f}s · "
                   f"{len(rows)} 个采样点 · 由 perfwatch 生成</p>")
        out.append("</body></html>")
        return "".join(out)


def latest_report(out_dir: Optional[str] = None) -> str:
    """最近一次生成的性能图（供 GUI「打开最近性能图」按钮）。"""
    d = out_dir or REPORT_DIR
    if not os.path.isdir(d):
        return ""
    files = [os.path.join(d, f) for f in os.listdir(d)
             if f.endswith(".html") and f.startswith("gui_")]
    if not files:
        return ""
    return max(files, key=os.path.getmtime)
