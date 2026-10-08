# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 加权准入（大图通道）离线仿真
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""用**实测单张解码耗时**仿真"加权准入"（weighted semaphore）的效果，先证明再改管线。

模型：18 路池；任务按权重 w 扣配额（预算 B = 路数 × 权重基数）。
  * 基线   ：无准入，任何时刻最多 18 张在跑（= 现状）
  * 加权   ：小图 w=1，大图按分档 p99 取 w，配额不足则等待（大图自然降并发）
输出：总墙钟、**有效占用核数**（= Σ单张耗时 / 墙钟）、以及按档的排队延迟。

注意：仿真假设"单张耗时与并发无关"（真实会有缓存/带宽干扰），所以它是**乐观上界**；
若连乐观上界都提不上去，就不值得改管线。

用法: python -E devtools/sim_weighted_admission.py [样本数=540] [路数=18]
"""
import heapq
import json
import os
import random
import struct
import sys
import threading
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

EXTS = [".jpeg", ".jpg", ".png", ".webp", ".bmp"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 540
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 18
# 分档权重：按实测 p99 取整（用户建议；p50 会低估 4 倍）
WEIGHT = [(1e6, 1), (4e6, 3), (12e6, 7), (float("inf"), 20)]
BUDGET_UNIT = 6            # 预算 = 路数 × 该基数（用户建议 18×6=108）


def weight_of(px):
    for lim, w in WEIGHT:
        if px < lim:
            return w
    return WEIGHT[-1][1]


def measure(n, workers):
    """实测每张图的解码耗时（18 路池，与建库同规模）。"""
    from hybrid_search.io_utils import collect_images, decode_rgb
    paths = collect_images(GALLERY_ROOT, EXTS)
    random.seed(11)
    random.shuffle(paths)
    sample = paths[:n]
    lock = threading.Lock()
    rows, idx = [], [0]

    def work():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(sample):
                return
            try:
                data = open(sample[i], "rb").read()
            except OSError:
                continue
            t0 = time.perf_counter()
            arr = decode_rgb(data)
            dt = time.perf_counter() - t0
            if arr is None:
                continue
            with lock:
                rows.append({"s": dt, "px": int(arr.shape[0]) * int(arr.shape[1])})
    ths = [threading.Thread(target=work, daemon=True) for _ in range(workers)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return rows, time.perf_counter() - t0


def simulate(rows, workers, budget):
    """事件驱动仿真：同时满足"在跑张数 < workers"与"在跑权重和 + w ≤ budget"才开工。

    正确建模加权准入：维护在跑任务的 (结束时刻, 权重) 堆，配额不足时时间推进到最近一次
    释放再判；budget=0 表示无准入（现状）。
    """
    order = list(rows)
    random.shuffle(order)                       # 大图均匀混入，符合真实建库顺序
    now, running, cur_w, total = 0.0, [], 0.0, 0.0

    def drain(upto):
        nonlocal cur_w
        while running and running[0][0] <= upto:
            _, w = heapq.heappop(running)
            cur_w -= w

    for r in order:
        w = weight_of(r["px"]) if budget else 1
        while True:
            drain(now)
            if len(running) < workers and (not budget or cur_w + w <= budget):
                break
            if not running:                  # 理论上不会发生（w ≤ budget）
                break
            now = running[0][0]              # 推进到最近一次释放
            drain(now)
        heapq.heappush(running, (now + r["s"], w))
        cur_w += w
        total += r["s"]
    wall = max([t for t, _ in running] + [now])
    return wall, total / max(wall, 1e-9), []


def main() -> int:
    rows, wall_real = measure(N, WORKERS)
    tot = sum(r["s"] for r in rows)
    print("实测：%d 张 / %.2f s（%.1f 张/s）| 总功 %.1f 核秒 | 有效核数 %.2f"
          % (len(rows), wall_real, len(rows) / wall_real, tot, tot / wall_real))
    by = defaultdict(list)
    for r in rows:
        by[str(weight_of(r["px"]))].append(r["s"])
    print("按权重分档：", {k: "n=%d 均%.0fms" % (len(v), 1000 * sum(v) / len(v))
                      for k, v in sorted(by.items(), key=lambda kv: int(kv[0]))})
    print("\n== 仿真（假设单张耗时与并发无关 = 乐观上界）")
    print("  %-22s %8s %10s %10s" % ("方案", "墙钟 s", "有效核数", "相对基线"))
    base = None
    out = {}
    for tag, budget in [("基线：18 路无准入", 0),
                        ("加权 B=18×4=%d" % (18 * 4), 18 * 4),
                        ("加权 B=18×6=%d" % (18 * 6), 18 * 6),
                        ("加权 B=18×8=%d" % (18 * 8), 18 * 8),
                        ("加权 B=18×12=%d" % (18 * 12), 18 * 12)]:
        w, cores, waits = simulate(rows, WORKERS, budget)
        if base is None:
            base = w
        out[tag] = {"wall": round(w, 3), "cores": round(cores, 2)}
        print("  %-22s %8.2f %10.2f %9.1f%%" % (tag, w, cores, 100 * (w / base - 1)))
    best = max(out.items(), key=lambda kv: kv[1]["cores"])
    print("\n最佳：%s（有效核数 %.2f，墙钟 %+.1f%%）"
          % (best[0], best[1]["cores"], 100 * (best[1]["wall"] / base - 1)))
    jf = os.path.join(_HERE, "perf_reports", "sim_admission_%s.json" % time.strftime("%Y%m%d-%H%M%S"))
    with open(jf, "w", encoding="utf-8") as f:
        json.dump({"ts": time.strftime("%Y%m%d-%H%M%S"), "n": len(rows), "workers": WORKERS,
                   "real_wall_s": round(wall_real, 3), "real_cores": round(tot / wall_real, 2),
                   "sim": out, "weights": WEIGHT}, f, ensure_ascii=False, indent=2)
    print("JSON:", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
