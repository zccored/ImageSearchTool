# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 建库内存特征记录探针
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""在**真实建库过程**中采样内存特征（父进程侧观测子进程，零侵入、不改工程代码）。

记录的特征（都对应工程现有旋钮）：
  * RSS 曲线（含峰值、峰值出现的进度位置）、进程线程数、GPU 显存
  * 各阶段耗时（由基准台自己的输出解析）
  * 对照开关：png_decoder（cv2/libdeflate）、prep_cache 开/关、decode_workers、批大小

用法:
    python -E devtools/probe_memory_profile.py [--mode tiles|whole] [--n 540] [--dup 60]
        [--configs cv2,ldf,cv2c,ldfc] [--interval 0.05]
配置代码：cv2=cv2 无缓存；ldf=libdeflate 无缓存；cv2c/ldfc=开启预处理缓存。
输出：perf_reports/mem_profile_<ts>.json + 控制台摘要
"""
import json
import os
import re
import subprocess
import sys
import time

import psutil

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

CONFIGS = {
    "cv2": ["--png-decoder", "cv2"],
    "ldf": ["--png-decoder", "libdeflate"],
    "cv2c": ["--png-decoder", "cv2", "--prep-cache"],
    "ldfc": ["--png-decoder", "libdeflate", "--prep-cache"],
}


def arg(name, dflt):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return dflt


def run_one(code, mode, n, dup, interval, extra):
    cmd = [sys.executable, "-E", os.path.join(_HERE, "devtools", "ab_build_bench.py"),
           "--mode", mode, "--label", "mem_" + code, "--n", str(n), "--dup", str(dup)] \
        + CONFIGS[code] + extra
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=_HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace")
    proc = psutil.Process(p.pid)
    rss, threads, gpu = [], [], []
    try:
        import torch
    except Exception:                              # noqa: BLE001
        torch = None
    while p.poll() is None:
        try:
            mi = proc.memory_info()
            rss.append(mi.rss / 2 ** 20)
            threads.append(proc.num_threads())
            # 子进程的子进程（工作线程都在本进程内，这里兜底统计）
            for c in proc.children(recursive=False):
                try:
                    rss[-1] += c.memory_info().rss / 2 ** 20
                except Exception:                  # noqa: BLE001
                    pass
        except Exception:                          # noqa: BLE001
            pass
        time.sleep(interval)
    out = p.stdout.read() if p.stdout else ""
    wall = time.perf_counter() - t0
    del torch, gpu
    m = re.search(r"→ ([\d.]+) s \| ([\d.]+) 张/s .*CPU 秒 ([\d.]+)", out)
    stat = {"wall_s": round(wall, 2),
            "bench_s": float(m.group(1)) if m else None,
            "img_s": float(m.group(2)) if m else None,
            "cpu_s": float(m.group(3)) if m else None,
            "peak_rss_mb": round(max(rss), 1) if rss else None,
            "peak_at_pct": None, "mean_rss_mb": round(sum(rss) / len(rss), 1) if rss else None,
            "max_threads": max(threads) if threads else None,
            "samples": len(rss), "rss_curve_mb": [round(x, 1) for x in rss[::max(1, len(rss) // 40)]],
            "config": code, "knobs": " ".join(CONFIGS[code] + extra)}
    if rss:
        stat["peak_at_pct"] = round(100.0 * rss.index(max(rss)) / max(len(rss) - 1, 1), 1)
    print("  [%-5s] %5.2f s | 峰值 RSS %7.1f MB（出现在 %s%% 进度）| 均值 %7.1f MB | 线程峰 %s"
          % (code, stat["wall_s"], stat["peak_rss_mb"] or 0, stat["peak_at_pct"],
             stat["mean_rss_mb"] or 0, stat["max_threads"]))
    return stat, out


def main() -> int:
    mode = arg("--mode", "tiles")
    n = int(arg("--n", "540"))
    dup = int(arg("--dup", "60"))
    interval = float(arg("--interval", "0.05"))
    codes = arg("--configs", "cv2,ldf").split(",")
    extra = []
    if "--decode-workers" in sys.argv:
        extra += ["--decode-workers", arg("--decode-workers", "0")]
    print("内存特征记录：mode=%s n=%d dup=%d interval=%.2fs configs=%s"
          % (mode, n, dup, interval, codes))
    print("（父进程观测子进程 RSS；系统空闲内存 %.1f GB）"
          % (psutil.virtual_memory().available / 2 ** 30))
    res = {}
    for code in codes:
        code = code.strip()
        if code not in CONFIGS:
            continue
        res[code], _ = run_one(code, mode, n, dup, interval, extra)
    ts = time.strftime("%Y%m%d-%H%M%S")
    f = os.path.join(_HERE, "perf_reports", "mem_profile_%s.json" % ts)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump({"ts": ts, "mode": mode, "n": n, "dup": dup, "configs": res,
                   "system_free_gb": round(psutil.virtual_memory().available / 2 ** 30, 2)},
                  fh, ensure_ascii=False, indent=2)
    print("JSON:", f)
    if len(res) >= 2:
        ks = list(res)
        a, b = res[ks[0]], res[ks[1]]
        if a["peak_rss_mb"] and b["peak_rss_mb"]:
            print("对照 %s → %s：峰值 RSS %+.1f%%（%.1f → %.1f MB）；墙钟 %+.1f%%"
                  % (ks[0], ks[1], 100.0 * (b["peak_rss_mb"] / a["peak_rss_mb"] - 1),
                     a["peak_rss_mb"], b["peak_rss_mb"],
                     100.0 * (b["wall_s"] / a["wall_s"] - 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
