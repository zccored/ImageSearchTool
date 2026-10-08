# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 测试总控 + 存活弹窗
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""串跑本轮全部实测，并弹一个置顶小窗表示进程存活 / 显示当前步骤。

步骤（每步独立子进程，输出落 perf_reports/suite_<ts>/）：
  1 缓存编码对照（写侧/读侧，确定缓存格式）
  2 内存特征（cv2 vs libdeflate，含新的全局 scratch 预算与旁路裁剪）
  3 建库 A/B 瓦片 + 索引语义比对
  4 建库 A/B 整图 + 数组逐位比对
用法: python -E devtools/run_suite_popup.py
"""
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

STEPS = [
    ("1/4 缓存编码对照（写侧/读侧）", ["devtools/probe_cache_codec.py", "100", "256"]),
    ("2/4 内存特征（cv2 vs libdeflate）",
     ["devtools/probe_memory_profile.py", "--mode", "tiles", "--n", "540", "--dup", "60",
      "--configs", "cv2,ldf"]),
    ("3/4 建库 A/B 瓦片", None),      # 展开为多条命令
    ("4/4 建库 A/B 整图", None),
]
AB_TILES = [
    ["devtools/ab_build_bench.py", "--mode", "tiles", "--label", "s_cv2",
     "--n", "540", "--dup", "60", "--png-decoder", "cv2"],
    ["devtools/ab_build_bench.py", "--mode", "tiles", "--label", "s_ldf",
     "--n", "540", "--dup", "60", "--png-decoder", "libdeflate"],
    ["devtools/ab_build_bench.py", "--compare-tiles", "s_cv2", "s_ldf"],
]
AB_WHOLE = [
    ["devtools/ab_build_bench.py", "--mode", "whole", "--label", "sw_cv2",
     "--n", "540", "--dup", "60", "--png-decoder", "cv2"],
    ["devtools/ab_build_bench.py", "--mode", "whole", "--label", "sw_ldf",
     "--n", "540", "--dup", "60", "--png-decoder", "libdeflate"],
    ["devtools/ab_build_bench.py", "--compare-arrays", "sw_cv2", "sw_ldf"],
]

STATE = {"step": "启动…", "detail": "", "done": 0, "total": 4, "alive": True}


def popup():
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:                              # noqa: BLE001
        return
    root = tk.Tk()
    root.title("ImageSearchTool · 测试运行中（进程存活）")
    root.attributes("-topmost", True)
    root.geometry("+%d+%d" % (max(root.winfo_screenwidth() - 470, 0), 60))
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    a = ttk.Label(frm, text="启动…", font=("Microsoft YaHei UI", 10, "bold"))
    a.pack(anchor="w")
    b = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=430, justify="left")
    b.pack(anchor="w", pady=(4, 2))
    bar = ttk.Progressbar(frm, length=430, mode="determinate", maximum=4)
    bar.pack(fill="x", pady=4)
    c = ttk.Label(frm, text="", font=("Consolas", 9), wraplength=430, justify="left")
    c.pack(anchor="w")
    ttk.Button(frm, text="隐藏窗口（不影响跑测）", command=root.withdraw).pack(
        anchor="e", pady=(6, 0))
    root.update()
    while STATE["alive"]:
        try:
            a.config(text=STATE["step"])
            b.config(text=STATE["detail"])
            bar.config(value=STATE["done"])
            c.config(text="已完成 %d/%d 步" % (STATE["done"], STATE["total"]))
            root.update()
        except Exception:                          # noqa: BLE001
            break
        time.sleep(0.2)
    try:
        root.destroy()
    except Exception:                              # noqa: BLE001
        pass


def main() -> int:
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(_HERE, "perf_reports", "suite_%s" % ts)
    os.makedirs(outdir, exist_ok=True)
    threading.Thread(target=popup, daemon=True).start()
    logs = []

    def run(title, cmds, idx):
        STATE["step"] = title
        for k, c in enumerate(cmds):
            STATE["detail"] = " ".join(c[:4]) + (" …" if len(c) > 4 else "")
            log = os.path.join(outdir, "%d_%d.log" % (idx, k))
            logs.append(log)
            with open(log, "w", encoding="utf-8") as f:
                subprocess.call([sys.executable, "-E", os.path.join(_HERE, c[0])] + c[1:],
                                cwd=_HERE, stdout=f, stderr=subprocess.STDOUT)
        STATE["done"] = idx

    run(STEPS[0][0], [STEPS[0][1]], 1)
    run(STEPS[1][0], [STEPS[1][1]], 2)
    run(STEPS[2][0], AB_TILES, 3)
    run(STEPS[3][0], AB_WHOLE, 4)
    STATE["alive"] = False
    print("全部完成；日志目录:", outdir)
    for lg in logs:
        print("---", os.path.basename(lg))
        try:
            with open(lg, encoding="utf-8", errors="replace") as f:
                lines = [x.rstrip() for x in f if x.strip()]
            for x in lines[-8:]:
                print("   ", x)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
