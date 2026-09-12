# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 验证：缩略图管线与对比窗缩放性能（真实大图）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""缩略图管线 + 对比窗缩放性能验证（真实报告/真实大图）。

用法: python devtools/verify_thumb_perf.py
"""
import os
import pickle
import sys
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import psutil  # noqa: E402

import gui as G  # noqa: E402
from compare_view import CompareWindow  # noqa: E402

if "--cold" in sys.argv:
    # 冷启动：把缩略图缓存指到空目录，测“首次打开”的真实代价
    import tempfile
    _cold_dir = tempfile.mkdtemp(prefix="thumbcold_")
    _Orig = G.ThumbCache

    class _Cold(_Orig):
        def __init__(self, root, size=96, quality=82):
            super().__init__(_cold_dir, size, quality)

    G.ThumbCache = _Cold
    print(f"[冷启动] 缓存目录 {_cold_dir}")

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "perf_reports", "dedup_report_F视频.pkl")
PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 2 ** 20


def pump(win, seconds=0.0, until=None, timeout=30.0):
    """跑 Tk 事件循环：until 给定时等到条件满足或超时；否则跑满 seconds。"""
    widget = win.win if hasattr(win, "win") else win
    t0 = time.time()
    while True:
        try:
            widget.update()
        except tk.TclError:
            return time.time() - t0
        elapsed = time.time() - t0
        if until is not None:
            if until() or elapsed > timeout:
                return elapsed
        elif seconds > 0 and elapsed >= seconds:
            return elapsed
        elif seconds <= 0:
            return elapsed
        time.sleep(0.01)


rep = pickle.load(open(CACHE, "rb"))
print(f"报告：{len(rep.groups)} 组 / {rep.n_images} 张")

root = tk.Tk()
root.geometry("+12000+12000")
root.deiconify()
try:
    app = G.App(root)
    app.dir_var.set(r"F:\视频")
    app.prefix = r"F:\视频\.gallery_index\gallery"

    print("\n== 1) 审查窗构建 + 首屏缩略图 ==")
    t0 = time.time()
    win = G.DedupWindow(app, rep)
    pump(win, 0.3)
    print(f"  构建 {time.time() - t0:.2f}s | 可见成员行 "
          f"{len(win._visible)} | 内存 {rss():.0f}MB")
    t0 = time.time()
    pump(win, until=lambda: len(win._applied) >= min(len(win._visible), 1),
         timeout=30)
    print(f"  首屏首批缩略图 {time.time() - t0:.2f}s "
          f"(已贴 {len(win._applied)} 张)")
    t0 = time.time()
    pump(win, until=lambda: len(win._applied) >= len(win._visible) * 0.8,
         timeout=60)
    print(f"  首屏基本填满 {time.time() - t0:.2f}s "
          f"(已贴 {len(win._applied)} 张) 内存 {rss():.0f}MB")
    print(f"  缓存统计 {win.cache.stats()}")

    print("\n== 2) 滚动到中部 / 底部 ==")
    for frac in (0.5, 0.9):
        win.tree.yview_moveto(frac)
        pump(win, 0.2)
        t0 = time.time()
        pump(win, until=lambda: len(win._applied) >= len(win._visible) * 0.8,
             timeout=60)
        print(f"  {frac:.0%} 处 {time.time() - t0:.2f}s "
              f"(已贴 {len(win._applied)} 张) 内存 {rss():.0f}MB")

    print("\n== 3) 停留后释放显示范围外的缩略图 ==")
    before = len(win._photo)
    pump(win, 3.2)
    print(f"  PhotoImage 持有 {before} -> {len(win._photo)} | "
          f"已贴 {len(win._applied)} 张 | 内存 {rss():.0f}MB")
    if len(win._photo) > len(win._visible):
        print("  ✗ 释放后仍持有超出可见范围的缩略图")
    else:
        print("  ✓ 已释放范围外缩略图")

    print("\n== 4) 重开窗口（磁盘缓存命中）==")
    win.win.destroy()
    pump(win, 0.2)
    t0 = time.time()
    win2 = G.DedupWindow(app, rep)
    pump(win2, 0.2)
    t0 = time.time()
    pump(win2, until=lambda: len(win2._applied) >= len(win2._visible) * 0.8,
         timeout=60)
    print(f"  二次打开首屏填满 {time.time() - t0:.2f}s "
          f"(已贴 {len(win2._applied)} 张) 内存 {rss():.0f}MB")
    print(f"  缓存统计 {win2.cache.stats()}")

    print("\n== 5) 一键选中 MD5相同且未入库 + 只看已选 ==")
    t0 = time.time()
    win2._select_exact_unindexed()
    pump(win2, 0.3)
    n_sel = len(win2._selected())
    shown = len(win2.path_item)
    print(f"  选中 {n_sel} 张 / 视图显示 {shown} 行 / 耗时 {time.time() - t0:.2f}s")
    print(f"  提示：{win2.msg_var.get()[:80]}")
    if n_sel == 0 or shown == 0:
        print("  ✗ 一键选中或过滤视图异常")
    else:
        print("  ✓ 一键选中后自动切到“只看含勾选的组”，勾选可见")

    print("\n== 6) 对比窗缩放渲染耗时（真实 24MP 图）==")
    g = rep.groups[0]
    p = next((m.path for m in g.members if os.path.exists(m.path)), None)
    cw = CompareWindow(win2.win, g.members, p, dedup=win2)
    cw.win.geometry("+12000+12000")
    pump(cw, 0.4)
    for pane_name, pane in (("左", cw.left), ("右", cw.right)):
        if pane.image is None:
            continue
        # 预热：把各采样层建好（首次 reduce 是一次性成本）
        for _ in range(6):
            pane.zoom_at(pane.canvas.winfo_width() / 2,
                         pane.canvas.winfo_height() / 2, 1.2)
        pump(cw, 0.3)
        times = []
        for _ in range(8):
            t0 = time.time()
            pane.zoom_at(pane.canvas.winfo_width() / 2,
                         pane.canvas.winfo_height() / 2,
                         1.2 if _ % 2 == 0 else 1 / 1.2)
            times.append((time.time() - t0) * 1000)
        best = []
        for _ in range(3):
            t0 = time.time()
            pane.render("best")
            best.append((time.time() - t0) * 1000)
        print(f"  {pane_name}区 稳态缩放 "
              f"{sum(times) / len(times):.1f}ms/次 "
              f"(最慢 {max(times):.1f}ms) | 高质量重绘 "
              f"{sum(best) / len(best):.1f}ms/次 | 分层="
              f"{'small' if pane._small is not None else '-'}"
              f"/{'q' if pane._q is not None else '-'}"
              f"/{'half' if pane._half is not None else '-'}")
    cw.close()
    print(f"\n峰值内存 {rss():.0f}MB")
    win2.win.destroy()
finally:
    try:
        root.destroy()
    except Exception:  # noqa: BLE001
        pass
