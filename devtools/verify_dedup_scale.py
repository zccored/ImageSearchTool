# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 真实规模（2.7 万张）下审查窗口的构建与滚动性能
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""真实规模（F:\\视频，6600 组 / 2.7 万张）下审查窗口的构建与滚动性能。

首次运行会做一次只读扫描（约 90s）并缓存报告；之后复用缓存秒开。
用法: python devtools/verify_dedup_scale.py [--rescan]
"""
import glob
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
from hybrid_search import dedup as DD  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402

ROOT = r"F:\视频"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "perf_reports")
CACHE = os.path.join(CACHE_DIR, "dedup_report_F视频.pkl")
PROC = psutil.Process()


def rss():
    return PROC.memory_info().rss / 2 ** 20


def get_report(rescan: bool):
    if not rescan and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            rep = pickle.load(f)
        print(f"复用缓存报告：{len(rep.groups)} 组")
        return rep
    cfg = Config()
    paths = collect_images(ROOT, cfg.extensions)
    print(f"扫描 {len(paths)} 张（只读，约 90s）…")
    t0 = time.time()
    rep = DD.scan_duplicates(paths,
                             prefix=os.path.join(ROOT, ".gallery_index",
                                                 "gallery"),
                             threshold=0.02)
    print(f"扫描完成 {time.time() - t0:.1f}s：{len(rep.groups)} 组 / "
          f"{rep.n_images} 张 / 可释放 {rep.wasted_bytes / 2**30:.1f}GB")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(rep, f, protocol=4)
    return rep


root = tk.Tk()
root.geometry("+12000+12000")
root.deiconify()
try:
    rep = get_report("--rescan" in sys.argv)
    app = G.App(root)
    app.dir_var.set(ROOT)
    app.all_images = []
    app.prefix = os.path.join(ROOT, ".gallery_index", "gallery")

    print(f"\n内存基线 {rss():.0f}MB")
    t0 = time.time()
    win = G.DedupWindow(app, rep)
    root.update()
    build = time.time() - t0
    roots = win.tree.get_children()
    members = sum(len(win.tree.get_children(r)) for r in roots)
    print(f"窗口构建 {build:.2f}s | 组行 {len(roots)} | 成员行 {members} | "
          f"内存 {rss():.0f}MB")

    # 懒加载：可见区缩略图
    t0 = time.time()
    for _ in range(3):
        win._load_visible_thumbs()
        root.update()
    print(f"首屏缩略图（3 轮懒加载）{time.time() - t0:.2f}s | "
          f"已解码 {len(win.thumb_done)} 张")

    # 滚到中部/底部再懒加载
    for frac in (0.5, 0.95):
        win.tree.yview_moveto(frac)
        root.update()
        t0 = time.time()
        win._load_visible_thumbs()
        root.update()
        print(f"滚到 {frac:.0%} 处懒加载 {time.time() - t0:.2f}s | "
              f"累计解码 {len(win.thumb_done)} 张")

    # 全选/反选在 2.7 万行上的耗时
    t0 = time.time()
    win._select_keep_best()
    root.update()
    print(f"一键“每组保留最佳” {time.time() - t0:.2f}s | "
          f"已选 {len(win._selected())} 张")
    t0 = time.time()
    win._select_none()
    root.update()
    print(f"全部不选 {time.time() - t0:.2f}s")
    print(f"峰值内存 {rss():.0f}MB")
    win.win.destroy()
finally:
    try:
        root.destroy()
    except Exception:  # noqa: BLE001
        pass
