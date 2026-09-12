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

"""结果网格回归：点击错位 + 不足 top_k（占位格子）。

构造 10 条命中，其中包含：正常图、超大 PNG(>30MP，无 draft → 占位)、
超大 JPEG(>30MP，draft 可预览)、损坏文件(占位)。验证：
  1) tiles 与 hits 一一对应（hit_index == 下标、网格无空洞）；
  2) 点选任意格子 → 详情/“打开原图”路径 == 该格对应的命中；
  3) 鼠标事件（event_generate）也走同一条路径。

用法: python devtools/verify_grid.py
"""
import os
import shutil
import sys
import tempfile
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image  # noqa: E402

import gui as G  # noqa: E402

work = tempfile.mkdtemp(prefix="gridtest_")
root = tk.Tk()
root.withdraw()
try:
    paths = []
    for i in range(7):                       # 7 张正常图
        p = os.path.join(work, f"ok{i}.jpg")
        Image.new("RGB", (400, 300), (30 * i % 256, 90, 180)).save(p)
        paths.append(p)
    p_huge_png = os.path.join(work, "huge.png")
    Image.new("RGB", (6000, 6000), (10, 120, 60)).save(p_huge_png)   # 36MP
    paths.append(p_huge_png)
    p_huge_jpg = os.path.join(work, "huge.jpg")
    Image.new("RGB", (7000, 5000), (200, 40, 90)).save(
        p_huge_jpg, quality=40)                                       # 35MP
    paths.append(p_huge_jpg)
    p_bad = os.path.join(work, "broken.jpg")
    with open(p_bad, "wb") as f:
        f.write(b"this is not an image at all")
    paths.append(p_bad)

    app = G.App(root)
    hits = [(i + 1, p, 0.9 - i * 0.01, 0.8, float("nan"), None, "full")
            for i, p in enumerate(paths)]
    app.last_hits = hits
    app._show_results(hits)

    print(f"命中 {len(hits)} 条 -> 渲染 {len(app.tiles)} 个格子")
    ok = True

    if len(app.tiles) != len(hits):
        print(f"  ✗ 格子数不符：{len(app.tiles)} != {len(hits)}")
        ok = False
    for i, t in enumerate(app.tiles):
        gi = t.frame.grid_info()
        want = divmod(i, 5)
        if t.hit_index != i:
            print(f"  ✗ 格子{i} hit_index={t.hit_index}")
            ok = False
        if (int(gi["row"]), int(gi["column"])) != want:
            print(f"  ✗ 格子{i} 网格位置 {(gi['row'], gi['column'])} != {want}")
            ok = False
    # 占位/正常判定
    kinds = []
    for i, t in enumerate(app.tiles):
        has_img = hasattr(t.btn, "image")
        kinds.append("图" if has_img else "占位")
    print("  各格状态:", " ".join(f"{i + 1}:{k}" for i, k in enumerate(kinds)))
    if not hasattr(app.tiles[7].btn, "image"):
        print("  ✓ 超大 PNG 走占位（PNG 不支持 draft）")
    else:
        print("  ✗ 超大 PNG 竟然出了预览")
        ok = False
    if hasattr(app.tiles[8].btn, "image"):
        print("  ✓ 超大 JPEG 经 draft 出预览")
    else:
        print("  ✗ 超大 JPEG 未出预览（draft 未生效）")
        ok = False
    if not hasattr(app.tiles[9].btn, "image"):
        print("  ✓ 损坏文件走占位")
    else:
        print("  ✗ 损坏文件竟然出了预览")
        ok = False

    # 鼠标事件需要窗口真正映射（withdrawn 的窗口收不到投递）：
    # 把窗口移到屏幕外再 deiconify，避免测试时抢焦点/闪窗
    root.geometry("+12000+12000")
    root.deiconify()
    root.update()

    # 点选一致性（方法调用 + 真实鼠标事件）
    for i in (0, 3, 7, 9):
        app._select_hit(i)
        got = app._selected_hit_path()
        want = hits[i][1]
        mark = "✓" if got == want else "✗"
        if got != want:
            ok = False
        print(f"  {mark} _select_hit({i}) -> {os.path.basename(got or '')}"
              f"  期望 {os.path.basename(want)}")
        if app.sel_tile is not app.tiles[i]:
            print(f"    ✗ 高亮格子不是 tiles[{i}]")
            ok = False
    for i in (2, 8):
        app.tiles[i].btn.event_generate("<Button-1>", x=2, y=2)
        root.update()
        got = app._selected_hit_path()
        mark = "✓" if got == hits[i][1] else "✗"
        if got != hits[i][1]:
            ok = False
        print(f"  {mark} 点击格子{i} -> {os.path.basename(got or '')}"
              f"  期望 {os.path.basename(hits[i][1])}"
              f"  (sel={getattr(app.sel_tile, 'hit_index', None)})")
        if hits[i][1] not in app.detail_var.get():
            print("    ✗ 详情栏不是该命中")
            ok = False
    print("结果:", "全部通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)
finally:
    try:
        root.destroy()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(work, ignore_errors=True)
