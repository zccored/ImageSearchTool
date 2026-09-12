# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 回归：查验去重 GUI 分组渲染/勾选/删除/移动/索引同步剔除
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""查验去重 GUI 回归：分组渲染 / 勾选 / 删除(回收站) / 移动 / 索引同步剔除。

用法: python devtools/verify_dedup_gui.py
"""
import os
import queue as _queue
import shutil
import sys
import tempfile
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import gui as G  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402
from hybrid_search.store import IndexFiles  # noqa: E402

# 关掉模态弹窗（测试自动确认）
G.messagebox.askokcancel = lambda *a, **k: True
G.messagebox.showinfo = lambda *a, **k: print(
    "    [info]", (a[1][:60] if len(a) > 1 else ""))
G.messagebox.showwarning = lambda *a, **k: print(
    "    [warn]", (a[1][:60] if len(a) > 1 else ""))
G.messagebox.showerror = lambda *a, **k: print(
    "    [error]", (a[1][:60] if len(a) > 1 else ""))

work = tempfile.mkdtemp(prefix="dedupgui_")
ok = True


def base_image(seed, w=900, h=700):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.stack([(xx / w * 200 + seed * 17) % 256,
                    (yy / h * 180 + 60) % 256,
                    ((xx + yy) / (w + h) * 220 + seed * 40) % 256],
                   axis=2).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for k in range(5):
        cx, cy = rng.randint(80, w - 80), rng.randint(80, h - 80)
        rad = rng.randint(40, 140)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                  fill=(rng.randint(0, 255), rng.randint(0, 255),
                        rng.randint(0, 255)))
    d.text((40, 40), f"IMG-{seed}", fill=(255, 255, 255))
    return im


def drain(app, want="dedup_done"):
    """取队列里的目标消息（跳过 progress 等）。"""
    while True:
        try:
            m = app.q.get_nowait()
        except _queue.Empty:
            return None, None
        if m[0] == want:
            return m[0], m[1]
        if m[0] == "error":
            return "error", m[1]


root = tk.Tk()
root.geometry("+12000+12000")
root.deiconify()
try:
    a = base_image(1)
    p_a = os.path.join(work, "a.jpg")
    a.save(p_a, quality=92)
    p_c1 = os.path.join(work, "a_复制1.jpg")
    shutil.copy2(p_a, p_c1)
    p_c2 = os.path.join(work, "a_复制2.jpg")
    shutil.copy2(p_a, p_c2)
    p_re = os.path.join(work, "a_重编码.jpg")
    a.save(p_re, quality=50)
    p_sm = os.path.join(work, "a_缩放.jpg")
    a.resize((810, 630)).save(p_sm, quality=92)
    others = []
    for s in (2, 3, 4):
        p = os.path.join(work, f"other{s}.jpg")
        base_image(s).save(p, quality=92)
        others.append(p)
    paths = [p_a, p_c1, p_c2, p_re, p_sm] + others
    print(f"图库 {len(paths)} 张：A 系列 5 张(1 原图 + 2 字节相同 + 2 近似) + 无关 3 张")

    # ---- 建索引（dedup=False 让全部入库，便于验证 prune）-------------
    cfg = Config()
    cfg.dedup = False
    cfg.workers = 4
    cfg.decode_workers = 2
    cfg.batch = 8
    prefix = os.path.join(work, "idx", "gallery")
    eng = HybridEngine(cfg)
    n0 = eng.build(prefix, paths=paths, force=True)
    del eng
    print(f"索引已建：{n0} 条")

    app = G.App(root)
    app.dir_var.set(work)
    app.all_images = list(paths)
    app.prefix = prefix

    print("\n== 1) 扫描 + 渲染 ==")
    app._dedup_worker(list(paths), 0.04)
    kind, rep = drain(app)
    if kind != "dedup_done":
        print("  ✗ 扫描失败:", rep)
        sys.exit(1)
    print(f"  组数={len(rep.groups)} 完全={rep.n_exact} 近似={rep.n_near} "
          f"图片={rep.n_images} 复用索引={rep.indexed_used} 解码={rep.decoded}")
    win = G.DedupWindow(app, rep)
    root.update()
    roots = win.tree.get_children()
    n_members = sum(len(win.tree.get_children(r)) for r in roots)
    print(f"  Treeview 组行 {len(roots)} 个，成员行 {n_members} 个")
    if len(rep.groups) != 1 or n_members != 5:
        print("  ✗ 分组渲染不符（应 1 组 5 张）")
        ok = False

    print("\n== 2) 勾选逻辑 ==")
    win._select_keep_best()
    sel = sorted(os.path.basename(p) for p, v in win.sel.items() if v)
    print("  每组保留最佳 ->", sel)
    if len(sel) != 4 or "a.jpg" in sel:
        print("  ✗ 应为 4 张（不含保留项 a.jpg）")
        ok = False
    win._select_exact()
    sel = sorted(os.path.basename(p) for p, v in win.sel.items() if v)
    print("  仅选完全重复 ->", sel)
    if sel != ["a_复制1.jpg", "a_复制2.jpg"]:
        print("  ✗ 应为 2 张字节相同副本")
        ok = False
    win._select_none()
    if any(win.sel.values()):
        print("  ✗ 全部不选失败")
        ok = False
    win._toggle(p_c1)
    iid = win.path_item[p_c1]
    if win.tree.set(iid, "sel") != "☑":
        print("  ✗ 点击切换后单元格未更新")
        ok = False
    else:
        print("  ✓ 单元格勾选状态随点击更新")

    print("\n== 3) 删除选中（回收站）+ 索引同步 ==")
    win._select_keep_best()
    before = len(app.all_images)
    win._delete_selected()
    root.update()
    gone = [os.path.basename(p) for p in (p_c1, p_c2, p_re, p_sm)
            if not os.path.exists(p)]
    print(f"  已移入回收站 {len(gone)}/4：{gone}")
    if len(gone) != 4:
        print("  ✗ 删除数量不符")
        ok = False
    if len(app.all_images) != before - 4:
        print(f"  ✗ 主列表未同步：{len(app.all_images)} != {before - 4}")
        ok = False
    n_after = len(IndexFiles(prefix).load_coarse()["paths"])
    print(f"  索引行数 {n0} -> {n_after}（应 {n0 - 4}）")
    if n_after != n0 - 4:
        print("  ✗ 索引未同步剔除")
        ok = False
    if rep.groups:
        print(f"  ✗ 删除后仍有 {len(rep.groups)} 组")
        ok = False

    print("\n== 4) 移动选中 ==")
    p_new = os.path.join(work, "b.jpg")
    base_image(7).save(p_new, quality=92)
    p_new2 = os.path.join(work, "b_复制.jpg")
    shutil.copy2(p_new, p_new2)
    cur = list(app.all_images) + [p_new, p_new2]
    app.all_images = cur
    # 先把新图入库，这样移动后索引里确实有需要剔除的条目
    eng = HybridEngine(cfg)
    eng.add(prefix, paths=[p_new, p_new2])
    del eng
    n_before_move = len(IndexFiles(prefix).load_coarse()["paths"])
    print(f"  新图入库后索引行数 {n_before_move}")
    app._dedup_worker(cur, 0.04)
    kind, rep2 = drain(app)
    print(f"  重新扫描：{len(rep2.groups)} 组")
    win2 = G.DedupWindow(app, rep2)
    root.update()
    win2._select_keep_best()
    dest = os.path.join(work, "moved_gallery")
    G.filedialog.askdirectory = lambda *a, **k: dest
    win2._move_selected()
    root.update()
    moved_ok = (not os.path.exists(p_new2)
                and os.path.exists(os.path.join(dest, "b_复制.jpg")))
    print(f"  b_复制.jpg 已移动到 {dest}: {moved_ok}")
    if not moved_ok:
        print("  ✗ 移动失败")
        ok = False
    n_after2 = len(IndexFiles(prefix).load_coarse()["paths"])
    print(f"  索引行数 {n_before_move} -> {n_after2}（应 {n_before_move - 1}）")
    if n_after2 != n_before_move - 1:
        print("  ✗ 移动后索引未同步剔除")
        ok = False

    print("\n结果:", "全部通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)
finally:
    try:
        root.destroy()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(work, ignore_errors=True)
