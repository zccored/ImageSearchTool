# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 回归：大图对比窗双击/载入/缩放/拖动投放/复选框联动
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""大图对比窗回归：双击打开 / 左右载入 / 缩放 / 拖动投放 / 复选框联动 /
一键选中“MD5相同且未入库” / F11 / Esc。

用法: python devtools/verify_compare.py
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
from compare_view import CompareWindow  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402

G.messagebox.askokcancel = lambda *a, **k: True
G.messagebox.showinfo = lambda *a, **k: print(
    "    [info]", (a[1][:70] if len(a) > 1 else "").replace("\n", " "))
G.messagebox.showwarning = lambda *a, **k: print("    [warn]")
G.messagebox.showerror = lambda *a, **k: print("    [error]")

work = tempfile.mkdtemp(prefix="cmptest_")
ok = True


def base_image(seed, w=1400, h=1000):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.stack([(xx / w * 200 + seed * 17) % 256,
                    (yy / h * 180 + 60) % 256,
                    ((xx + yy) / (w + h) * 220 + seed * 40) % 256],
                   axis=2).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for k in range(6):
        cx, cy = rng.randint(100, w - 100), rng.randint(100, h - 100)
        rad = rng.randint(60, 200)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                  fill=(rng.randint(0, 255), rng.randint(0, 255),
                        rng.randint(0, 255)))
    d.text((50, 50), f"IMG-{seed}", fill=(255, 255, 255))
    return im


class FakeEvent:
    """模拟 Tk 事件（拖动/双击用）。"""

    def __init__(self, x=0, y=0, x_root=0, y_root=0, delta=120):
        self.x, self.y = x, y
        self.x_root, self.y_root = x_root, y_root
        self.delta = delta


def drain(app, want="dedup_done"):
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
    # ---- 构造：1 原图 + 2 字节相同副本(未入库) + 1 近似变体 -----------
    a = base_image(1)
    p_a = os.path.join(work, "a.jpg")
    a.save(p_a, quality=92)
    p_c1 = os.path.join(work, "a_复制1.jpg")
    shutil.copy2(p_a, p_c1)
    p_c2 = os.path.join(work, "a_复制2.jpg")
    shutil.copy2(p_a, p_c2)
    p_re = os.path.join(work, "a_重编码.jpg")
    a.save(p_re, quality=55)
    paths = [p_a, p_c1, p_c2, p_re]

    cfg = Config()
    cfg.dedup = False
    cfg.workers = 4
    cfg.decode_workers = 2
    prefix = os.path.join(work, "idx", "gallery")
    eng = HybridEngine(cfg)
    eng.build(prefix, paths=[p_a], force=True)      # 只把原图入库
    del eng

    app = G.App(root)
    app.dir_var.set(work)
    app.all_images = list(paths)
    app.prefix = prefix
    app._dedup_worker(paths, 0.02)
    kind, rep = drain(app)
    print(f"扫描：{len(rep.groups)} 组 / {rep.n_images} 张")
    win = G.DedupWindow(app, rep)
    root.update()

    print("\n== 1) 一键选中“MD5相同且未入库” ==")
    win._select_exact_unindexed()
    sel = sorted(os.path.basename(p) for p, v in win.sel.items() if v)
    print("  选中:", sel)
    if sel != ["a_复制1.jpg", "a_复制2.jpg"]:
        print("  ✗ 应只选中两张未入库的字节副本")
        ok = False
    win._select_none()

    print("\n== 2) 双击打开对比窗 ==")
    g = rep.groups[0]
    member_iid = win.path_item[g.members[1].path]
    bbox = win.tree.bbox(member_iid)
    win._on_double(FakeEvent(y=bbox[1] + bbox[3] // 2))
    root.update()
    if not win._compares:
        print("  ✗ 未打开对比窗")
        sys.exit(1)
    cw = win._compares[-1]
    cw.win.geometry("+12000+12000")
    root.update()
    print(f"  对比窗已开：左={os.path.basename(cw.left.path or '')} "
          f"右={os.path.basename(cw.right.path or '')} "
          f"小图 {len(cw._thumb_rows)} 项")
    if cw.left.path != g.members[1].path:
        print("  ✗ 左侧不是双击的那张")
        ok = False
    if len(cw._thumb_rows) != len(g.members):
        print("  ✗ 小图栏数量不符")
        ok = False

    print("\n== 3) 单击小图 / 缩放 / 复位 ==")
    other = g.members[3].path
    cw.set_side("left", other)
    root.update()
    if cw.left.path != other:
        print("  ✗ 左侧未切换")
        ok = False
    z0 = cw.left.zoom
    cw._on_wheel(cw.left, FakeEvent(x=200, y=200, delta=120))
    root.update()
    z1 = cw.left.zoom
    print(f"  缩放 {z0:.4f} -> {z1:.4f}（应放大）")
    if not z1 > z0:
        print("  ✗ 滚轮未放大")
        ok = False
    cw.left.fit()
    root.update()
    cw.right.fit()
    root.update()
    if cw.left.photo is None or cw.right.photo is None:
        print("  ✗ 渲染未产出图像")
        ok = False
    else:
        print(f"  ✓ 左右均已渲染 "
              f"({cw.left.photo.width()}x{cw.left.photo.height()} / "
              f"{cw.right.photo.width()}x{cw.right.photo.height()})")

    print("\n== 4) 拖动投放（小图 -> 右展示区；左区 -> 右区）==")
    right_cv = cw.right.canvas
    rx = right_cv.winfo_rootx() + right_cv.winfo_width() // 2
    ry = right_cv.winfo_rooty() + right_cv.winfo_height() // 2
    drag_path = g.members[3].path          # 近似变体（与右区当前不同）
    cw._strip_press(drag_path, FakeEvent(x=5, y=5, x_root=0, y_root=0))
    cw._strip_motion(FakeEvent(x_root=rx, y_root=ry))
    cw._strip_release(FakeEvent(x_root=rx, y_root=ry))
    root.update()
    print(f"  小图拖到右侧 -> {os.path.basename(cw.right.path or '')}")
    if cw.right.path != drag_path:
        print("  ✗ 拖动未落到右展示区")
        ok = False
    if cw.active != "right":
        print("  ✗ 投放后未把右侧设为活动图")
        ok = False
    # 左展示区 -> 右展示区
    cw.set_side("left", p_c2)              # 换成另一张，确保断言有区分度
    root.update()
    left_cv = cw.left.canvas
    lx = left_cv.winfo_rootx() + 40
    ly = left_cv.winfo_rooty() + 40
    left_path = cw.left.path
    cw._pane_press(cw.left, FakeEvent(x=40, y=40, x_root=lx, y_root=ly))
    cw._pane_motion(cw.left, FakeEvent(x=120, y=140, x_root=rx, y_root=ry))
    cw._pane_release(cw.left, FakeEvent(x=120, y=140, x_root=rx, y_root=ry))
    root.update()
    print(f"  左区拖到右侧 -> {os.path.basename(cw.right.path or '')}"
          f"（应={os.path.basename(left_path or '')}）")
    if cw.right.path != left_path:
        print("  ✗ 左区拖动未落到右展示区")
        ok = False

    print("\n== 5) 复选框联动 ==")
    row = cw._thumb_rows[p_c1]
    row["var"].set(True)
    cw._toggle_check(p_c1)                     # 模拟点击复选框
    print(f"  审查窗 sel[{os.path.basename(p_c1)}] = {win.sel.get(p_c1)}")
    if not win.sel.get(p_c1):
        print("  ✗ 对比窗勾选未同步到审查窗")
        ok = False
    win.sel[p_c1] = False                      # 审查窗侧改动
    cw.sync_checks()
    root.update()
    if cw._thumb_rows[p_c1]["var"].get():
        print("  ✗ 审查窗改动未回灌对比窗")
        ok = False
    else:
        print("  ✓ 双向联动正常")

    print("\n== 6) F11 全屏 / Esc 关闭 ==")
    cw.toggle_fullscreen()
    root.update()
    fs = bool(cw.win.attributes("-fullscreen"))
    cw.toggle_fullscreen()
    root.update()
    print(f"  F11 全屏属性 = {fs}")
    if not fs:
        print("  ✗ 全屏未生效")
        ok = False
    cw.close()
    root.update()
    if CompareWindow._open:
        print("  ✗ 关闭后仍在 _open 列表")
        ok = False
    else:
        print("  ✓ 关闭正常（_open 已清空）")

    print("\n结果:", "全部通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)
finally:
    try:
        root.destroy()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(work, ignore_errors=True)
