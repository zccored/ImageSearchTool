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

"""重复图对比预览窗（双击审查窗口的某一行打开）。

布局（黑底、窄边框、可 F11 无边框全屏）：

    ┌───────────────┬───────────────┬────────┐
    │  左展示区     │  右展示区     │ 小图栏 │
    │ （Canvas）    │ （临时对比）  │ ☑ 缩略 │
    │               │               │ ☑ 缩略 │
    └───────────────┴───────────────┴────────┘

交互：
  * 单击小图 → 载入左展示区；单击展示区 → 该侧成为“活动图”（边框高亮 +
    小图栏对应项高亮）；
  * 拖动小图 → 可放到左/右任意展示区；从左展示区拖到右展示区 = 右边作为
    临时对比槽；拖动时目标区边缘高亮；
  * 滚轮缩放（以光标为锚点）、中键拖动平移、双击复位；
  * 小图栏复选框与“重复图审查窗口”的勾选双向联动；
  * F11 无边框全屏，Esc 退出全屏 / 关闭（并关闭所有对比窗）。

渲染质量：原图按可见区域裁剪后再重采样（缩放用 LANCZOS，放大 ≤4x 用 BICUBIC，
>4x 用 NEAREST 便于看像素），不整图缩放，因此既清晰又不卡。
"""
from __future__ import annotations

import os
import tkinter as tk
from typing import Callable, Dict, List, Optional

from PIL import Image, ImageOps, ImageTk

try:
    from hybrid_search.io_utils import human_bytes
except Exception:                       # noqa: BLE001 —— 独立运行时的兜底
    def human_bytes(n: float) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if abs(n) < 1024 or unit == "GB":
                return f"{n:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}GB"


THUMB_EDGE = 96
THUMB_ROW_H = THUMB_EDGE + 8
STRIP_W = 296                       # 小图栏总宽（含信息文字）
CACHE_MAX = 2                       # 原图缓存张数（27MP 图 ≈ 80MB/张）
MIP_THRESHOLD = 0.35                # 缩到该比例以下改用 1/4 采样源
ZOOM_MIN, ZOOM_MAX = 0.05, 16.0


def _fast_resize(img: Image.Image, size: tuple,
                 resample: int) -> Image.Image:
    """大倍率降采样：先整数倍 reduce（box 平均，C 循环很快）再精修。

    PIL 直接 resize 的降采样极慢（实测 1.7MP → 0.17MP：BILINEAR 213ms、
    LANCZOS 563ms），这正是“缩放太卡”的主因；先 reduce 到接近目标尺寸后
    只剩小图重采样，耗时降到 ~10ms 量级。"""
    tw, th = max(1, size[0]), max(1, size[1])
    if img.width > tw * 2 or img.height > th * 2:
        f = max(1, min(img.width // max(tw, 1), img.height // max(th, 1)))
        if f >= 2:
            img = img.reduce(f)
    if img.size == (tw, th):
        return img
    return img.resize((tw, th), resample)


class _Pane:
    """一个展示区（Canvas + 该侧的图像状态）。"""

    def __init__(self, owner: "CompareWindow", canvas: tk.Canvas, name: str):
        self.owner = owner
        self.canvas = canvas
        self.name = name
        self.path: Optional[str] = None
        self.image: Optional[Image.Image] = None
        self._q = None              # 1/4 层（懒构建）
        self._half = None           # 1/2 层（懒构建）
        self._small = None          # 适应画布的预览层（懒构建）
        self._small_scale = 1.0
        self.zoom = 1.0
        self.cx = 0.0               # 视口中心（原图像素坐标）
        self.cy = 0.0
        self.item = None
        self.photo = None           # 保活
        self._press = None
        self._drag_ready = False
        self._last_key = None
        self._hq_job = None

    # -- 视图 ---------------------------------------------------------
    def set_image(self, img: Image.Image):
        self.image = img
        self._q = self._half = self._small = None
        self._small_scale = 1.0
        self._last_key = None

    def fit(self):
        if self.image is None:
            return
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 10 or h < 10:
            w, h = 800, 700
        iw, ih = self.image.size
        self.zoom = min(w / max(iw, 1), h / max(ih, 1))
        self.cx, self.cy = iw / 2.0, ih / 2.0
        self.render("best")

    def _level_for(self, cw: int, ch: int):
        """按当前缩放挑采样层：只允许“向下采样”，绝不用低分辨率层放大。

        大图缩放到整图可见时，若直接裁剪原图再降采样，PIL 会做一次
        几百万像素的滤波（实测 0.2~0.5s）；用预先 reduce 出来的小层裁剪，
        像素搬运量降一两个数量级，缩放才跟手。"""
        if self.image is None:
            return None, 1.0
        iw, ih = self.image.size
        fit = min(cw / max(iw, 1), ch / max(ih, 1))
        need = min(1.0, fit * 1.6)
        if need < 0.95 and (self._small is None
                            or abs(self._small_scale - need) > 0.05):
            r = 4 if min(iw, ih) >= 1200 else 2
            base = self._q if self._q is not None else self.image.reduce(r)
            w = max(1, int(iw * need))
            h = max(1, int(ih * need))
            self._small = (base.resize((w, h), Image.Resampling.BOX)
                           if base.size != (w, h) else base)
            self._small_scale = w / iw
        z = self.zoom
        if self._small is not None and z <= self._small_scale * 1.02:
            return self._small, self._small_scale
        if z <= 0.26:                       # 1/4 层
            if self._q is None:
                self._q = self.image.reduce(4)
            return self._q, 0.25
        if z <= 0.52:                       # 1/2 层
            if self._half is None:
                self._half = self.image.reduce(2)
            return self._half, 0.5
        return self.image, 1.0

    def render(self, quality: str = "fast"):
        cv = self.canvas
        if self.image is None:
            cv.delete("all")
            cv.create_text(max(cv.winfo_width() // 2, 60),
                           max(cv.winfo_height() // 2, 60),
                           text="（拖入 / 单击右侧小图）", fill="#4a4a52",
                           font=("Microsoft YaHei UI", 11))
            self.item = None
            return
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 10 or ch < 10:
            return
        src, sc = self._level_for(cw, ch)
        iw, ih = src.size
        z = self.zoom * sc                 # 源图 -> 画布 的缩放比
        key = (id(src), round(z, 5), round(self.cx, 2), round(self.cy, 2),
               cw, ch, quality)
        if key == self._last_key:
            return
        self._last_key = key
        vw, vh = cw / z, ch / z            # 可见区域（源图坐标）
        cx, cy = self.cx * sc, self.cy * sc
        x0 = max(0.0, min(cx - vw / 2, iw - vw)) if vw < iw else \
            (iw - vw) / 2
        y0 = max(0.0, min(cy - vh / 2, ih - vh)) if vh < ih else \
            (ih - vh) / 2
        box = (int(round(x0)), int(round(y0)),
               int(round(min(x0 + vw, iw))), int(round(min(y0 + vh, ih))))
        crop = src.crop(box)
        tw, th = max(1, int(round(crop.width * z))), \
            max(1, int(round(crop.height * z)))
        if (tw, th) != crop.size:
            # 交互中用 BILINEAR（快），停手 200ms 后自动补一帧 LANCZOS（最清晰）
            resample = (Image.Resampling.LANCZOS if quality == "best"
                        else Image.Resampling.BILINEAR)
            crop = _fast_resize(crop, (tw, th), resample)
        self.photo = ImageTk.PhotoImage(crop)
        px = int(round((box[0] - x0) * z))
        py = int(round((box[1] - y0) * z))
        if self.item is None:
            self.item = cv.create_image(px, py, anchor="nw",
                                        image=self.photo)
        else:
            cv.itemconfigure(self.item, image=self.photo)
            cv.coords(self.item, px, py)
        self.cx, self.cy = (x0 + vw / 2.0) / sc, (y0 + vh / 2.0) / sc

    def request_hq(self):
        """停手后补一帧高质量渲染（LANCZOS）。"""
        if self._hq_job is not None:
            return
        try:
            self._hq_job = self.canvas.after(200, self._hq)
        except tk.TclError:
            self._hq_job = None

    def _hq(self):
        self._hq_job = None
        self.render("best")

    def zoom_at(self, sx: float, sy: float, factor: float):
        if self.image is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.image.size
        vw, vh = cw / self.zoom, ch / self.zoom
        x0 = self.cx - vw / 2
        y0 = self.cy - vh / 2
        wx, wy = x0 + sx / self.zoom, y0 + sy / self.zoom   # 光标下的原图点
        nz = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        self.zoom = nz
        self.cx = wx - (sx - cw / 2) / nz
        self.cy = wy - (sy - ch / 2) / nz
        self.cx = max(0.0, min(self.cx, iw))
        self.cy = max(0.0, min(self.cy, ih))
        self.render("fast")
        self.request_hq()

    def pan(self, dx: float, dy: float):
        if self.image is None:
            return
        self.cx -= dx / self.zoom
        self.cy -= dy / self.zoom
        iw, ih = self.image.size
        self.cx = max(0.0, min(self.cx, iw))
        self.cy = max(0.0, min(self.cy, ih))
        self.render("fast")
        self.request_hq()


class CompareWindow:
    """重复图大图对比窗口。"""

    _open: List["CompareWindow"] = []      # 供 Esc 一键全关

    def __init__(self, master: tk.Misc, members: list, start_path: str,
                 dedup=None):
        self.master = master
        self.members = list(members)
        self.dedup = dedup                 # DedupWindow（复选框联动）
        self.paths = [m.path for m in members]
        self.by_path = {m.path: m for m in members}
        self._cache: Dict[str, Image.Image] = {}
        self._cache_order: List[str] = []
        self._drag = None
        self._fullscreen = False
        self._thumb_rows: Dict[str, dict] = {}
        self.active: Optional[str] = None
        self.win = tk.Toplevel(master)
        self.win.title("重复图对比预览")
        self.win.configure(background="#000000")
        self.win.geometry("1500x900")
        self.win.minsize(900, 560)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.win.bind("<Escape>", lambda _e: self._on_escape())
        self.win.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.win.bind("<f>", lambda _e: self.toggle_fullscreen())

        # ---- 布局 ----
        body = tk.Frame(self.win, background="#000000")
        body.pack(fill="both", expand=True, padx=2, pady=2)
        body.columnconfigure(0, weight=1, uniform="pane")
        body.columnconfigure(1, weight=1, uniform="pane")
        body.columnconfigure(2, weight=0)
        body.rowconfigure(0, weight=1)

        self.left = _Pane(self, self._make_canvas(body, 0), "左")
        self.right = _Pane(self, self._make_canvas(body, 1), "右")
        self._make_strip(body)

        self.hint = tk.Label(
            self.win, background="#000000", foreground="#5a5a62",
            anchor="w", font=("Microsoft YaHei UI", 8),
            text="滚轮缩放 · 中键拖动平移 · 双击复位 · 单击小图载入左侧 · "
                 "拖动小图/图片到左右区 · F11 全屏 · Esc 退出全屏/关闭全部预览")
        self.hint.pack(fill="x", side="bottom")

        self._bind_pane(self.left)
        self._bind_pane(self.right)
        self._build_strip_rows()
        self.set_side("left", start_path, fit=True)
        # 右侧默认放组内第二张（若有），方便立即对比
        others = [p for p in self.paths if p != start_path]
        if others:
            self.set_side("right", others[0], fit=True)
        self.set_active("left")
        CompareWindow._open.append(self)
        self.win.after(60, self._first_fit)
        try:
            self.win.focus_force()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _make_canvas(self, parent: tk.Frame, col: int) -> tk.Canvas:
        cv = tk.Canvas(parent, background="#0b0b0d", highlightthickness=2,
                       highlightbackground="#26262c", highlightcolor="#26262c",
                       bd=0)
        cv.grid(row=0, column=col, sticky="nsew", padx=2)
        return cv

    def _make_strip(self, parent: tk.Frame):
        wrap = tk.Frame(parent, background="#000000", width=STRIP_W)
        wrap.grid(row=0, column=2, sticky="ns")
        wrap.grid_propagate(False)
        self.strip_canvas = tk.Canvas(wrap, background="#000000", bd=0,
                                      highlightthickness=0,
                                      width=STRIP_W - 16)
        vsb = tk.Scrollbar(wrap, orient="vertical",
                           command=self.strip_canvas.yview)
        self.strip_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.strip_canvas.pack(side="left", fill="both", expand=True)
        self.strip_inner = tk.Frame(self.strip_canvas, background="#000000")
        self._strip_win = self.strip_canvas.create_window(
            (0, 0), window=self.strip_inner, anchor="nw")
        self.strip_inner.bind(
            "<Configure>", lambda _e: self.strip_canvas.configure(
                scrollregion=self.strip_canvas.bbox("all")))
        for w in (self.strip_canvas, self.strip_inner):
            w.bind("<MouseWheel>", self._on_strip_wheel, add="+")

    def _build_strip_rows(self):
        for m in self.members:
            row = tk.Frame(self.strip_inner, background="#000000")
            row.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=bool(self.dedup and
                                           self.dedup.sel.get(m.path)))
            cb = tk.Checkbutton(
                row, variable=var, background="#000000",
                activebackground="#000000", selectcolor="#222228",
                highlightthickness=0, bd=0,
                command=lambda p=m.path: self._toggle_check(p))
            cb.pack(side="left", anchor="n")
            photo = self._thumb(m.path)
            lbl = tk.Label(row, image=photo, background="#000000", bd=0,
                           highlightthickness=2, highlightbackground="#000000",
                           cursor="hand2")
            lbl.image = photo
            lbl.pack(side="left", anchor="n")
            # 信息列：名称 / 尺寸·体积 / 相似度 / 标记，逐行显示不截断
            info = tk.Frame(row, background="#000000")
            info.pack(side="left", fill="x", expand=True, padx=(4, 2))
            name, l2, l3, l4 = self._strip_info(m)
            labels, colors = [], []
            for text, color, font in (
                    (name, "#dcdce4", ("Microsoft YaHei UI", 8)),
                    (l2, "#9aa2ac", ("Microsoft YaHei UI", 7)),
                    (l3, "#7fa6d1", ("Microsoft YaHei UI", 7)),
                    (l4, "#c8a06a", ("Microsoft YaHei UI", 7))):
                if not text:
                    labels.append(None)
                    colors.append(None)
                    continue
                lb = tk.Label(info, text=text, background="#000000",
                              foreground=color, font=font, anchor="w",
                              justify="left",
                              wraplength=STRIP_W - THUMB_EDGE - 56)
                lb.pack(fill="x")
                labels.append(lb)
                colors.append(color)
            self._thumb_rows[m.path] = {"var": var, "lbl": lbl, "row": row,
                                        "cb": cb, "info": labels,
                                        "colors": colors}
            for w in (lbl, info) + tuple(x for x in labels if x):
                w.bind("<ButtonPress-1>",
                       lambda e, p=m.path: self._strip_press(p, e))
                w.bind("<B1-Motion>", self._strip_motion)
                w.bind("<ButtonRelease-1>", self._strip_release)

    @staticmethod
    def _strip_info(m) -> tuple:
        """(名称, 尺寸·体积, 相似度, 标记) —— 四行，信息完整不截断。"""
        name = os.path.basename(m.path)
        dims = f"{m.w}×{m.h}" if m.w else "尺寸未知"
        line2 = f"{dims} · {human_bytes(m.size)}"
        if m.cos is not None:
            line3 = f"汉明 {m.hamming * 100:.2f}% · 余弦 {m.cos:.4f}"
        else:
            line3 = f"汉明 {m.hamming * 100:.2f}%"
        flags = []
        if m.exact_copy:
            flags.append("字节相同")
        flags.append("已入库" if m.indexed else "未入库")
        if not os.path.exists(m.path):
            flags.append("⚠文件不存在")
        return name, line2, line3, " · ".join(flags)

    # ------------------------------------------------------------------
    # 图像加载与缩略图
    # ------------------------------------------------------------------
    def _load(self, path: str) -> Optional[Image.Image]:
        if path in self._cache:
            self._cache_order.remove(path)
            self._cache_order.append(path)
            return self._cache[path]
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                im.load()
        except Exception:               # noqa: BLE001
            return None
        self._cache[path] = im
        self._cache_order.append(path)
        while len(self._cache_order) > CACHE_MAX:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return im

    def _thumb(self, path: str) -> Optional[ImageTk.PhotoImage]:
        try:
            with Image.open(path) as im:
                im.draft("RGB", (THUMB_EDGE * 4, THUMB_EDGE * 4))
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.load()
            im.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(im)
        except Exception:               # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # 载入 / 活动图
    # ------------------------------------------------------------------
    def set_side(self, side: str, path: str, fit: bool = False):
        pane = self.left if side == "left" else self.right
        img = self._load(path)
        if img is None:
            return
        pane.path = path
        pane.set_image(img)
        if fit:
            pane.fit()
        else:
            pane.render("best")
        self._highlight_strip()
        if side == "left":
            self.set_active("left")

    def set_active(self, side: str):
        self.active = side
        for s, pane in (("left", self.left), ("right", self.right)):
            on = (s == side)
            pane.canvas.configure(
                highlightbackground="#3d8fd1" if on else "#26262c",
                highlightcolor="#3d8fd1" if on else "#26262c")
        self._highlight_strip()

    def _highlight_strip(self):
        act = self.left.path if self.active == "left" else self.right.path
        for path, row in self._thumb_rows.items():
            on = (path == act)
            row["lbl"].configure(highlightbackground="#3d8fd1" if on
                                 else "#000000")
            for lb, base in zip(row.get("info", ()), row.get("colors", ())):
                if lb is not None:
                    lb.configure(foreground="#ffffff" if on else base)

    def _toggle_check(self, path: str):
        if self.dedup is not None:
            self.dedup._toggle(path)          # 与审查窗口同一份勾选状态
            self.sync_checks()

    def sync_checks(self):
        """审查窗口（批量勾选/删除）变化后回灌本窗复选框。"""
        if self.dedup is None:
            return
        for path, row in self._thumb_rows.items():
            want = bool(self.dedup.sel.get(path))
            if row["var"].get() != want:
                row["var"].set(want)

    # ------------------------------------------------------------------
    # 事件绑定
    # ------------------------------------------------------------------
    def _bind_pane(self, pane: _Pane):
        cv = pane.canvas
        cv.bind("<MouseWheel>", lambda e: self._on_wheel(pane, e))
        cv.bind("<ButtonPress-1>", lambda e: self._pane_press(pane, e))
        cv.bind("<B1-Motion>", lambda e: self._pane_motion(pane, e))
        cv.bind("<ButtonRelease-1>", lambda e: self._pane_release(pane, e))
        cv.bind("<ButtonPress-2>", lambda e: self._pan_press(pane, e))
        cv.bind("<B2-Motion>", lambda e: self._pan_motion(pane, e))
        cv.bind("<Double-Button-1>", lambda e: pane.fit())
        cv.bind("<Configure>", lambda _e: pane.render())

    def _on_wheel(self, pane: _Pane, event):
        pane.zoom_at(event.x, event.y, 1.2 if event.delta > 0 else 1 / 1.2)
        self.set_active("left" if pane is self.left else "right")

    def _pan_press(self, pane: _Pane, event):
        pane._press = (event.x, event.y)

    def _pan_motion(self, pane: _Pane, event):
        if not pane._press:
            return
        px, py = pane._press
        pane.pan(event.x - px, event.y - py)
        pane._press = (event.x, event.y)

    def _pane_press(self, pane: _Pane, event):
        pane._press = (event.x, event.y)
        pane._drag_ready = pane.path is not None
        self.set_active("left" if pane is self.left else "right")

    def _pane_motion(self, pane: _Pane, event):
        if not pane._press or not pane._drag_ready:
            return
        px, py = pane._press
        if abs(event.x - px) + abs(event.y - py) < 8:
            return
        self._drag = {"path": pane.path, "origin": pane.name}
        pane._drag_ready = False
        self._update_drag(event)

    def _pane_release(self, pane: _Pane, event):
        if self._drag is not None:
            self._end_drag(event)
        pane._press = None
        pane._drag_ready = False

    # ---- 小图栏拖动 ---------------------------------------------------
    def _strip_press(self, path: str, event):
        # 单击即载入左区；若随后移动超过阈值则转为拖动
        self._drag = {"path": path, "origin": "strip"}
        self.set_active("left")
        self.set_side("left", path)

    def _strip_motion(self, event):
        if self._drag is None:
            return
        self._update_drag(event)

    def _strip_release(self, event):
        if self._drag is not None:
            self._end_drag(event)

    def _on_strip_wheel(self, event):
        self.strip_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    # ---- 拖动浮层与投放 ----------------------------------------------
    def _pane_at(self, x_root: int, y_root: int) -> Optional[_Pane]:
        for pane in (self.left, self.right):
            cv = pane.canvas
            try:
                x0, y0 = cv.winfo_rootx(), cv.winfo_rooty()
            except tk.TclError:
                continue
            if (x0 <= x_root <= x0 + cv.winfo_width()
                    and y0 <= y_root <= y0 + cv.winfo_height()):
                return pane
        return None

    def _update_drag(self, event):
        target = self._pane_at(event.x_root, event.y_root)
        for pane in (self.left, self.right):
            on = pane is target
            pane.canvas.configure(
                highlightbackground="#5fd3c4" if on else
                ("#3d8fd1" if (self.active == ("left" if pane is self.left
                                               else "right")) else "#26262c"),
                highlightcolor="#5fd3c4" if on else "#26262c")
        if self._drag is None:
            return
        if self._drag.get("float") is None:
            try:
                img = self._load(self._drag["path"])
                if img is None:
                    return
                thumb = img.copy()
                thumb.thumbnail((110, 110), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                top = tk.Toplevel(self.win)
                top.overrideredirect(True)
                top.attributes("-topmost", True)
                lbl = tk.Label(top, image=photo, bd=2, relief="solid",
                               background="#000000")
                lbl.image = photo
                lbl.pack()
                self._drag["float"] = top
                self._drag["float_size"] = (thumb.width, thumb.height)
            except Exception:           # noqa: BLE001
                self._drag["float"] = False
        top = self._drag.get("float")
        if top:
            w, h = self._drag.get("float_size", (100, 100))
            top.geometry(f"+{event.x_root + 14}+{event.y_root + 14}")

    def _end_drag(self, event):
        drag = self._drag
        self._drag = None
        if not drag:
            return
        top = drag.get("float")
        if top:
            try:
                top.destroy()
            except tk.TclError:
                pass
        target = self._pane_at(event.x_root, event.y_root)
        if target is not None:
            side = "left" if target is self.left else "right"
            self.set_side(side, drag["path"], fit=True)
            self.set_active(side)
        else:
            self.set_active(self.active or "left")
            self._highlight_strip()
            for pane in (self.left, self.right):
                on = (("left" if pane is self.left else "right")
                      == self.active)
                pane.canvas.configure(
                    highlightbackground="#3d8fd1" if on else "#26262c")

    # ------------------------------------------------------------------
    def _on_escape(self):
        """Esc：全屏时先退出全屏，否则一次性关闭所有对比预览窗。"""
        if self._fullscreen:
            self.toggle_fullscreen()
            return
        CompareWindow.close_all()

    def toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        try:
            self.win.attributes("-fullscreen", self._fullscreen)
        except tk.TclError:
            pass
        self.win.after(120, self._refit_all)

    def _first_fit(self):
        self.left.fit()
        self.right.fit()

    def _refit_all(self):
        for pane in (self.left, self.right):
            if pane.image is not None:
                pane.render()

    def close(self):
        if self._fullscreen:
            self.toggle_fullscreen()
            return
        try:
            if self._drag and self._drag.get("float"):
                self._drag["float"].destroy()
        except Exception:               # noqa: BLE001
            pass
        try:
            CompareWindow._open.remove(self)
        except ValueError:
            pass
        try:
            self.win.destroy()
        except tk.TclError:
            pass

    @classmethod
    def close_all(cls):
        for w in list(cls._open):
            w.close()
