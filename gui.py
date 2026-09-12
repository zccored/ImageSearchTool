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

"""
可视化界面（tkinter，无需额外 GUI 依赖）。

用法： python gui.py

功能：
  1. 自定义图库位置（输入或浏览选择目录）
  2. 一键“扫描图库”：递归遍历文件夹内全部图片，支持多格式
     （jpg/jpeg/png/bmp/tif/tiff/webp/…，可在“建库参数-图片格式”里增删），
     列表显示大小/格式并可缩略图预览，可勾选任意多张图
  3. “索引”：全部入库 / 仅索引勾选的图片 / 增量入库（自动去重）
  4. 检索：选择查询图（库内双击，或浏览外部文件），两级检索出 Top-K 缩略图网格，
     点选查看详情、一键打开原图 / 复制路径 / 导出总览图
  5. 全部参数做成输入控件：两套参数页（建库参数 / 检索参数）随时微调后生效

后台线程执行扫描/建库/查询，界面不卡死；日志区实时输出引擎日志。
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback

import tkinter as tk
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageDraw, ImageTk

from hybrid_search.config import Config
from hybrid_search.engine import HybridEngine
from hybrid_search.io_utils import human_bytes
from hybrid_search.store import IndexFiles
from hybrid_search.thumbs import ThumbCache

# 与 io_utils 保持一致：抬高 Pillow 解压炸弹上限，避免超大图预览直接抛错
Image.MAX_IMAGE_PIXELS = 256 * 1024 * 1024

import peer_launcher as PL      # “切换启动”全栈图库管理器（校验/登记/启动）
from perfwatch import StageProfiler, latest_report  # 阶段性能画像（可选导出）

LOGGER = logging.getLogger("hybrid_search")

# ==========================================================================
# 日志桥：引擎日志 -> 队列 -> 主线程 -> 文本框
# ==========================================================================
class QueueLogHandler(logging.Handler):
    def __init__(self, q: "queue.Queue"):
        super().__init__()
        self.q = q
        fmt = logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S")
        self.setFormatter(fmt)

    def emit(self, record):
        try:
            self.q.put(("log", self.format(record)))
        except Exception:  # noqa: BLE001
            pass


# ==========================================================================
# 缩略图 LRU 缓存（PIL 解码 + 缩略，PhotoImage 需保活引用）
# 巨型图（>30M 像素）不做整张预览，防止单次点选吃数百 MB 内存
# ==========================================================================
class ThumbStore:
    PREVIEW_MAX_PX = 30_000_000       # 不支持 draft 的格式（PNG/WebP…）预览上限
    DRAFT_MAX_PX = 200_000_000        # JPEG 可 draft 降采样解码，上限放宽

    def __init__(self, cap=240):
        self.cap = cap
        self._cache: dict = {}
        self._order: list = []

    def get(self, path: str, edge: int,
            box: tuple | None = None) -> "ImageTk.PhotoImage | None":
        key = (path, edge, box)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        try:
            with Image.open(path) as im:
                w0, h0 = im.size
                px = w0 * h0
                # 先让 JPEG 走 draft 降采样解码（1/2、1/4、1/8），
                # 大图缩略图从 ~100ms 降到几毫秒；PNG/WebP 等 draft 无效
                im.draft("RGB", (max(edge * 4, 128), max(edge * 4, 128)))
                if im.size == (w0, h0) and px > self.PREVIEW_MAX_PX:
                    return None          # 不支持 draft 的超大图跳过预览
                if px > self.DRAFT_MAX_PX:
                    return None
                im.load()
            im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            if box is not None:
                # 瓦片命中：在原图上按比例叠加命中框（红框，左上角角标）
                from PIL import ImageDraw
                d = ImageDraw.Draw(im)
                w, h = im.size
                sx, sy = w / max(w0, 1), h / max(h0, 1)
                x0, y0, x1, y1 = (float(v) for v in box)
                d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                            outline="#ff4040",
                            width=max(2, int(edge / 40)))
            photo = ImageTk.PhotoImage(im)
        except Exception:  # noqa: BLE001
            return None
        self._cache[key] = photo
        self._order.append(key)
        while len(self._order) > self.cap:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return photo


# ==========================================================================
# 后台缩略图加载器
#   实测：24~27MP 的 JPEG 单张解码 ~120ms（熵解码占主导），在主线程解码
#   会让界面直接卡死。这里把“解码 + 落盘缓存”放到工作线程池，主线程只做
#   PIL→PhotoImage 的转换（必须主线程），因此滚动始终流畅。
# ==========================================================================
class ThumbLoader:
    def __init__(self, cache: ThumbCache, workers: int = 4):
        self.cache = cache
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                       thread_name_prefix="thumb")
        self._lock = threading.Lock()
        self._pending: dict = {}        # key -> Future
        self._ready: dict = {}          # key -> PIL.Image（待主线程取走）
        self._failed: set = set()

    def request(self, key: str, path: str) -> bool:
        with self._lock:
            if (key in self._pending or key in self._ready
                    or key in self._failed):
                return False
            self._pending[key] = self.pool.submit(self._work, key, path)
            return True

    def _work(self, key: str, path: str):
        img = self.cache.get(key)          # 命中磁盘缓存：只读几 KB
        if img is None:
            img = self.cache.make(path, key)   # 未命中：解码原图并落盘
        with self._lock:
            self._pending.pop(key, None)
            if img is None:
                self._failed.add(key)
            else:
                self._ready[key] = img

    def drain(self) -> dict:
        with self._lock:
            out, self._ready = self._ready, {}
        return out

    def failed_keys(self) -> set:
        with self._lock:
            return set(self._failed)

    def busy(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self):
        try:
            self.pool.shutdown(wait=False)
        except Exception:               # noqa: BLE001
            pass


# ==========================================================================
# 可滚动容器（结果网格用）
# ==========================================================================
class ScrollFrame(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#1e1e22")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _on_canvas_resize(self, e):
        self.canvas.itemconfigure(self._win, width=e.width)

    def _on_wheel(self, e):
        self.canvas.yview_scroll(-1 * (e.delta // 120), "units")


# ==========================================================================
# 结果格子（缩略图 + 排名标签，可点选）
# ==========================================================================
class ResultTile:
    """结果格子（缩略图 + 排名标签，可点选）。

    关键：`hit_index` 记录它在 last_hits 中的真实下标。缩略图生成失败时用
    占位格子顶替，保证网格与命中列表**一一对应**（历史 bug：跳过失败项导致
    点击选中的图片与画面上的不是同一张）。"""

    def __init__(self, parent, thumb, text, cmd, hit_index: int = -1,
                 placeholder: str | None = None):
        self.hit_index = hit_index
        self.frame = ttk.Frame(parent, padding=2, style="Tile.TFrame")
        if thumb is not None:
            self.btn = tk.Label(self.frame, image=thumb, cursor="hand2",
                                background="#2b2b31", bd=0)
            self.btn.image = thumb
            self.btn.pack()
            clickable = [self.btn]
        else:
            holder = tk.Frame(self.frame, width=128, height=128,
                              background="#2b2b31")
            holder.pack()
            holder.pack_propagate(False)
            self.btn = tk.Label(holder, text=placeholder or "(无预览)",
                                cursor="hand2", background="#2b2b31",
                                foreground="#8a8a8a", justify="center",
                                wraplength=116,
                                font=("Microsoft YaHei UI", 8))
            self.btn.pack(expand=True, fill="both")
            clickable = [self.btn, holder]
        self.label = tk.Label(self.frame, text=text,
                              background="#2b2b31", foreground="#dcdcdc",
                              font=("Microsoft YaHei UI", 8), anchor="w")
        self.label.pack(fill="x")
        for w in clickable + [self.label, self.frame]:
            w.bind("<Button-1>", lambda _e, c=cmd: c())

    def highlight(self, on: bool):
        bg = "#3d5a80" if on else "#2b2b31"
        for w in (self.frame,):
            w.configure(style="TileSel.TFrame" if on else "Tile.TFrame")


# ==========================================================================
# 重复图审查窗口
# ==========================================================================
class DedupWindow:
    """重复图审查：分组列出“一对多”重复，勾选后删除/移动。

    交互约定：
      * ☑ = 将被处理（删除/移动），☐ = 保留；
      * 点击成员行切换勾选；点击组标题行 = 该组“保留最佳、其余全选/全不选”；
      * 删除默认走 Windows 回收站（可还原）；移动保留相对图库根的目录结构。
    """

    def __init__(self, app, report):
        self.app = app
        self.rep = report
        self.sel: dict = {}              # path -> 是否勾选
        self.item_path: dict = {}        # tree item -> path
        self.path_item: dict = {}        # path -> tree item
        self.group_item: dict = {}       # gid -> item
        self.group_of_path: dict = {}    # path -> DupGroup
        self.flat_items: list = []       # 成员行按显示顺序
        self.thumb_done: set = set()
        self._thumb_job = None
        self._poll_job = None
        self._evict_job = None
        self._compares: list = []        # 已打开的大图对比窗
        # ---- 缩略图：磁盘缓存 + 后台解码（见 ThumbLoader 注释）--------
        idx_dir = (os.path.dirname(app.prefix) if app.prefix
                   else os.path.join(app.dir_var.get().strip() or ".",
                                     ".gallery_index"))
        self.cache = ThumbCache(idx_dir, size=96)
        self.loader = ThumbLoader(self.cache, workers=3)
        self._photo: dict = {}           # key -> PhotoImage（保活）
        self._item_key: dict = {}        # item -> key
        self._key_items: dict = {}       # key -> [item, ...]
        self._applied: set = set()       # 已贴上缩略图的 item
        self._requested: set = set()     # 已发起请求的 key
        self._visible: set = set()       # 当前可见 item
        self._placeholder = None
        self._only_selected = False      # 只看含勾选的组

        self.win = tk.Toplevel(app.root)
        self.win.title("查验去重 · 重复图审查")
        self.win.geometry("1280x860")
        self.win.minsize(980, 600)
        try:
            ttk.Style().configure("Dup.Treeview", rowheight=62)
        except tk.TclError:
            pass

        top = ttk.Frame(self.win, padding=8)
        top.pack(fill="x")
        ttk.Label(
            top,
            text=(f"扫描 {report.scanned} 张 · 完全重复 {report.n_exact} 组 / "
                  f"近似重复 {report.n_near} 组 · 涉及 {report.n_images} 张 · "
                  f"可释放 {human_bytes(report.wasted_bytes)} · "
                  f"近似阈值 {report.threshold * 100:.1f}%"
                  + (f" · {len(report.errors)} 张读取失败" if report.errors
                     else "")),
            foreground="#3d8fd1",
            font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        ttk.Label(top,
                  text="☑ = 将被删除/移动；点击行切换勾选，点击组标题=该组“保留最佳”批量切换。"
                       "每组第一张为默认保留候选（像素最多→体积最大→最新）。"
                       "双击任意一行 → 打开大图对比窗（左右各一张，可缩放/拖动对比）。",
                  foreground="#888888",
                  font=("Microsoft YaHei UI", 8)).pack(anchor="w")

        bar = ttk.Frame(self.win, padding=(8, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="每组保留最佳，其余全选",
                   command=self._select_keep_best).pack(side="left")
        ttk.Button(bar, text="仅选完全重复(字节相同)",
                   command=self._select_exact).pack(side="left", padx=6)
        ttk.Button(bar, text="一键选中：MD5相同且未入库",
                   command=self._select_exact_unindexed).pack(side="left")
        ttk.Button(bar, text="全部不选",
                   command=self._select_none).pack(side="left")
        ttk.Button(bar, text="反选",
                   command=self._invert).pack(side="left", padx=6)
        ttk.Label(bar, text="   显示组数:").pack(side="left")
        self.limit_var = tk.StringVar(value="500")
        cb = ttk.Combobox(bar, textvariable=self.limit_var, width=7,
                          state="readonly",
                          values=("100", "500", "1000", "全部"))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self._repopulate())
        ttk.Label(bar, text="（大报告先看空间最大的组，缩小显示范围可加速）",
                  foreground="#888888",
                  font=("Microsoft YaHei UI", 8)).pack(side="left", padx=6)
        self.only_sel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="只看含勾选的组", variable=self.only_sel_var,
                        command=self._on_only_selected).pack(side="left",
                                                             padx=(8, 0))
        self.msg_var = tk.StringVar(value="")
        ttk.Label(self.win, textvariable=self.msg_var, foreground="#ffd28f",
                  font=("Microsoft YaHei UI", 8)).pack(fill="x", padx=10)

        mid = ttk.Frame(self.win, padding=8)
        mid.pack(fill="both", expand=True)
        cols = ("sel", "size", "dims", "md5", "state")
        self.tree = ttk.Treeview(mid, columns=cols, show="tree headings",
                                 style="Dup.Treeview", selectmode="browse")
        self.tree.heading("#0", text="缩略图 / 名称")
        self.tree.heading("sel", text="勾选")
        self.tree.heading("size", text="大小")
        self.tree.heading("dims", text="尺寸")
        self.tree.heading("md5", text="MD5")
        self.tree.heading("state", text="状态")
        self.tree.column("#0", width=430, anchor="w", stretch=True)
        self.tree.column("sel", width=52, anchor="center", stretch=False)
        self.tree.column("size", width=90, anchor="e", stretch=False)
        self.tree.column("dims", width=90, anchor="center", stretch=False)
        self.tree.column("md5", width=90, anchor="center", stretch=False)
        self.tree.column("state", width=230, anchor="w", stretch=False)
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.tag_configure("exact", background="#1c2a20")
        self.tree.tag_configure("near", background="#241f2b")
        self.tree.tag_configure("keep", foreground="#9fe0a8")
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<Double-Button-1>", self._on_double)
        self.tree.bind("<space>", lambda _e: self._toggle_current())
        self.win.bind("<Escape>", lambda _e: self._close_compares())

        bottom = ttk.Frame(self.win, padding=8)
        bottom.pack(fill="x")
        self.sum_var = tk.StringVar(value="已选 0 张")
        ttk.Label(bottom, textvariable=self.sum_var).pack(side="left")
        ttk.Button(bottom, text="关闭", command=self.win.destroy).pack(
            side="right")
        self.btn_move = ttk.Button(bottom, text="移动到…",
                                   command=self._move_selected)
        self.btn_move.pack(side="right", padx=6)
        self.btn_del = ttk.Button(bottom, text="删除选中(回收站)",
                                  command=self._delete_selected)
        self.btn_del.pack(side="right")
        self.sync_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bottom, text="同步从索引移除已删除/移动的条目",
                        variable=self.sync_var).pack(side="right", padx=10)

        self._populate()
        self._refresh_summary()
        # 懒加载：滚动/改变大小后只补可见行的缩略图
        self.tree.bind("<Configure>", lambda _e: self._schedule_thumbs(),
                       add="+")
        self.tree.bind("<MouseWheel>", lambda _e: self._schedule_thumbs(),
                       add="+")
        self.tree.bind("<<TreeviewOpen>>", lambda _e: self._schedule_thumbs(),
                       add="+")
        self.win.bind("<Destroy>", lambda _e: self._on_destroy())
        self._watch_scroll()
        self._schedule_thumbs()

    def _watch_scroll(self):
        """滚动条拖动 / PgDn / 程序化滚动不会发 MouseWheel 事件，
        这里每 250ms 比一次 yview，位置变了就补加载可见行。"""
        if not self._watch_job_alive():
            return
        try:
            pos = tuple(round(v, 4) for v in self.tree.yview())
        except tk.TclError:
            return
        if pos != getattr(self, "_last_yview", None):
            self._last_yview = pos
            self._schedule_thumbs()
        try:
            self._scroll_job = self.win.after(250, self._watch_scroll)
        except tk.TclError:
            self._scroll_job = None

    def _watch_job_alive(self) -> bool:
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def _on_destroy(self):
        """关窗：停掉后台解码线程池，避免线程悬挂。"""
        for attr in ("_thumb_job", "_poll_job", "_evict_job", "_scroll_job"):
            job = getattr(self, attr, None)
            if job:
                try:
                    self.win.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, attr, None)
        self.loader.shutdown()

    # ---- 渲染 --------------------------------------------------------
    def _group_limit(self) -> int:
        """0 = 全部。"""
        v = self.limit_var.get().strip()
        return 0 if v == "全部" else max(1, int(v))

    def _on_only_selected(self):
        self._only_selected = bool(self.only_sel_var.get())
        self._repopulate()
        if self._only_selected:
            n = sum(1 for g in self.rep.groups
                    if any(self.sel.get(m.path) for m in g.members))
            self.msg_var.set(f"只显示含勾选的组：{n} 组"
                             if n else "当前没有任何勾选（先点上方批量勾选按钮）")
        else:
            self.msg_var.set("")

    def _repopulate(self):
        """按“显示组数”重建列表（勾选状态按路径保留）。"""
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.item_path.clear()
        self.path_item.clear()
        self.group_item.clear()
        self.group_of_path.clear()
        self.flat_items = []
        self._item_key.clear()
        self._key_items.clear()
        self._applied.clear()
        self._requested.clear()
        self._photo.clear()
        self._populate()
        self._schedule_thumbs()

    def _populate(self):
        # 组数可能上千（实测 6900 组 / 2.4 万张）：默认只列前 500 组（按可释放
        # 空间降序），缩略图“只解码可见行”，其余交给“显示组数”下拉
        limit = self._group_limit()
        if self._only_selected:
            # 只看含勾选的组（此时忽略“显示组数”上限，否则命中可能落在范围外）
            groups = [g for g in self.rep.groups
                      if any(self.sel.get(m.path) for m in g.members)]
        else:
            groups = (self.rep.groups if limit <= 0
                      else self.rep.groups[:limit])
        open_gids = {g.gid for g in groups[:40]}
        for g in groups:
            kind_txt = ("完全重复" if g.all_exact
                        else ("近似重复" if g.kind == "near" else "含完全重复"))
            head = (f"组 {g.gid} · {kind_txt} · {len(g.members)} 张 · "
                    f"可释放 {human_bytes(g.wasted_bytes)}")
            if g.max_hamming:
                head += f" · 最大汉明 {g.max_hamming * 100:.2f}%"
            if g.min_cos is not None:
                head += f" · 最小余弦 {g.min_cos:.4f}"
            if g.max_hamming > self.rep.threshold * 1.2:
                # 组内差异超过阈值 = 通过“链式传递”并进来的（A~B~C 但 A 与 C 不近），
                # 提示用户逐张看汉明，别整组一键删
                head += " · ⚠链式传递(并非每张都与保留项直接相似)"
            pid = self.tree.insert("", "end", text=head,
                                   open=g.gid in open_gids,
                                   values=("", "", "", "", ""),
                                   tags=(g.kind,))
            self.group_item[g.gid] = pid
            for mi, m in enumerate(g.members):
                self._insert_member(pid, g, m, mi)

    def _insert_member(self, pid: str, g, m, mi: int):
        state = []
        if mi == 0:
            state.append("保留候选")
        if m.exact_copy and mi > 0:
            state.append("字节相同")
        if m.hamming:
            state.append(f"汉明{m.hamming * 100:.2f}%")
        if m.cos is not None:
            state.append(f"余弦{m.cos:.4f}")
        if m.indexed:
            state.append("已入库")
        if not os.path.exists(m.path):
            state.append("⚠文件不存在")
        iid = self.tree.insert(
            pid, "end", text=" " + m.name,
            values=("☑" if self.sel.get(m.path) else "☐",
                    human_bytes(m.size),
                    f"{m.w}×{m.h}" if m.w else "-",
                    (m.md5[:8] or "-"), " ".join(state)),
            tags=("keep",) if mi == 0 else ())
        self.item_path[iid] = m.path
        self.path_item[m.path] = iid
        self.group_of_path[m.path] = g
        self.flat_items.append(iid)
        key = ThumbCache.key_for(m.path, m.md5)
        self._item_key[iid] = key
        self._key_items.setdefault(key, []).append(iid)

    # ---- 缩略图：请求可见行 → 后台解码 → 主线程贴图 → 滑出即释放 -------
    def _schedule_thumbs(self):
        if self._thumb_job is None:
            try:
                self._thumb_job = self.win.after(80, self._request_visible)
            except tk.TclError:
                self._thumb_job = None

    def _visible_range(self) -> tuple:
        """可见行范围。注意：窗口尚未布局时 tree.yview() 会返回 (0,1)
        （看起来“全部可见”），曾导致一次性贴 6000+ 张图、卡 20 秒，
        这里用控件高度做守卫。"""
        n = len(self.flat_items)
        if n == 0:
            return 0, 0
        try:
            h = self.tree.winfo_height()
            y0, y1 = self.tree.yview()
        except tk.TclError:
            return 0, min(n, 60)
        if h < 40 or ((y1 - y0) > 0.999 and n > 200):
            return 0, min(n, 60)
        first = max(0, int(y0 * n) - 12)
        last = min(n, int(y1 * n) + 24)
        if last - first > 400:
            last = first + 400
        return first, last

    def _request_visible(self):
        """只处理可见行：命中磁盘缓存的主线程直接贴图（~1ms），
        未命中的交给后台线程解码（首帧稍慢，但界面不卡）。
        每轮最多处理 40 行，避免一次性阻塞主线程。"""
        self._thumb_job = None
        if not self.flat_items:
            return
        first, last = self._visible_range()
        self._visible = set(self.flat_items[first:last])
        budget = 40
        for i in range(first, last):
            if budget <= 0:
                self._schedule_thumbs()
                break
            iid = self.flat_items[i]
            if iid in self._applied or not self.tree.exists(iid):
                continue
            key = self._item_key.get(iid)
            path = self.item_path.get(iid)
            if not key or not path:
                continue
            img = self.cache.get(key)      # 命中：几 KB 小图，主线程可接受
            if img is not None:
                self._apply(iid, key, img)
                budget -= 1
            elif key not in self._requested:
                self._requested.add(key)
                self.loader.request(key, path)
                budget -= 1
        if self.loader.busy() or self._poll_job is None:
            self._schedule_poll()
        self._schedule_evict()

    def _schedule_poll(self):
        if self._poll_job is None:
            try:
                self._poll_job = self.win.after(50, self._poll_thumbs)
            except tk.TclError:
                self._poll_job = None

    def _poll_thumbs(self):
        self._poll_job = None
        try:
            if not self.win.winfo_exists():
                return
        except tk.TclError:
            return
        ready = self.loader.drain()
        for key, img in ready.items():
            self._apply_key(key, img)
        failed = self.loader.failed_keys()
        if failed:
            for key in failed:
                for iid in self._key_items.get(key, []):
                    if self.tree.exists(iid) and iid not in self._applied:
                        self._apply_placeholder(iid)
        if self.loader.busy() or ready:
            self._schedule_poll()

    def _apply_key(self, key: str, img):
        """贴图：只贴当前可见的行（同一 md5 可能对应几十个副本行，
        若全贴会把“已应用”集合撑爆，也让内存回收失效）。
        没有任何可见行用到时不留 PhotoImage（磁盘缓存里已有，下次秒回）。"""
        photo = ImageTk.PhotoImage(img)
        used = False
        for iid in self._key_items.get(key, []):
            if iid not in self._visible or not self.tree.exists(iid):
                continue
            try:
                self.tree.item(iid, image=photo)
            except tk.TclError:
                continue
            self._applied.add(iid)
            used = True
        if used:
            self._photo[key] = photo

    def _apply(self, iid: str, key: str, img):
        if key not in self._photo:
            self._photo[key] = ImageTk.PhotoImage(img)
        photo = self._photo[key]
        try:
            self.tree.item(iid, image=photo)
        except tk.TclError:
            return
        self._applied.add(iid)

    def _apply_placeholder(self, iid: str):
        if self._placeholder is None:
            im = Image.new("RGB", (96, 96), (38, 38, 44))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 95, 95], outline=(70, 70, 78))
            d.text((10, 42), "无预览", fill=(120, 120, 130))
            self._placeholder = ImageTk.PhotoImage(im)
        try:
            self.tree.item(iid, image=self._placeholder)
        except tk.TclError:
            return
        self._applied.add(iid)

    def _schedule_evict(self):
        """滚动停住 2.5s 后，释放显示范围以外的缩略图（内存回收）。"""
        if self._evict_job is not None:
            try:
                self.win.after_cancel(self._evict_job)
            except tk.TclError:
                pass
        try:
            self._evict_job = self.win.after(2500, self._evict_off_screen)
        except tk.TclError:
            self._evict_job = None

    def _evict_off_screen(self):
        self._evict_job = None
        first, last = self._visible_range()
        # 双保险：上次请求算出的可见集合 ∪ 当前滚动范围，绝不释放屏幕上的图
        keep_items = set(self.flat_items[first:last]) | set(self._visible)
        if not keep_items:
            return
        for iid in list(self._applied):
            if iid in keep_items:
                continue
            if self.tree.exists(iid):
                try:
                    self.tree.item(iid, image="")
                except tk.TclError:
                    pass
            self._applied.discard(iid)
        # 没有任何行在用这张 PhotoImage 就释放它（下次从磁盘缓存秒回）
        alive = {self._item_key[i] for i in self._applied
                 if i in self._item_key}
        for key in list(self._photo):
            if key not in alive:
                self._photo.pop(key, None)

    # ---- 勾选 --------------------------------------------------------
    def _on_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if item in self.item_path:
            self._toggle(self.item_path[item])
        else:
            self._toggle_group(item)

    def _toggle_current(self):
        item = self.tree.focus()
        if item in self.item_path:
            self._toggle(self.item_path[item])
        elif item:
            self._toggle_group(item)

    # ---- 双击：打开大图对比窗 -----------------------------------------
    def _on_double(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if item in self.item_path:
            path = self.item_path[item]
        else:
            children = self.tree.get_children(item)
            if not children:
                return
            path = self.item_path.get(children[0])
        if not path:
            return
        g = self.group_of_path.get(path)
        if g is None:
            return
        from compare_view import CompareWindow
        w = CompareWindow(self.win, g.members, path, dedup=self)
        self._compares.append(w)

    def _close_compares(self):
        from compare_view import CompareWindow
        CompareWindow.close_all()
        self._compares = [w for w in self._compares if w in CompareWindow._open]

    def _sync_compares(self):
        """批量勾选/删除后，把勾选状态回灌到已打开的对比窗。"""
        alive = []
        for w in self._compares:
            try:
                if w.win.winfo_exists():
                    w.sync_checks()
                    alive.append(w)
            except tk.TclError:
                pass
        self._compares = alive

    def _toggle(self, path: str):
        self.sel[path] = not self.sel.get(path, False)
        iid = self.path_item.get(path)
        if iid and self.tree.exists(iid):
            self.tree.set(iid, "sel", "☑" if self.sel[path] else "☐")
        self._refresh_summary()
        self._sync_compares()

    def _toggle_group(self, item: str):
        children = self.tree.get_children(item)
        paths = [self.item_path[c] for c in children if c in self.item_path]
        if len(paths) < 2:
            return
        rest = paths[1:]                      # 第一张为保留候选
        any_on = any(self.sel.get(p) for p in rest)
        for p in rest:
            self.sel[p] = not any_on
            iid = self.path_item.get(p)
            if iid and self.tree.exists(iid):
                self.tree.set(iid, "sel", "☑" if self.sel[p] else "☐")
        self._refresh_summary()

    def _select_keep_best(self):
        for g in self.rep.groups:
            for mi, m in enumerate(g.members):
                self.sel[m.path] = mi > 0
        self._sync_all_cells()

    def _select_exact(self):
        for g in self.rep.groups:
            for mi, m in enumerate(g.members):
                self.sel[m.path] = (mi > 0 and m.exact_copy)
        self._sync_all_cells()

    def _select_exact_unindexed(self):
        """一键选中：与保留项 MD5 完全相同、且尚未入库的副本。

        语义：入库的那一份是“正本”，未入库的字节级副本是纯冗余，删除不损失
        任何内容；与保留项不同（近似变体）的一律不选。
        命中往往分散在几千组里（实测 2316 组），因此选中后自动切到
        “只看含勾选的组”，否则默认只显示 500 组，用户会以为按钮没反应。"""
        n = 0
        for g in self.rep.groups:
            for mi, m in enumerate(g.members):
                hit = bool(mi > 0 and m.exact_copy and not m.indexed)
                self.sel[m.path] = hit
                n += int(hit)
        if n == 0:
            messagebox.showinfo(
                "一键选中", "没有“MD5 完全相同且未入库”的副本。\n\n"
                            "（可能这些字节级副本都已入库，或本批重复只有近似变体）",
                parent=self.win)
            return
        self.only_sel_var.set(True)
        self._only_selected = True
        self._repopulate()
        groups = sum(1 for g in self.rep.groups
                     if any(self.sel.get(m.path) for m in g.members))
        total = sum(os.path.getsize(m.path) for g in self.rep.groups
                    for m in g.members
                    if self.sel.get(m.path) and os.path.exists(m.path))
        self.msg_var.set(
            f"已选中 {n} 张“MD5 相同且未入库”的副本，分布在 {groups} 组"
            f"（约 {human_bytes(total)}）；已自动切换到“只看含勾选的组”，"
            f"取消勾选该开关可回到全部组。")

    def _select_none(self):
        for p in list(self.sel):
            self.sel[p] = False
        self._sync_all_cells()

    def _invert(self):
        for p in list(self.sel):
            self.sel[p] = not self.sel[p]
        self._sync_all_cells()

    def _sync_all_cells(self):
        for p, iid in self.path_item.items():
            if self.tree.exists(iid):
                self.tree.set(iid, "sel", "☑" if self.sel.get(p) else "☐")
        self._refresh_summary()
        self._sync_compares()

    def _selected(self) -> list:
        return [p for p, on in self.sel.items() if on and os.path.exists(p)]

    def _refresh_summary(self):
        paths = self._selected()
        total = 0
        for p in paths:
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
        shown = len(self.path_item)
        extra = (f"（含 {len(paths) - shown} 张未显示组）"
                 if len(paths) > shown else "")
        self.sum_var.set(f"已选 {len(paths)} 张 · 合计 "
                         f"{human_bytes(total)}{extra}")
        state = "normal" if paths else "disabled"
        for b in (self.btn_del, self.btn_move):
            b.configure(state=state)

    # ---- 动作 --------------------------------------------------------
    def _delete_selected(self):
        from hybrid_search import dedup as DD
        paths = self._selected()
        if not paths:
            return
        total = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
        sample = "\n".join("  " + os.path.basename(p) for p in paths[:8])
        more = f"\n  …等 {len(paths)} 张" if len(paths) > 8 else ""
        # 风险提示：链式组（成员并非都与保留项直接相似）单独计数
        sel_set = set(paths)
        chained = sum(1 for g in self.rep.groups
                      if g.max_hamming > self.rep.threshold * 1.2
                      and any(m.path in sel_set for m in g.members))
        warn = ""
        if chained:
            warn += (f"\n⚠ 其中 {chained} 组属于“链式传递”组：组内并非每张都与保留项"
                     f"直接相似，建议逐张核对汉明距离后再删。\n")
        if len(paths) > 500:
            warn += f"\n⚠ 本次将删除 {len(paths)} 张，数量较大，建议先小批量试一次。\n"
        if not messagebox.askokcancel(
                "删除到回收站",
                f"将把 {len(paths)} 张图移入 Windows 回收站（可还原）：\n\n"
                f"{sample}{more}\n\n合计 {human_bytes(total)}。{warn}\n继续？",
                parent=self.win):
            return
        ok, bad = DD.recycle_paths(paths)
        if not ok:
            messagebox.showerror("删除失败", f"没有文件被删除。\n"
                                             f"首个原因：{bad[0][1] if bad else '未知'}",
                                 parent=self.win)
            return
        self._after_removed(ok, "已删除")
        if bad:
            messagebox.showwarning(
                "部分失败", f"成功 {len(ok)} 张，失败 {len(bad)} 张：\n"
                            + "\n".join(f"  {os.path.basename(p)}: {r}"
                                        for p, r in bad[:5]), parent=self.win)
        self._sync_index(ok)

    def _move_selected(self):
        from hybrid_search import dedup as DD
        paths = self._selected()
        if not paths:
            return
        dest = filedialog.askdirectory(
            title="选择目标图库位置（保留相对目录结构）", parent=self.win)
        if not dest:
            return
        base = self.app.dir_var.get().strip() or None
        ok, bad = DD.move_paths(paths, dest, base_root=base)
        if not ok:
            messagebox.showerror("移动失败", f"没有文件被移动。\n"
                                             f"首个原因：{bad[0][1] if bad else '未知'}",
                                 parent=self.win)
            return
        self._after_removed(ok, f"已移动到 {dest}")
        if bad:
            messagebox.showwarning(
                "部分失败", f"成功 {len(ok)} 张，失败 {len(bad)} 张：\n"
                            + "\n".join(f"  {os.path.basename(p)}: {r}"
                                        for p, r in bad[:5]), parent=self.win)
        self._sync_index(ok)

    def _after_removed(self, removed: list, what: str):
        rm = set(removed)
        rm_items = set()
        for p in removed:
            self.sel.pop(p, None)
            iid = self.path_item.pop(p, None)
            if iid:
                rm_items.add(iid)
                if self.tree.exists(iid):
                    self.tree.delete(iid)
            g = self.group_of_path.pop(p, None)
            if g is not None:
                g.members[:] = [m for m in g.members if m.path != p]
        if rm_items:
            self.flat_items = [i for i in self.flat_items if i not in rm_items]
            for iid in rm_items:
                self._applied.discard(iid)
                key = self._item_key.pop(iid, None)
                if key and key in self._key_items:
                    self._key_items[key] = [i for i in self._key_items[key]
                                            if i != iid]
            self._evict_off_screen()
        for g in list(self.rep.groups):
            pid = self.group_item.get(g.gid)
            if len(g.members) < 2:
                if pid and self.tree.exists(pid):
                    self.tree.delete(pid)
                self.group_item.pop(g.gid, None)
                self.rep.groups.remove(g)
            elif pid and self.tree.exists(pid):
                self.tree.item(pid, text=(
                    f"组 {g.gid} · "
                    f"{'完全重复' if g.all_exact else '近似重复'} · "
                    f"{len(g.members)} 张 · 可释放 {human_bytes(g.wasted_bytes)}"))
        # 同步主界面列表
        keep_list = [p for p in self.app.all_images if p not in rm]
        self.app.all_images = keep_list
        self.app.indexed_set -= {os.path.normcase(os.path.abspath(p))
                                 for p in rm}
        self.app._log(f"查验去重：{what} {len(removed)} 张"
                      f"（剩余重复组 {len(self.rep.groups)}）")
        self._refresh_summary()
        self._sync_compares()
        if not self.rep.groups:
            self.win.destroy()
            messagebox.showinfo("查验去重", f"{what}后已无重复图。")

    def _sync_index(self, removed: list):
        if not self.sync_var.get() or not removed:
            return
        try:
            from hybrid_search.store import prune
            from hybrid_search import tile_index as TI
            self.app._drop_engines("索引剔除前释放内存", silent=True)
            prefixes = []
            if self.app.prefix and os.path.exists(self.app.prefix + ".meta.json"):
                prefixes.append(self.app.prefix)
            tp = TI.tiles_prefix_of(self.app.prefix) if self.app.prefix else ""
            if tp and os.path.exists(tp + ".meta.json"):
                prefixes.append(tp)
            for p in prefixes:
                r = prune(p, removed)
                if r["removed"]:
                    self.app._log(f"索引同步：{os.path.basename(p)}.* 剔除 "
                                  f"{r['removed']} 条（剩 {r['kept']} 条）")
        except Exception as e:  # noqa: BLE001
            self.app._log(f"索引同步失败（可稍后重建/增量刷新）：{e}")


# ==========================================================================
# 主程序
# ==========================================================================
class App:
    W = 1500
    H = 940

    def __init__(self, root: tk.Tk, auto_handoff: str = None):
        self.root = root
        self.auto_handoff = auto_handoff          # 交接文件（自动增量模式）
        root.title("二值法粗筛 + ResNet精排 · 混合图库检索 (可视化)")
        root.geometry(f"{self.W}x{self.H}")
        root.minsize(1180, 720)

        self.q: "queue.Queue" = queue.Queue()
        self.busy = False
        self.worker: threading.Thread | None = None
        self._peer_ok = False               # 切换目标（全栈管理器）校验通过？
        self._peer_proc = None              # 最近一次切换启动的进程句柄
        self.search_mode_var = tk.StringVar(value="full")   # 检索模式：full/tiles/hybrid
        self._hybrid_warned = False         # 混合模式性能警告是否已提示过（会话内一次）
        self.thumbs = ThumbStore()
        self.all_images: list = []          # 扫描到的图片（绝对路径）
        self.selection: list = []           # 列表里勾选的行
        self.indexed_set: set = set()       # 当前索引前缀下已入库路径
        self.prefix: str = ""
        self.last_hits: list = []           # 最近一次检索结果 [(rank,path,fine,coarse)]
        self.sel_tile: ResultTile | None = None
        self.tiles: list = []
        self._tile_photos: list = []        # 保活
        self.last_query: str = ""

        # ---- 索引引擎缓存 ----
        # 每次搜图都新建引擎会重新加载整个索引：444k 瓦片库实测 3.6s、RSS+1.26GB，
        # 用户观感是“搜图后加载索引卡 4 秒”。这里按 (前缀, 关键参数) 缓存引擎，
        # 索引被改写/参数变化/手动释放时失效；最多保留两套（整图 + 瓦片）。
        self._eng_lock = threading.Lock()
        self._eng_cache: dict = {}          # prefix -> (签名, HybridEngine)
        # ---- 性能图导出开关（默认关；开启时弹“有性能损耗”提示）----
        self.perf_build_var = tk.BooleanVar(value=False)
        self.perf_search_var = tk.BooleanVar(value=False)
        self._perf_ok: set = set()          # 已确认开启过的开关 key
        self._last_perf_report = ""

        # ---- 日志 ----
        handler = QueueLogHandler(self.q)
        logging.getLogger("hybrid_search").addHandler(handler)
        logging.getLogger("hybrid_search").setLevel(logging.INFO)

        self._build_style()
        self._build_ui()
        # 阶段进度渲染状态
        self._prog_phase = None
        self._prog_t0 = time.monotonic()
        self._prog_done0 = 0
        root.after(33, self._pump)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if auto_handoff:
            # 自动交接模式：界面正常构建后延迟启动（等窗口与日志就绪）
            root.after(800, self._auto_handoff_start)

    # ------------------------------------------------------------------
    # 自动交接：处理 img_server 交接文件并增量建库（GUI 全程可视）
    # ------------------------------------------------------------------
    def _auto_handoff_start(self):
        import os as _os
        from hybrid_search import handoff as H

        p = self.auto_handoff
        if not _os.path.isfile(p):
            self._log(f"【自动交接】文件不存在：{p}")
            messagebox.showerror("自动交接", f"交接文件不存在：\n{p}")
            return
        try:
            req = H.load_request(p)
        except Exception as e:                 # noqa: BLE001
            messagebox.showerror("自动交接", f"交接文件无效：{e}")
            return
        roots = [r["path"] for r in req.get("roots", [])]
        if not roots:
            messagebox.showerror("自动交接", "请求中缺少增量根目录(roots)")
            return
        explicit_prefix = req.get("prefix") or ""
        loc0 = None if explicit_prefix else H.locate_gallery_root(roots[0])
        prefix = (explicit_prefix or (loc0["prefix"] if loc0 else _os.path.join(
            roots[0], ".gallery_index", "gallery")))
        self.prefix = prefix
        if roots:
            self.dir_var.set(roots[0])
        self._log(f"【自动交接】收到 img_server 请求：{req.get('request_id', '')}")
        for r in roots:
            self._log(f"    增量来源: {r}")
        if loc0 and loc0["root"] != roots[0]:
            self._log(f"    自动定位图库根: {roots[0]} -> {loc0['root']}"
                      "（子目录自身无索引，增量并入上级图库索引）")
        self._log(f"    索引前缀 : {prefix}")
        self._log("开始校验并增量建库（路径+MD5 去重，重复内容自动跳过）…")
        self._set_busy(True, "自动交接：校验并增量建库中…")
        threading.Thread(target=self._auto_handoff_worker, args=(p,),
                         daemon=True).start()

    def _auto_handoff_worker(self, req_path: str):
        from hybrid_search import handoff as H

        def cb(done, total, phase):
            self.q.put(("progress", (done, total, phase)))

        self.q.put(("progress", (0, 1, "fused")))
        try:
            result = H.process_request_file(req_path, progress=cb)
            self.q.put(("handoff_done", result))
        except Exception as e:                 # noqa: BLE001
            import traceback as _tb
            self.q.put(("error", f"自动交接失败：{e}\n{_tb.format_exc()}"))

    def _on_handoff_done(self, result: dict):
        self.progress.configure(value=0)
        self._prog_phase = None
        ok = result.get("ok")
        added = result.get("total_added", 0)
        secs = result.get("total_secs", 0)
        prefix = result.get("prefix", self.prefix)
        self.prefix = prefix
        msg = (f"自动增量完成：新增 {added} 张，耗时 {secs}s -> {prefix}"
               if ok else "自动增量完成但存在错误（见日志/result 文件）")
        self._set_busy(False, msg)
        self._log("【自动交接】" + msg)
        for nt in result.get("notices", []):
            self._log("    [提示] " + nt)
        self._log(f"    明细文件: {result.get('request_id', '')}")
        # 静默刷新已索引标记（自动模式不弹“请先扫描图库”）
        try:
            self._load_indexed_set()
            self._apply_indexed_tag()
            if self.all_images:
                self.scan_info_var.set(
                    f"共 {len(self.all_images)} 张（自动交接后）")
        except Exception:  # noqa: BLE001 —— 刷新失败不影响主流程
            pass

    # ------------------------------------------------------------------
    # 关闭窗口：提示并销毁（后台任务均为 daemon 线程，随进程结束自动终止）
    # ------------------------------------------------------------------
    def _on_close(self):
        if self.busy:
            self._log("后台任务仍在运行：窗口关闭后任务将被中断（不写入部分索引）")
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # 处理过程可视化面板（左下角：24fps 实时帧绘制）
    # ------------------------------------------------------------------
    def _build_viz_panel(self, pane) -> None:
        box = ttk.LabelFrame(pane, text="处理过程可视化 · 24fps",
                             width=320)
        pane.add(box, weight=0)
        self.viz_box = box
        self.viz_status_var = tk.StringVar(value="等待索引任务…")
        ttk.Label(box, textvariable=self.viz_status_var,
                  foreground="#9aa7b0", font=("Microsoft YaHei UI", 8),
                  anchor="w").pack(side="bottom", fill="x", padx=6, pady=2)
        self.viz_canvas = tk.Canvas(box, width=296, height=150,
                                    background="#0d1117",
                                    highlightthickness=0)
        self.viz_canvas.pack(fill="both", expand=True, padx=4, pady=(4, 0))
        # 占位文案 + 帧图像锚点
        self.viz_placeholder = self.viz_canvas.create_text(
            148, 62, text="粗筛 / ResNet 处理画面将在这里实时绘制\n"
                          "（开始建索引后自动出现，24fps 采样刷新）",
            fill="#5a6a75", font=("Microsoft YaHei UI", 9), justify="center")
        # 可视化帧队列：按阶段分开排队（粗筛帧先播完，随后自动切换播放
        # ResNet 采样象限帧，保持“①完成后原位覆盖②”的顺序观感）；
        # worker 线程只入队；主线程 pump 每 tick 取一帧绘制。
        # 容量=96 帧≈4s 回放窗口：快速任务(几百张)可完整回放两段动画，
        # 超大图库丢最旧帧时也始终展示“最近正在处理”的画面。
        self._viz_queue = {"coarse": deque(maxlen=96),
                           "fine": deque(maxlen=96)}
        self._viz_last_draw = 0.0
        self._viz_phase = None
        self._viz_frames = 0
        self._viz_photo = None

    # 可视化帧回调（engine worker 线程里被调用：只入队，零重量）
    def _viz_cb(self):
        def cb(_path, data, phase):
            try:
                self.q.put(("viz", (phase, data)))
            except Exception:  # noqa: BLE001
                pass
        return cb

    def _maybe_draw_viz(self):
        """24fps 节流：每 tick 最多取一帧绘制；帧率超限时自然丢弃，
        绘制绝不拖慢主流程。粗筛帧队列播完前不切 ResNet 帧（顺序观感）。

        任务结束（_viz_fast）后进入**快放**：跳帧只画最后一帧，让排队帧在
        1 秒内播完——否则用户会以为“ResNet 还在跑、输出被中断了”。"""
        fast = bool(getattr(self, "_viz_fast", False))
        q = (self._viz_queue["coarse"] or self._viz_queue["fine"])
        if not q:
            if fast:
                self._viz_fast = False
                self.viz_status_var.set(
                    f"✅ 可视化回放结束（累计 {self._viz_frames} 帧）· 建库已完成")
            return
        now = time.monotonic()
        if now - self._viz_last_draw < (1.0 / 60.0 if fast else 1.0 / 24.0):
            return
        self._viz_last_draw = now
        if fast:                      # 快放：丢弃中间帧，只绘制最后一帧
            for _ in range(15):
                q = (self._viz_queue["coarse"] or self._viz_queue["fine"])
                if not q or len(q) <= 1:
                    break
                q.popleft()
        if self._viz_queue["coarse"]:
            phase, data = self._viz_queue["coarse"].popleft()
        else:
            phase, data = self._viz_queue["fine"].popleft()
        try:
            self._draw_viz(phase, data)
        except Exception as e:  # noqa: BLE001 —— 可视化失败绝不致命
            self._log(f"可视化绘制跳过: {e}")

    def _draw_viz(self, phase: str, data) -> None:
        """粗筛：64×64 二值激活点阵（荧光绿点）；精排：16×16 中心采样象限
        （原图彩色马赛克）。两者复用同一画布，后者自然覆盖前者。"""
        import numpy as np
        from PIL import Image, ImageTk

        if phase != self._viz_phase:          # 阶段切换：帧计数清零，覆盖绘制
            self._viz_phase = phase
            self._viz_frames = 0
        self._viz_frames += 1

        cw = max(self.viz_canvas.winfo_width(), 100)
        ch = max(self.viz_canvas.winfo_height(), 100)
        if phase == "coarse":
            n = int(np.sqrt(data.size)) if data.ndim == 1 else data.shape[0]
            cell = max(1, min(cw // n, ch // n))
            up = np.repeat(np.repeat((data > 127), cell, axis=0), cell, axis=1)
            rgb = np.zeros((up.shape[0], up.shape[1], 3), dtype=np.uint8)
            rgb[...] = (10, 15, 22)                     # 底色
            rgb[up] = (140, 255, 140)                   # 激活点 = “散点”
        else:                                            # fine 采样象限
            n = data.shape[0]
            cell = max(1, min(cw // n, ch // n))
            rgb = np.repeat(np.repeat(data, cell, axis=0), cell, axis=1)
        img = Image.fromarray(rgb)
        self._viz_photo = ImageTk.PhotoImage(img)
        self.viz_canvas.delete("vizimg")
        self.viz_canvas.create_image(cw // 2, ch // 2, image=self._viz_photo,
                                     anchor="center", tags="vizimg")
        self.viz_canvas.itemconfigure(self.viz_placeholder, state="hidden")
        label = "① 粗筛·二值点阵" if phase == "coarse" else "② ResNet·采样象限"
        self.viz_status_var.set(f"{label} · 帧 #{self._viz_frames}（24fps）")

    # ------------------------------------------------------------------
    # 样式
    # ------------------------------------------------------------------
    def _build_style(self):
        st = ttk.Style()
        if "vista" in st.theme_names():
            st.theme_use("vista")
        st.configure("Tile.TFrame", background="#2b2b31")
        st.configure("TileSel.TFrame", background="#3d5a80")
        st.configure("TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))
        st.configure("TButton", padding=(10, 3))

    # ------------------------------------------------------------------
    # 界面骨架
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(8, 6, 8, 4))
        top.pack(fill="x")
        self._build_toolbar(top)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=2)

        # 左：参数页
        left = ttk.Frame(body, width=390)
        body.add(left, weight=0)
        self._build_param_pages(left)

        # 右：图片列表 / 检索结果 + 日志
        right = ttk.Panedwindow(body, orient="vertical")
        body.add(right, weight=1)
        upper = ttk.Frame(right)
        right.add(upper, weight=3)
        self._build_upper(upper)

        # 底部：左侧“处理过程可视化”(24fps 实时) + 右侧运行日志
        lower_pane = ttk.Panedwindow(right, orient="horizontal")
        right.add(lower_pane, weight=1)
        self._build_viz_panel(lower_pane)
        lower = ttk.LabelFrame(lower_pane, text="运行日志")
        lower_pane.add(lower, weight=1)
        self.log_txt = tk.Text(lower, height=7, state="disabled",
                               font=("Consolas", 9), wrap="none",
                               background="#101014", foreground="#c8c8cc",
                               insertbackground="#c8c8cc")
        self.log_txt.pack(fill="both", expand=True, padx=4, pady=4)

        # 状态栏
        bar = ttk.Frame(self.root, padding=(8, 2))
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="就绪：先选图库目录 → 扫描 → 建索引 → 查询")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(
            side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, length=220, mode="determinate")
        self.progress.pack(side="right", padx=6)

    # ------------------------------------------------------------------
    # 顶部工具栏：图库目录 + 扫描/建索引动作
    # ------------------------------------------------------------------
    def _build_toolbar(self, parent: ttk.Frame):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text="图库目录:").pack(side="left")
        self.dir_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.dir_var, width=52).pack(
            side="left", padx=4, fill="x", expand=True)
        ttk.Button(row, text="浏览…", command=self._pick_dir).pack(side="left")
        self.btn_scan = ttk.Button(row, text="扫描图库",
                                   command=self._scan_gallery)
        self.btn_scan.pack(side="left", padx=6)

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", pady=(4, 0))
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="包含子目录", variable=self.recursive_var,
                        command=self._on_scan_options).pack(side="left")
        self.verify_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="校验可解码(慢)", variable=self.verify_var,
                        command=self._on_scan_options).pack(side="left", padx=8)
        self.scan_info_var = tk.StringVar(value="尚未扫描")
        ttk.Label(row2, textvariable=self.scan_info_var,
                  foreground="#888888").pack(side="left", padx=10)

        row3 = ttk.Frame(parent)
        row3.pack(fill="x", pady=(4, 0))
        self.btn_build_all = ttk.Button(row3, text="① 全部入库并建索引",
                                        command=lambda: self._index_all())
        self.btn_build_sel = ttk.Button(row3, text="② 仅索引勾选的图片",
                                        command=lambda: self._index_selected())
        self.btn_add_new = ttk.Button(row3, text="③ 增量入库(自动去重新图)",
                                      command=lambda: self._add_new())
        self.btn_dedup = ttk.Button(row3, text="查验去重(找重复图，实验功能，慎开)",
                                    command=self._dedup_check)
        self.btn_refresh_idx = ttk.Button(row3, text="刷新已索引标记",
                                          command=self._refresh_indexed_state)
        for b in (self.btn_build_all, self.btn_build_sel, self.btn_add_new,
                  self.btn_dedup, self.btn_refresh_idx):
            b.pack(side="left", padx=(0, 8))

        # 第 3.5 行：子图(局部/瓦片)索引 —— 大图切 512px+25% 重叠瓦片入库
        row_t = ttk.Frame(parent)
        row_t.pack(fill="x", pady=(4, 0))
        self.tiles_state_var = tk.StringVar(value="子图索引：未构建（切 512px 瓦片）")
        ttk.Label(row_t, textvariable=self.tiles_state_var,
                  foreground="#888888").pack(side="left", padx=(0, 10))
        self.btn_tiles = ttk.Button(row_t, text="④ 子图索引(512 切块·建/增量)",
                                    command=self._tiles_index)
        self.btn_tiles.pack(side="left")
        # 性能图 / 内存：报告默认关闭（参数页勾选），这里只做查看与手动释放
        self.btn_perf_open = ttk.Button(row_t, text="📈 最近性能图",
                                        command=self._open_last_perf,
                                        state="normal" if latest_report()
                                        else "disabled")
        self.btn_perf_open.pack(side="left", padx=(10, 6))
        ttk.Button(row_t, text="🧹 释放索引内存",
                   command=lambda: self._drop_engines("手动释放")).pack(side="left")
        ttk.Button(row_t, text="⏩ 优化索引存储",
                   command=self._compact_index).pack(side="left", padx=(6, 0))

        # 第 4 行：切换启动全栈图库管理器（建库中禁用；目标缺失/哈希不符禁用）
        row4 = ttk.Frame(parent)
        row4.pack(fill="x", pady=(4, 0))
        self.peer_state_lbl = ttk.Label(row4, textvariable=None,
                                        foreground="#666666")
        self.peer_state_var = tk.StringVar(value="切换目标校验中…")
        self.peer_state_lbl.configure(textvariable=self.peer_state_var)
        self.peer_state_lbl.pack(side="left", padx=(0, 10))
        self.btn_peer_switch = ttk.Button(row4, text="⇄ 切换启动：全栈图库管理器",
                                          command=self._on_peer_switch)
        self.btn_peer_switch.pack(side="left", padx=(0, 6))
        self.btn_peer_inspect = ttk.Button(row4, text="校验/信任…",
                                           command=self._on_peer_inspect)
        self.btn_peer_inspect.pack(side="left")
        self._peer_refresh_state()

    # ------------------------------------------------------------------
    # 切换启动：全栈图库管理器（peer_launcher，见同目录模块注释）
    #   * 建库(busy)期间禁用；切换目标不存在/未登记/内容哈希不符 → 禁用
    #   * 校验通过并确认后：独立进程启动对方 main.py，短暂探测存活，
    #     成功则关闭本程序（两套程序互切闭环）
    # ------------------------------------------------------------------
    def _peer_refresh_state(self):
        st = PL.check()
        self._peer_ok = bool(st["ok"])
        self._peer_apply_state()
        if self._peer_ok:
            self.peer_state_var.set("切换目标就绪，可切换")
            self.peer_state_lbl.configure(foreground="#1a7f37")
        else:
            short = {PL.ST_MISSING: "目标不存在（本包独立分发时禁用）",
                     PL.ST_UNREGISTERED: "目标未登记，请点“校验/信任…”",
                     PL.ST_MISMATCH: "目标哈希不一致，已禁用（防篡改）"}
            self.peer_state_var.set("切换不可用：" +
                                    short.get(st["code"], st["code"]))
            self.peer_state_lbl.configure(foreground="#b35900")

    def _peer_apply_state(self):
        state = "normal" if (self._peer_ok and not self.busy) else "disabled"
        for b in (self.btn_peer_switch, self.btn_peer_inspect):
            try:
                b.configure(state=state)
            except tk.TclError:
                pass

    def _on_peer_inspect(self):
        """查看校验详情；目标存在但未登记/哈希不符时，允许人工确认后登记。"""
        st = PL.check()
        if st["ok"]:
            messagebox.showinfo("切换启动校验", st["reason"])
            return
        if st["code"] == PL.ST_MISSING:
            messagebox.showwarning("切换启动校验", st["reason"])
            return
        sure = messagebox.askyesno(
            "信任并登记？",
            st["reason"]
            + "\n\n请人工确认该文件来源可信、内容未被篡改或植入后门。\n"
              "确认后将把当前哈希写入本包 peer_manifest.json（登记留痕）。\n\n"
              "是否登记并启用切换？")
        if sure:
            r = PL.register()
            if r["ok"]:
                self._log(f"已登记切换目标信任：sha256 {r['sha256'][:16]}…")
                messagebox.showinfo("登记完成", r["reason"])
            else:
                messagebox.showerror("登记失败", r["reason"])
        self._peer_refresh_state()

    def _on_peer_switch(self):
        if self.busy:
            return                       # 建库中：按钮已禁用，此处兜底
        st = PL.check()
        if not st["ok"]:
            self._peer_refresh_state()
            messagebox.showwarning("切换启动", st["reason"])
            return
        if not messagebox.askyesno(
                "切换启动",
                f"即将关闭本程序（图库检索管理器），并启动：\n\n"
                f"{PL.PEER_MAIN}\n\n"
                "若需回到本程序，请稍后手动重新打开。\n继续？"):
            return
        proc, err = PL.launch()
        if err or proc is None:
            messagebox.showerror("启动失败", err or "未知错误")
            self._peer_refresh_state()
            return
        self._peer_proc = proc
        self._status(f"已启动 {os.path.basename(PL.PEER_MAIN)}"
                     f"（pid={proc.pid}），探测存活中…")
        self.root.after(2000, self._peer_proc_check)

    def _peer_proc_check(self):
        proc = self._peer_proc
        if proc is None:
            return
        rc = proc.poll()
        if rc is None:
            # 目标进程存活（2 秒内未崩溃）→ 正常关闭本程序
            self._log(f"切换启动成功（{os.path.basename(PL.PEER_MAIN)} 运行中），"
                      "关闭本程序…")
            self.root.destroy()
            return
        # 目标启动后立即退出：视为失败，保留本程序并给出原因
        self._peer_proc = None
        tail = ""
        lp = os.path.join(os.environ.get("TEMP") or ".", "peer_launch_err.log")
        if os.path.isfile(lp):
            try:
                with open(lp, "r", encoding="utf-8", errors="replace") as f:
                    tail = f.read()[-500:]
            except OSError:
                pass
        self._log(f"切换启动失败（进程提前退出 rc={rc}）" +
                  (f"：{tail[-300:]}" if tail else ""))
        messagebox.showerror(
            "启动失败",
            f"目标程序启动后立即退出（rc={rc}）。\n"
            "本程序保持打开。" + (f"\n\n{tail[-400:]}" if tail else ""))
        self._peer_refresh_state()

    # ------------------------------------------------------------------
    # 左侧参数页（全部做成输入控件）
    # ------------------------------------------------------------------
    def _build_param_pages(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        self.entries: dict = {}

        page_build = ttk.Frame(nb, padding=8)
        nb.add(page_build, text="建库参数")
        page_search = ttk.Frame(nb, padding=8)
        nb.add(page_search, text="检索参数")

        self._add_int(page_build, "指纹边长(像素,平方)", "coarse_size", 64, 8, 256,
                      tooltip="二值指纹尺寸：64 → 64×64=4096bit/张")
        self._add_int(page_build, "高斯模糊核(奇数)", "blur", 5, 1, 31,
                      tooltip="二值化前的降噪强度")
        self._add_float(page_build, "Hu矩 融合权重", "hu_weight", 0.35, 0.0, 1.0)
        self._add_float(page_build, "指纹 融合权重", "fp_weight", 0.65, 0.0, 1.0)
        self._add_check(page_build, "使用轮廓 Hu 矩特征", "use_hu", True)
        self._add_check(page_build, "使用二值图像指纹特征", "use_fp", True)
        self._add_check(page_build, "白像素过半时取反(白底图库)", "invert_binary", False)

        self._sep(page_build, "ResNet 精排")
        self._add_combo(page_build, "模型", "model",
                        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
        self._add_combo(page_build, "设备", "device", ["auto", "cuda", "cpu"])
        self._add_int(page_build, "批大小(0=自动)", "batch", 0, 0, 512)
        self._add_check(page_build, "GPU 半精度 FP16", "fp16", True)
        self._add_check(page_build, "预构建 ResNet 全库索引(快/占内存)", "store_fine", True)
        self._add_check(page_build, "新索引用侧车 .npy 存储(加载快/省内存)",
                        "fast_load", False)
        self._add_check(page_build, "屏蔽 libpng/iCCP 噪音输出(推荐)",
                        "silence_png_warnings", True)
        self._add_check(page_build, "启用预处理缓存(重复建库更快/占磁盘)",
                        "prep_cache", True)
        self._add_check(page_build, "MD5 内容去重", "dedup", True)
        self._add_int(page_build, "粗筛并行线程(0=自动≤8)", "workers", 0, 0, 64,
                      tooltip="0=自动(不超过8且不超过CPU核数)，1=串行最省内存；\n"
                              "解码在C层释放GIL，多线程可线性提速")
        self._add_int(page_build, "精排解码线程(0=自动≤8)", "decode_workers", 0,
                      0, 64,
                      tooltip="图像读盘+解码+预处理的多线程数；\n"
                              "解码与GPU前向重叠，GPU场景建议4~8")
        self._sep(page_build, "解码并发(建库吞吐调优)")
        self._add_int(page_build, "大图解码并发上限(>12MP)", "big_decode_conc",
                      16, 1, 64,
                      tooltip="同时解码 12MP+ 大图(大 PNG/高分辨率 JPEG)的线程数"
                              "上限。\n每张峰值内存 36MB~100MB+，16GB 内存机建议 "
                              "16~20；\n数值越高 CPU 越满，但内存余量不足会触发"
                              "页回收反而变慢")
        self._add_int(page_build, "瓦片建库·同时解码图数", "tile_decode_slots",
                      18, 1, 64,
                      tooltip="子图(瓦片)索引第一级同时解码的原图数上限，"
                              "配合上面的并发门使用")
        self._add_int(page_build, "瓦片建库·GPU批等待(ms)", "tile_flush_ms",
                      20, 2, 500,
                      tooltip="瓦片不足一批时的最长等待毫秒。\n小(如10)：批更碎"
                              "(1-2行小批唤醒多)；大(如30)：批更整、唤醒更少")
        self._add_int(page_build, "torch推理线程(0=默认)", "torch_threads", 0,
                      0, 128,
                      tooltip="CPU前向线程数。多数机型1线程最快\n"
                              "(实测多线程反而慢)，GPU场景无需设置")
        self._add_text(page_build, "图片格式(逗号分隔)",
                       "extensions", "jpg,jpeg,png,bmp,tif,tiff,webp",
                       tooltip="扫描与建索引支持的扩展名，改完重新扫描生效")
        self._sep(page_build, "性能图导出")
        self._add_perf_flag(page_build, "索引阶段导出性能图(有性能损耗)",
                            "perf_build", self.perf_build_var,
                            tooltip="开启后：本次建库/增量会被采样（0.4s 一行："
                                    "CPU/GPU/内存/磁盘IO + 阶段打点），\n"
                                    "任务结束写 HTML+JSON 到 perf_reports/。"
                                    "采样开销 <1%，但报告占磁盘且长任务会持续增长")

        self._sep(page_search, "检索流程")
        self._add_int(page_search, "粗筛候选数 coarse_k", "coarse_k", 300, 10, 100000)
        self._add_int(page_search, "最终返回数 top_k", "top_k", 10, 1, 100)
        self._add_check(page_search, "剔除查询图自身", "exclude_self", True)
        self._sep(page_search, "索引位置")
        self._add_text(page_search, "索引前缀(留空=自动放图库内)",
                       "prefix", "",
                       tooltip="例如 D:/idx/my_gallery\n留空自动为 <图库目录>/.gallery_index/gallery")
        self._sep(page_search, "性能图导出")
        self._add_perf_flag(page_search, "搜图阶段导出性能图(有性能损耗)",
                            "perf_search", self.perf_search_var,
                            tooltip="开启后：每次检索都会被采样并产出报告，\n"
                                    "含“加载索引/粗筛/精排”阶段耗时与内存曲线；\n"
                                    "用于排查“加载索引慢、内存不释放”一类问题")
        hint = ("提示：与索引相关的几何参数(指纹边长/模糊核/Hu·指纹开关/取反)修改后\n"
                "已建索引会拒绝打开，需重新“全部入库”重建（防错乱）；\n"
                "权重/候选数/top_k/剔除自身可随时微调。")
        ttk.Label(page_search, text=hint, foreground="#999999", justify="left",
                  font=("Microsoft YaHei UI", 8)).pack(anchor="w", pady=6)

    # ---------- 参数控件工厂 ----------
    def _row(self, parent, label: str) -> ttk.Frame:
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=24, anchor="w").pack(side="left")
        return f

    def _sep(self, parent, title: str):
        ttk.Label(parent, text=title, foreground="#3d8fd1",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(
            anchor="w", pady=(10, 2))

    def _add_int(self, page, label, key, default, lo, hi, tooltip=None):
        f = self._row(page, label)
        var = tk.StringVar(value=str(default))
        e = ttk.Entry(f, textvariable=var, width=12)
        e.pack(side="left")
        self.entries[key] = (var, "int", (lo, hi))
        if tooltip:
            self._tip(e, tooltip)

    def _add_float(self, page, label, key, default, lo, hi, tooltip=None):
        f = self._row(page, label)
        var = tk.StringVar(value=str(default))
        e = ttk.Entry(f, textvariable=var, width=12)
        e.pack(side="left")
        self.entries[key] = (var, "float", (lo, hi))
        if tooltip:
            self._tip(e, tooltip)

    def _add_check(self, page, label, key, default):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(page, text=label, variable=var).pack(
            anchor="w", pady=1)
        self.entries[key] = (var, "bool", None)

    def _add_perf_flag(self, page, label, key, var, tooltip=None):
        """性能图导出开关：勾选时弹一次“有性能损耗”提示，取消则回退为关。"""
        cb = ttk.Checkbutton(page, text=label, variable=var,
                             command=lambda: self._on_perf_toggle(key, var))
        cb.pack(anchor="w", pady=1)
        if tooltip:
            self._tip(cb, tooltip)

    def _on_perf_toggle(self, key: str, var):
        if not var.get() or key in self._perf_ok:
            return
        ok = messagebox.askokcancel(
            "性能图导出 · 性能损耗提示",
            "开启后，任务期间将以 0.4 秒间隔采样并记录：\n"
            "  系统/进程 CPU%、GPU SM%、显存、进程内存(RSS)、磁盘读写 MB/s，\n"
            "并配合阶段打点；任务结束后写入 HTML + JSON 到 perf_reports/。\n\n"
            "• 采样开销很小（微秒级读取，实测影响 <1%），但请注意：\n"
            "• 报告会占磁盘，长时间建库的采样行随时长线性增长（0.4s/行）；\n"
            "• 关掉开关立即停止采样，已生成的报告会保留。\n\n"
            "是否开启？")
        if ok:
            self._perf_ok.add(key)
        else:
            var.set(False)

    def _add_combo(self, page, label, key, options):
        f = self._row(page, label)
        var = tk.StringVar(value=options[0])
        cb = ttk.Combobox(f, textvariable=var, values=options, width=10,
                          state="readonly")
        cb.pack(side="left")
        self.entries[key] = (var, "choice", tuple(options))

    def _add_text(self, page, label, key, default, tooltip=None):
        f = self._row(page, label)
        var = tk.StringVar(value=default)
        e = ttk.Entry(f, textvariable=var, width=22)
        e.pack(side="left", fill="x", expand=True)
        self.entries[key] = (var, "text", None)
        if tooltip:
            self._tip(e, tooltip)

    @staticmethod
    def _tip(widget, text: str):
        tip = tk.Toplevel()
        tip.withdraw()
        tip.overrideredirect(True)
        lbl = tk.Label(tip, text=text, background="#ffffe0", justify="left",
                       relief="solid", borderwidth=1, padx=6, pady=4,
                       font=("Microsoft YaHei UI", 9))
        lbl.pack()

        def show(e):
            try:
                tip.geometry(f"+{e.x_root + 12}+{e.y_root + 14}")
                tip.deiconify()
                tip.lift()
            except tk.TclError:
                pass

        def hide(_e=None):
            try:
                tip.withdraw()
            except tk.TclError:
                pass
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ==================================================================
    # 参数读取（校验失败返回 None 并输出原因）
    # ==================================================================
    def cfg_from_widgets(self, require_prefix=False) -> Config | None:
        cfg = Config()
        # 界面字段名 -> Config 属性名的别名映射
        alias = {"blur": "coarse_blur"}
        try:
            for key, (var, kind, bound) in self.entries.items():
                target = alias.get(key, key)
                if kind == "bool":
                    setattr(cfg, target, bool(var.get()))
                    continue
                raw = str(var.get()).strip()
                if kind == "int":
                    v = int(raw)
                    lo, hi = bound
                    if not (lo <= v <= hi):
                        raise ValueError(f"{key}={v} 超出范围 [{lo},{hi}]")
                    setattr(cfg, target, v)
                elif kind == "float":
                    v = float(raw)
                    lo, hi = bound
                    if not (lo <= v <= hi):
                        raise ValueError(f"{key}={v} 超出范围 [{lo},{hi}]")
                    setattr(cfg, target, v)
                elif kind == "choice":
                    setattr(cfg, target, raw)
                elif kind == "text":
                    if key == "extensions":
                        cfg.extensions = tuple(
                            sorted({e if e.startswith(".") else "." + e
                                    for e in raw.replace("，", ",").split(",")
                                    if e.strip()}))
                    elif key == "prefix":
                        prefix = raw.strip()
                        self.prefix = prefix if prefix else self._auto_prefix()
            if require_prefix and not self.prefix:
                raise ValueError("请先填写“索引前缀”或选择图库目录后自动生成")
            return cfg
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("参数错误", f"参数校验失败：{e}")
            return None

    def _auto_prefix(self) -> str:
        d = self.dir_var.get().strip().rstrip("\\/")
        if not d:
            return ""
        return os.path.join(d, ".gallery_index", "gallery")

    # ==================================================================
    # 工具栏动作：目录 / 扫描 / 索引 / 标记
    # ==================================================================
    def _pick_dir(self):
        d = filedialog.askdirectory(title="选择图库目录")
        if d:
            self.dir_var.set(d)

    def _on_scan_options(self):
        pass  # 选项变化不影响布局，扫描时再读

    def _scan_gallery(self):
        d = self.dir_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showwarning("提示", "请先选择有效的图库目录")
            return
        cfg = self.cfg_from_widgets()
        if cfg is None:
            return
        if not self.prefix:
            self.prefix = self._auto_prefix()
        self._set_busy(True, f"正在扫描 {d} …")
        threading.Thread(target=self._scan_worker,
                         args=(d, cfg, self.recursive_var.get(),
                               self.verify_var.get()), daemon=True).start()

    def _scan_worker(self, d: str, cfg: Config, recursive: bool, verify: bool):
        try:
            from hybrid_search.io_utils import collect_images
            t0 = time.time()
            paths = collect_images(d, cfg.extensions, sort=True)
            if not recursive:
                root = os.path.normcase(os.path.abspath(d)) + os.sep
                paths = [p for p in paths
                         if os.path.sep not in
                         os.path.normcase(os.path.abspath(p))[len(root):]]
            from collections import Counter
            cnt = Counter(os.path.splitext(p)[1].lower() for p in paths)
            broken = 0
            if verify and paths:
                from hybrid_search.io_utils import load_rgb
                ok = []
                for i, p in enumerate(paths):
                    if i % 20 == 0:
                        self.q.put(("progress", (i, len(paths), "verify")))
                    if load_rgb(p) is None:
                        broken += 1
                    else:
                        ok.append(p)
                paths = ok
            self.q.put(("scan_done",
                        {"paths": paths, "count": len(paths), "broken": broken,
                         "elapsed": time.time() - t0,
                         "formats": dict(cnt.most_common())}))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"扫描失败：{e}\n{traceback.format_exc()}"))

    def _index_all(self):
        if not self._require_scan():
            return
        cfg = self.cfg_from_widgets(require_prefix=True)
        if cfg is None:
            return
        self._reset_viz()
        self._run_index("全部图片新建索引", cfg,
                        lambda eng: eng.build(self.prefix,
                                              paths=list(self.all_images),
                                              force=True,
                                              progress=self._prog_cb(),
                                              frame_sink=self._viz_cb()))

    def _index_selected(self):
        sel = self._checked_rows()
        if not sel:
            messagebox.showwarning("提示", "请先在图片列表里勾选要入库的图片")
            return
        cfg = self.cfg_from_widgets(require_prefix=True)
        if cfg is None:
            return
        existing = self._index_exists()
        self._reset_viz()
        self._run_index(
            f"索引勾选的 {len(sel)} 张", cfg,
            (lambda eng: eng.build(self.prefix, paths=sel, force=True,
                                   progress=self._prog_cb(),
                                   frame_sink=self._viz_cb()))
            if not existing else
            (lambda eng: eng.add(self.prefix, paths=sel,
                                 progress=self._prog_cb(),
                                 frame_sink=self._viz_cb())))

    def _add_new(self):
        if not self._require_scan():
            return
        cfg = self.cfg_from_widgets(require_prefix=True)
        if cfg is None:
            return
        if not self._index_exists():
            messagebox.showwarning("提示", "索引不存在，请先“全部入库并建索引”")
            return
        self._reset_viz()
        self._run_index("增量入库（自动去重）", cfg,
                        lambda eng: eng.add(self.prefix,
                                            paths=list(self.all_images),
                                            progress=self._prog_cb(),
                                            frame_sink=self._viz_cb()))

    # ------------------------------------------------------------------
    # 索引引擎缓存：避免每次搜图都重新加载索引（444k 瓦片库 3.6s/1.26GB）
    # ------------------------------------------------------------------
    @staticmethod
    def _eng_sig(cfg: Config, prefix: str) -> tuple:
        """影响已加载索引语义/特征的参数 —— 变则缓存失效。"""
        return (prefix, cfg.model, cfg.device, bool(cfg.fp16), cfg.coarse_size,
                cfg.coarse_blur, bool(cfg.use_hu), bool(cfg.use_fp),
                bool(cfg.invert_binary), getattr(cfg, "png_decoder", "cv2"),
                float(cfg.hu_weight), float(cfg.fp_weight))

    def _engine_for(self, cfg: Config, prefix: str):
        """返回 (引擎, 是否命中缓存)。未命中时加载索引（慢，见 open()）。"""
        sig = self._eng_sig(cfg, prefix)
        with self._eng_lock:
            hit = self._eng_cache.get(prefix)
            if hit is not None and hit[0] == sig:
                return hit[1], True
        eng = HybridEngine(cfg)
        eng.open(prefix)
        with self._eng_lock:
            self._eng_cache[prefix] = (sig, eng)
            # 最多两套：整图 + 瓦片（混合检索同时用）；多出的先淘汰
            while len(self._eng_cache) > 2:
                for k in list(self._eng_cache):
                    if k != prefix:
                        self._eng_cache.pop(k, None)
                        break
                else:
                    break
        return eng, False

    def _drop_engines(self, reason: str = "", silent: bool = False) -> float:
        """释放缓存的索引引擎（npz/memmap + 桶表），并回收内存。"""
        with self._eng_lock:
            n = len(self._eng_cache)
            self._eng_cache.clear()
        freed = 0.0
        try:
            import gc
            import psutil
            p = psutil.Process()
            before = p.memory_info().rss
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:       # noqa: BLE001
                pass
            freed = max(0.0, (before - p.memory_info().rss) / 2 ** 20)
        except Exception:           # noqa: BLE001
            pass
        if not silent:
            # 可能由 worker 线程调用：日志一律经消息队列回主线程渲染
            self.q.put(("log", f"已释放索引内存：{n} 套引擎，"
                               f"RSS 回落 {freed:.0f} MB"
                               + (f"（{reason}）" if reason else "")))
        return freed

    # ------------------------------------------------------------------
    # 性能图（可选导出）
    # ------------------------------------------------------------------
    def _perf_start(self, stage: str, title: str, cfg: Config,
                    prefix: str = "", meta: dict | None = None):
        """按开关创建阶段画像器；未开启返回 None（零开销）。"""
        var = self.perf_build_var if stage == "index" else self.perf_search_var
        if not var.get():
            return None
        prof = StageProfiler(stage, title, cfg=cfg, prefix=prefix, meta=meta)
        prof.start()
        return prof

    def _perf_finish(self, prof, status: str = "ok",
                     extra: list | None = None, note: str = "") -> str:
        """worker 线程调用：只写报告并记路径，UI 更新交给主线程。"""
        if prof is None:
            return ""
        path = prof.stop(status=status, extra=extra, note=note)
        if path:
            self._last_perf_report = path
        return path

    def _perf_report_ready(self, path: str, modal: bool = False):
        """主线程：提示已导出 + 点亮“最近性能图”按钮。

        模态弹窗会阻塞 Tk 主循环；若此时可视化还在回放排队帧，界面会像
        “卡住/输出中断”。因此有回放在跑时先等它播完（最多 ~6 秒）再弹。
        """
        if not path:
            return
        self._last_perf_report = path
        self._log(f"性能图已导出：{path}")
        btn = getattr(self, "btn_perf_open", None)
        if btn is not None:
            try:
                btn.configure(state="normal")
            except tk.TclError:
                pass
        if not modal:
            return
        self._pending_perf_modal = [path, 0]
        self.root.after(200, self._pump_perf_modal)

    def _viz_busy(self) -> bool:
        return bool(self._viz_queue.get("coarse") or self._viz_queue.get("fine"))

    def _pump_perf_modal(self):
        """等可视化回放结束后再弹性能图提示（最多等 6 秒，避免一直不弹）。"""
        pend = getattr(self, "_pending_perf_modal", None)
        if not pend:
            return
        path, waited = pend
        if self._viz_busy() and waited < 30:
            self._pending_perf_modal = [path, waited + 1]
            self.root.after(200, self._pump_perf_modal)
            return
        self._pending_perf_modal = None
        messagebox.showinfo("性能图已导出", f"已写入：\n{path}")

    # ------------------------------------------------------------------
    # 索引存储格式转换（旧 npz -> 侧车 .npy 快载）
    # ------------------------------------------------------------------
    def _compact_index(self):
        cfg = self.cfg_from_widgets(require_prefix=True)
        if cfg is None:
            return
        prefixes = [p for p in (self.prefix, self._tiles_prefix())
                    if p and os.path.exists(p + ".meta.json")]
        if not prefixes:
            messagebox.showwarning("提示", "没有找到可转换的索引")
            return
        if not messagebox.askokcancel(
                "优化索引存储（加速加载）",
                "把索引从 .npz（读取时整体解压进内存）转成侧车 .npy（可 mmap）：\n\n"
                "• 加载更快：444k 瓦片库实测 3.6s → <1s\n"
                "• 常驻内存更省：约 1.3GB → 约 0.3GB\n"
                "• 原 .npz 默认保留（可回退），转换期间请勿关闭程序\n\n"
                "将处理：" + "、".join(os.path.basename(p) for p in prefixes)
                + "\n\n是否继续？"):
            return
        self._set_busy(True, "转换索引存储格式…")
        threading.Thread(target=self._compact_worker, args=(cfg, prefixes),
                         daemon=True).start()

    def _compact_worker(self, cfg: Config, prefixes: list):
        from hybrid_search.store import compact
        prof = self._perf_start("index", "索引存储格式转换", cfg,
                                prefix=", ".join(prefixes))
        try:
            self._drop_engines("转换前释放索引内存", silent=True)
            total = len(prefixes)
            for i, p in enumerate(prefixes):
                r = compact(
                    p, delete_legacy=False,
                    progress=(lambda d, t, ph, _i=i: self._compact_prog(
                        _i, total, d, t, ph, prof)))
                msg = (f"{os.path.basename(p)}.* 已是侧车格式"
                       if r.get("already") else
                       f"{os.path.basename(p)}.* 转换完成：{r['n']} 行，"
                       f"{r['sec']:.1f}s")
                self.q.put(("log", msg))
            self.q.put(("compact_done",
                        {"perf": self._perf_finish(prof, note="完成")}))
        except Exception as e:  # noqa: BLE001
            self._perf_finish(prof, status="error", note=str(e)[:120])
            self.q.put(("error", f"索引存储转换失败：{e}\n{traceback.format_exc()}"))

    def _compact_prog(self, i: int, total: int, done: int, t: int,
                      phase: str, prof):
        self.q.put(("progress", (i * t + done, total * t, "compact")))
        if prof is not None:
            prof.bump(i * t + done, total * t)

    def _on_compact_done(self, data: dict):
        self._set_busy(False, "索引存储已优化（下次加载更快、内存更省）")
        self._drop_engines("格式已变，缓存失效", silent=True)
        self._perf_report_ready(data.get("perf", ""), modal=True)

    def _open_last_perf(self):
        p = self._last_perf_report or latest_report()
        if not p or not os.path.exists(p):
            messagebox.showinfo("性能图", "还没有生成过性能图。\n"
                                          "可在“建库参数/检索参数”页勾选导出开关。")
            return
        try:
            os.startfile(p)         # noqa: S606 Windows 打开默认浏览器
        except Exception as e:      # noqa: BLE001
            messagebox.showwarning("打开失败", f"{p}\n{e}")

    def _run_index(self, title: str, cfg: Config, op):
        # 阶段进度状态复位
        self._prog_phase = None
        self._prog_t0 = time.monotonic()
        self._prog_done0 = 0
        self._set_busy(True, f"{title} …（操作期间请勿重复点击）")
        threading.Thread(target=self._index_worker, args=(title, cfg, op),
                         daemon=True).start()

    def _index_worker(self, title: str, cfg: Config, op):
        prof = self._perf_start("index", title, cfg, prefix=self.prefix,
                                meta={"图片数": len(self.all_images)})
        self._cur_prof = prof          # 供 _prog_cb 在阶段切换时打点
        self._prog_prof_phase = None
        try:
            # 建库前先释放缓存引擎：把内存让给解码流水线（瓦片库可占 1GB+）
            self._drop_engines("建库前腾内存", silent=True)
            t0 = time.time()
            eng = HybridEngine(cfg)
            if prof:
                prof.mark("初始化引擎", 耗时=round(time.time() - t0, 3))
                prof.mark("解码+特征提取(粗筛与ResNet同一条流水线)")
            n = op(eng)
            if prof:
                # op() 返回即代表“特征提取 + 落盘”全部结束（engine 内部有
                # save/done 阶段边界事件，这里再补一条终结打点）
                prof.mark("op 返回：提取+落盘完成", 张数=n)
            self.q.put(("indexed", {"n": n, "prefix": self.prefix, "title": title,
                                    "perf": self._perf_finish(
                                        prof, note=f"{n} 张",
                                        extra=[("结果", [("入库张数", n)])])}))
        except Exception as e:  # noqa: BLE001
            self._perf_finish(prof, status="error", note=str(e)[:120])
            self.q.put(("error", f"{title}失败：{e}\n{traceback.format_exc()}"))
        finally:
            self._cur_prof = None

    # ------------------------------------------------------------------
    # 查验去重（一对多重复图）
    # ------------------------------------------------------------------
    def _dedup_check(self):
        """扫描图库找重复图：完全重复(MD5) + 近似重复(指纹汉明)。"""
        if not self._require_scan():
            return
        if self.busy:
            return
        th = simpledialog.askfloat(
            "查验去重",
            "近似重复判定阈值：指纹汉明比例上限（%）\n\n"
            "2 = 4096 位二值指纹里允许差异 ≤82 位（推荐：只抓重编码/缩放/换格式）\n"
            "更小更严格（少误报、可能漏检），更大更宽松"
            "（4~10 会把画师“差分图”等刻意近似变体也并成一组）",
            initialvalue=2.0, minvalue=0.2, maxvalue=20.0, parent=self.root)
        if th is None:
            return
        threshold = float(th) / 100.0
        if not messagebox.askokcancel(
                "查验去重",
                f"将扫描 {len(self.all_images)} 张图，查找“一对多”重复：\n\n"
                "• 完全重复：文件内容 MD5 相同（字节一致）\n"
                f"• 近似重复：二值指纹汉明 ≤ {th:.1f}%（重编码/缩放/换格式）\n"
                "• 已入库图片直接复用索引里的 MD5/指纹/特征，不重新解码\n"
                "• 结果只做展示与勾选，删除/移动由你确认后才执行\n\n"
                "是否开始？"):
            return
        self._reset_viz()
        self._set_busy(True, "查验重复图：复用索引 + 解码未入库文件…")
        threading.Thread(target=self._dedup_worker,
                         args=(list(self.all_images), threshold),
                         daemon=True).start()

    def _dedup_worker(self, paths: list, threshold: float):
        try:
            from hybrid_search import dedup as DD
            prefix = self.prefix if self._index_exists() else None

            def prog(done, total, phase):
                self.q.put(("progress", (done, max(int(total), 1),
                                         f"查验去重·{phase}")))

            rep = DD.scan_duplicates(paths, prefix=prefix,
                                     threshold=threshold, progress=prog)
            self.q.put(("dedup_done", rep))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"查验去重失败：{e}\n{traceback.format_exc()}"))

    def _on_dedup_done(self, rep):
        self._set_busy(False)
        msg = (f"查验去重：扫描 {rep.scanned} 张，用时 {rep.elapsed:.1f}s，"
               f"复用索引 {rep.indexed_used} 张，解码 {rep.decoded} 张")
        if rep.errors:
            msg += f"，{len(rep.errors)} 张读取失败"
        self._log(msg)
        if not rep.groups:
            self._status("查验去重：未发现重复图")
            messagebox.showinfo("查验去重", msg + "\n\n未发现重复图。")
            return
        self._log(f"发现 {len(rep.groups)} 组重复（完全 {rep.n_exact} / "
                  f"近似 {rep.n_near}），涉及 {rep.n_images} 张，"
                  f"可释放 {human_bytes(rep.wasted_bytes)}")
        self._status(f"查验去重完成：{len(rep.groups)} 组重复，"
                     f"可释放 {human_bytes(rep.wasted_bytes)}")
        DedupWindow(self, rep)

    def _prog_cb(self, prof=None):
        """engine 进度回调 -> 消息队列（worker 线程调用，主线程 pump 渲染）。

        阶段切换时**同时给性能图打点**：这样性能图上能明确看到
        “解码+特征提取 / 写盘 / 全部完成”的边界，不会误判 ResNet 还没跑完。
        """
        def cb(done, total, phase):
            self.q.put(("progress", (done, total, phase)))
            p = prof if prof is not None else getattr(self, "_cur_prof", None)
            if p is None:
                return
            p.bump(done, total)
            if phase != getattr(self, "_prog_prof_phase", None):
                self._prog_prof_phase = phase
                label = self._PHASE_LABELS.get(phase, phase or "")
                p.mark(f"阶段：{label}", 计数=f"{done}/{total}")
                if phase == "done":
                    p.mark("特征提取与落盘全部完成")
        return cb

    def _reset_viz(self):
        """新一轮索引任务开始：清空帧队列、重置阶段与计数。"""
        self._viz_queue["coarse"].clear()
        self._viz_queue["fine"].clear()
        self._viz_phase = None
        self._viz_frames = 0
        self.viz_status_var.set("等待首帧…")

    def _refresh_indexed_state(self):
        if not self.all_images:
            messagebox.showwarning("提示", "请先扫描图库")
            return
        if self._index_exists():
            self._load_indexed_set()
        else:
            self.indexed_set = set()
        self._apply_indexed_tag()
        self._log(f"已索引标记刷新：库内 {len(self.indexed_set)} 张")

    def _index_exists(self) -> bool:
        if not self.prefix:
            self.prefix = self._auto_prefix()
        return bool(self.prefix) and os.path.exists(self.prefix + ".meta.json")

    def _tiles_prefix(self) -> str:
        """瓦片索引前缀：与整图索引同目录下的 gallery_tiles。"""
        from hybrid_search.tile_index import tiles_prefix_of
        if not self.prefix:
            self.prefix = self._auto_prefix()
        return tiles_prefix_of(self.prefix) if self.prefix else ""

    def _tiles_index_exists(self) -> bool:
        tp = self._tiles_prefix()
        return bool(tp) and os.path.exists(tp + ".meta.json")

    def _tiles_index(self):
        """④ 子图索引：无 → 全量切块构建；已有 → 增量（参数从 meta 恢复）。"""
        if not self._require_scan():
            return
        cfg = self.cfg_from_widgets()
        if cfg is None:
            return
        tp = self._tiles_prefix()
        if not tp:
            messagebox.showwarning("提示", "请先选择图库目录或填写索引前缀")
            return
        exists = self._tiles_index_exists()
        self._reset_viz()
        self._set_busy(True,
                       "子图索引增量（切块去重）…" if exists
                       else "子图索引构建（大图切 512px+25% 重叠瓦片）…")
        threading.Thread(target=self._tiles_index_worker,
                         args=(cfg, tp, exists), daemon=True).start()

    def _tiles_index_worker(self, cfg: Config, tp: str, exists: bool):
        title = "子图索引增量" if exists else "子图索引构建"
        prof = self._perf_start("index", title, cfg, prefix=tp,
                                meta={"图片数": len(self.all_images),
                                      "模式": "增量" if exists else "全量"})
        try:
            from hybrid_search import tile_index as TI
            self._drop_engines("瓦片建库前腾内存", silent=True)
            eng = HybridEngine(cfg)
            prog = self._prog_cb(prof)
            # 瓦片建库进度回调为 (done,total) 两参；带第三参时是阶段边界
            # （save/done），必须原样透传，否则性能图看不到“结束状态”
            cb2 = (lambda d, t, ph="tiles": prog(d, t, ph)) if prog else None
            # 过程可视化帧(低开销)：每瓦片一张 64×64 二值帧 + 每文件首块 16×16
            # 象限帧，复用既有 24fps 绘制节流与有界帧队列，不影响建库速率
            viz = self._viz_cb()
            if exists:
                t0 = time.time()
                eng.open(tp)
                if prof:
                    prof.mark("加载已有瓦片索引", 复用="否",
                              行数=eng.coarse.size,
                              耗时=round(time.time() - t0, 2))
                    prof.mark("增量解码+特征提取")
                n = TI.add_tiles(eng, tp, paths=list(self.all_images),
                                 progress=cb2, frame_sink=viz)
            else:
                if prof:
                    prof.mark("全量解码+特征提取")
                n = TI.build_tiles(eng, tp, paths=list(self.all_images),
                                   progress=cb2, frame_sink=viz)
            if prof:
                prof.mark("写盘完成")
            self.q.put(("tiles_done", {"n": n, "title": title, "prefix": tp,
                                       "exists": exists,
                                       "perf": self._perf_finish(
                                           prof, note=f"{n} 块",
                                           extra=[("结果", [("入库瓦片", n)])])}))
        except Exception as e:  # noqa: BLE001
            perf = self._perf_finish(prof, status="error", note=str(e)[:120])
            self.q.put(("error",
                        f"{title}失败：{e}\n{traceback.format_exc()}"))
            if perf:
                self.q.put(("perf", perf))

    def _load_indexed_set(self):
        try:
            data = IndexFiles(self.prefix).load_coarse()
            self.indexed_set = {os.path.normcase(os.path.abspath(p))
                                for p in data["paths"]}
        except Exception as e:  # noqa: BLE001
            self._log(f"读取已索引清单失败：{e}")

    # ==================================================================
    # 右区：图片列表 / 检索
    # ==================================================================
    def _build_upper(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # ---------- 页1：图片列表 ----------
        page_list = ttk.Frame(nb, padding=4)
        nb.add(page_list, text="图片列表")
        split = ttk.Panedwindow(page_list, orient="horizontal")
        split.pack(fill="both", expand=True)
        left = ttk.Frame(split)
        split.add(left, weight=3)
        cols = ("idx", "size", "name")
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 selectmode="extended")
        for cid, text, w in (("idx", "已索引", 60), ("size", "大小", 80),
                             ("name", "文件（双击=设为查询图）", 400)):
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=w, anchor="w")
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_double)

        right = ttk.Frame(split, width=330)
        split.add(right, weight=0)
        self.preview_lbl = tk.Label(right, text="（选中图片后在此预览）",
                                    background="#1e1e22", foreground="#777777",
                                    width=38, height=16)
        self.preview_lbl.pack(fill="both", expand=True, padx=2)
        self.preview_info = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.preview_info, wraplength=320,
                  foreground="#bbbbbb", font=("Microsoft YaHei UI", 8)).pack(
            fill="x", padx=4, pady=2)

        # ---------- 页2：检索 ----------
        page_q = ttk.Frame(nb, padding=4)
        nb.add(page_q, text="以图搜图")
        qrow = ttk.Frame(page_q)
        qrow.pack(fill="x")
        ttk.Label(qrow, text="查询图片:").pack(side="left")
        self.query_var = tk.StringVar()
        ttk.Entry(qrow, textvariable=self.query_var, width=60).pack(
            side="left", fill="x", expand=True, padx=4)
        ttk.Button(qrow, text="浏览…", command=self._pick_query).pack(side="left")
        self.btn_q = ttk.Button(qrow, text="开始检索",
                                command=self._do_search)
        self.btn_q.pack(side="left", padx=8)

        mrow = ttk.Frame(page_q)
        mrow.pack(fill="x", pady=(3, 0))
        ttk.Label(mrow, text="检索模式:").pack(side="left")
        for val, txt in (("full", "整图索引 (Top≤10)"),
                         ("tiles", "局部索引·瓦片 (Top≤10)"),
                         ("hybrid", "混合·两者都搜 (Top≤20)")):
            ttk.Radiobutton(mrow, text=txt, value=val,
                            variable=self.search_mode_var,
                            command=self._on_search_mode).pack(
                side="left", padx=(8, 0))
        self.mode_note_var = tk.StringVar(value="")
        ttk.Label(mrow, textvariable=self.mode_note_var,
                  foreground="#b35900").pack(side="left", padx=12)

        mid = ttk.Frame(page_q)
        mid.pack(fill="x", pady=4)
        self.result_info_var = tk.StringVar(value="（尚无结果：选查询图后点“开始检索”"
                                                  "；列表里双击任意图可直接设为查询）")
        ttk.Label(mid, textvariable=self.result_info_var, foreground="#aaaaaa",
                  wraplength=900).pack(side="left")
        ttk.Button(mid, text="打开原图", command=self._open_selected).pack(
            side="right", padx=4)
        ttk.Button(mid, text="复制路径", command=self._copy_selected).pack(
            side="right", padx=4)
        ttk.Button(mid, text="导出总览图…", command=self._export_sheet).pack(
            side="right", padx=4)

        self.grid_host = ScrollFrame(page_q)
        self.grid_host.pack(fill="both", expand=True)

        det = ttk.LabelFrame(page_q, text="命中详情")
        det.pack(fill="x", pady=(2, 0))
        self.detail_var = tk.StringVar(value="点击结果缩略图查看详情")
        ttk.Label(det, textvariable=self.detail_var, wraplength=1100,
                  font=("Consolas", 9)).pack(anchor="w", padx=6, pady=3)

    # ==================================================================
    # 树/预览交互
    # ==================================================================
    def _on_tree_select(self, _e=None):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        path = self.tree.set(iid, "name")
        # 预览
        photo = self.thumbs.get(path, 300)
        if photo:
            self.preview_lbl.configure(image=photo, text="")
            self.preview_lbl.image = photo
        else:
            self.preview_lbl.configure(image="",
                                       text="（无预览：巨型图或无法解码）")
            self.preview_lbl.image = None
        try:
            size = os.path.getsize(path)
            info = f"{os.path.basename(path)}\n{human_bytes(size)}\n{path}"
        except OSError:
            info = path
        self.preview_info.set(info)

    def _on_tree_double(self, _e=None):
        sel = self.tree.selection()
        if not sel:
            return
        path = self.tree.set(sel[0], "name")
        self.query_var.set(path)
        self.last_query = path
        self._status(f"已把 {os.path.basename(path)} 设为查询图，可切到“以图搜图”开始检索")

    def _checked_rows(self):
        """勾选 = 当前选中的行（多选）"""
        out = []
        for iid in self.tree.selection():
            p = self.tree.set(iid, "name")
            if p:
                out.append(p)
        return out

    def _apply_indexed_tag(self):
        for iid in self.tree.get_children():
            p = self.tree.set(iid, "name")
            key = os.path.normcase(os.path.abspath(p))
            tag = "✔" if key in self.indexed_set else "·"
            self.tree.set(iid, "idx", tag)
            if key in self.indexed_set:
                self.tree.item(iid, tags=("idx",))
            else:
                self.tree.item(iid, tags=("new",))

    # ==================================================================
    # 查询
    # ==================================================================
    def _pick_query(self):
        p = filedialog.askopenfilename(
            title="选择查询图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.gif"),
                       ("所有文件", "*.*")])
        if p:
            self.query_var.set(p)
            self.last_query = p

    def _on_search_mode(self):
        """检索模式切换：混合模式首次选中弹性能警告；tiles/hybrid 提示索引缺失。"""
        mode = self.search_mode_var.get()
        if mode in ("tiles", "hybrid"):
            if not self._tiles_index_exists():
                self.mode_note_var.set("（局部索引未构建：先点工具栏 ④ 子图索引）")
            else:
                self.mode_note_var.set("")
        else:
            self.mode_note_var.set("")
        if mode == "hybrid" and not self._hybrid_warned:
            self._hybrid_warned = True
            ok = messagebox.askokcancel(
                "混合检索 · 性能提示",
                "混合模式将【整图索引】与【局部(瓦片)索引】同时检索：\n\n"
                "• 两套索引并行打分，耗时约为单套的 2 倍，需同时载入两套精排特征；\n"
                "• 同一原图的全图/局部命中去重，相似度取两路较高值重新统计；\n"
                "• 结果固定返回 20 条（其它模式 10 条）。\n\n"
                "是否继续？")
            if not ok:
                self.search_mode_var.set("full")
                self.mode_note_var.set("")

    def _do_search(self):
        q = self.query_var.get().strip()
        if not q or not os.path.exists(q):
            messagebox.showwarning("提示", "请选择存在的查询图片")
            return
        mode = self.search_mode_var.get()
        if mode == "full":
            if not self._index_exists():
                messagebox.showwarning("提示", "索引不存在，请先在“图片列表”页建索引")
                return
        else:
            if not self._tiles_index_exists():
                messagebox.showwarning(
                    "提示", "局部(瓦片)索引不存在，请先点工具栏“④ 子图索引”构建")
                return
            if mode == "hybrid" and not self._index_exists():
                messagebox.showwarning("提示", "整图索引不存在，无法混合检索")
                return
        cfg = self.cfg_from_widgets(require_prefix=True)
        if cfg is None:
            return
        self._set_busy(True, f"检索中：{os.path.basename(q)} …")
        self.last_query = q
        threading.Thread(target=self._search_worker, args=(q, cfg),
                         daemon=True).start()

    def _search_worker(self, q: str, cfg: Config):
        mode = self.search_mode_var.get()
        prof = self._perf_start("search", f"以图搜图({mode})", cfg,
                                prefix=self.prefix,
                                meta={"查询图": os.path.basename(q),
                                      "模式": mode})
        try:
            if mode == "full":
                t0 = time.time()
                eng, cached = self._engine_for(cfg, self.prefix)
                if prof:
                    prof.mark("加载索引", 复用="是" if cached else "否",
                              耗时=round(time.time() - t0, 3),
                              库内=eng.coarse.size)
                    prof.mark("粗筛+ResNet检索")
                out = eng.search(q, coarse_k=cfg.coarse_k, top_k=cfg.top_k)
            else:
                from hybrid_search import tile_index as TI
                tp = TI.tiles_prefix_of(self.prefix)
                if mode == "tiles":
                    t0 = time.time()
                    eng, cached = self._engine_for(cfg, tp)
                    if prof:
                        prof.mark("加载瓦片索引", 复用="是" if cached else "否",
                                  耗时=round(time.time() - t0, 3),
                                  瓦片数=eng.coarse.size)
                        prof.mark("瓦片检索(切块聚合+精排)")
                    out = TI.search_tiles_tiled(eng, q, top_k=cfg.top_k,
                                                coarse_k=cfg.coarse_k,
                                                method="lsh")
                else:                      # hybrid：两路各取 Top20 再合并（结果 ≤20）
                    t0 = time.time()
                    eng_full, c1 = self._engine_for(cfg, self.prefix)
                    eng_t, c2 = self._engine_for(cfg, tp)
                    # 共享特征提取器（同 cfg 模型），避免重复加载模型
                    ex = getattr(eng_full, "_extractor", None)
                    if ex is not None:
                        eng_t._extractor = ex  # noqa: SLF001 同项目协作
                    if prof:
                        prof.mark("加载两套索引", 复用=f"{c1}/{c2}",
                                  耗时=round(time.time() - t0, 3))
                        prof.mark("整图检索")
                    o_full = eng_full.search(q, coarse_k=cfg.coarse_k,
                                             top_k=20)
                    if prof:
                        prof.mark("瓦片检索")
                    o_t = TI.search_tiles_tiled(eng_t, q, top_k=20,
                                                coarse_k=cfg.coarse_k,
                                                method="lsh")
                    out = TI.merge_hybrid(o_full, o_t, q, top_k=20)
            hits = [(h.rank, h.path, h.fine_score, h.coarse_score, h.d_fp,
                     h.box, h.match_kind) for h in out.hits]
            perf = ""
            if prof:
                perf = self._perf_finish(
                    prof, note=f"命中 {len(hits)} 条",
                    extra=[("检索耗时明细(s)", list(out.times.items())),
                           ("结果", [("库内条目", out.db_size),
                                     ("候选保留", out.coarse_kept),
                                     ("仅粗筛", int(bool(out.coarse_only))),
                                     ("剔除自身", int(bool(out.self_excluded)))])])
            self.q.put(("search_done", {
                "hits": hits, "db": out.db_size, "kept": out.coarse_kept,
                "coarse_only": out.coarse_only,
                "self_excluded": out.self_excluded, "times": out.times,
                "method": mode, "perf": perf}))
        except Exception as e:  # noqa: BLE001
            self._perf_finish(prof, status="error", note=str(e)[:120])
            self.q.put(("error", f"检索失败：{e}\n{traceback.format_exc()}"))

    # ---- 结果网格 ----
    def _show_results(self, hits):
        for w in self.grid_host.inner.winfo_children():
            w.destroy()
        self.tiles = []
        self._tile_photos = []
        self.sel_tile = None
        if not hits:
            self.result_info_var.set("（无结果：粗筛未命中任何候选）")
            return
        import math
        cols = 5
        no_preview = 0
        for i, row in enumerate(hits):
            rank, path, fine, coarse, dfp, box, kind = row
            photo = self.thumbs.get(path, 128, box=box if kind != "full"
                                    else None)
            if photo is None:
                no_preview += 1
            name = os.path.basename(path)
            tag = {"tile": "局部", "both": "双", "full": ""}[kind]
            head = f"{tag}·" if tag else ""
            label_txt = f"{rank}. {head}{name[:10]}  cos={fine:.3f}" \
                if fine == fine else f"{rank}. {head}{name[:10]}  (仅粗筛)"
            if photo is None:
                label_txt += "  ⚠无预览"
            # 关键：缩略图失败也要用占位格子占住这一格，保证
            # tiles[k].hit_index == k（历史 bug：跳过失败项导致点击错位、
            # 网格出现空洞、不足 top_k 个）
            tile = ResultTile(self.grid_host.inner, photo, label_txt,
                              lambda idx=i: self._select_hit(idx),
                              hit_index=i,
                              placeholder="无预览\n(超大图或解码失败)")
            r, c = divmod(i, cols)
            tile.frame.grid(row=r, column=c, padx=4, pady=4, sticky="n")
            self.tiles.append(tile)
            if photo is not None:
                self._tile_photos.append(photo)
        if no_preview:
            self.result_info_var.set(
                f"{len(hits)} 个结果（{no_preview} 个无预览：超大图或解码失败）")
        return no_preview

    def _select_hit(self, i: int):
        if not (0 <= i < len(self.last_hits)):
            return
        for t in self.tiles:
            t.highlight(t.hit_index == i)
        self.sel_tile = next((t for t in self.tiles if t.hit_index == i), None)
        rank, path, fine, coarse, dfp, box, kind = self.last_hits[i]
        fine_txt = f"{fine:.4f}" if fine == fine else "n/a(未精排)"
        kind_txt = {"tile": "局部命中(瓦片)", "both": "整图+局部双命中",
                    "full": "整图命中"}[kind]
        box_txt = "" if box is None else \
            f"    命中框: ({int(box[0])},{int(box[1])})-({int(box[2])},{int(box[3])})" \
            " (原图像素, 缩略图中已叠红框)"
        self.detail_var.set(
            f"#{rank}  {path}\n"
            f"类型={kind_txt}    ResNet相似度={fine_txt}    "
            f"粗筛综合分={coarse:.4f}    指纹汉明比例={dfp:.4f}{box_txt}")

    def _selected_hit_path(self):
        if self.sel_tile is None:
            return None
        idx = self.sel_tile.hit_index          # 用格子自带的真实下标（不再靠列表位置）
        if 0 <= idx < len(self.last_hits):
            return self.last_hits[idx][1]
        return None

    def _open_selected(self):
        p = self._selected_hit_path()
        if not p or not os.path.exists(p):
            messagebox.showinfo("提示", "先在结果区点选一个缩略图")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("错误", f"无法打开图片：{e}")

    def _copy_selected(self):
        p = self._selected_hit_path()
        if not p:
            messagebox.showinfo("提示", "先在结果区点选一个缩略图")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(p)
        self._status(f"已复制路径：{p}")

    def _export_sheet(self):
        if not self.last_hits:
            messagebox.showinfo("提示", "没有可导出的检索结果")
            return
        p = filedialog.asksaveasfilename(
            title="保存 Top-K 总览图", defaultextension=".png",
            initialfile="search_result.png",
            filetypes=[("PNG 图片", "*.png")])
        if not p:
            return
        from hybrid_search.visuals import save_contact_sheet
        ok = save_contact_sheet([h[1] for h in self.last_hits], p)
        if ok:
            self._status(f"总览图已保存：{p}")

    # ==================================================================
    # 线程结果泵 & 状态
    # ==================================================================
    def _pump(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "progress":
                    self._render_progress(msg[1])
                elif kind == "viz":
                    # 按阶段入不同帧队列（融合建库时两阶段帧同时产生，
                    # 绘制端先播完粗筛帧，再原位覆盖 ResNet 象限帧）
                    phase = msg[1][0]
                    bucket = self._viz_queue.get(phase)
                    if bucket is not None:
                        bucket.append(msg[1])
                elif kind == "scan_done":
                    self._on_scan_done(msg[1])
                elif kind == "indexed":
                    self._on_index_done(msg[1])
                elif kind == "search_done":
                    self._on_search_done(msg[1])
                elif kind == "tiles_done":
                    self._on_tiles_done(msg[1])
                elif kind == "dedup_done":
                    self._on_dedup_done(msg[1])
                elif kind == "perf":
                    self._perf_report_ready(msg[1])
                elif kind == "compact_done":
                    self._on_compact_done(msg[1])
                elif kind == "handoff_done":
                    self._on_handoff_done(msg[1])
                elif kind == "error":
                    self._log("【错误】\n" + msg[1])
                    messagebox.showerror("操作失败", msg[1].splitlines()[0])
                    self._set_busy(False)
        except queue.Empty:
            pass
        # 33ms 轮询 ≈ 30fps 数据源；真正绘制再按 24fps 节流
        self._maybe_draw_viz()
        self.root.after(33, self._pump)

    # ---- 双阶段进度渲染（融合建库 / 粗筛 / 校验 统一入口）-------------
    _PHASE_LABELS = {
        "fused": "粗筛+ResNet 融合提取(单遍解码)",
        "coarse": "① 二值法粗筛",
        "fine": "② ResNet 全库特征",
        "verify": "解码校验",
        "save": "③ 特征提取完成 · 正在写盘",
        "done": "✅ 全部完成（粗筛与精排均已落盘）",
        "compact": "索引存储格式转换",
    }

    def _render_progress(self, payload):
        """payload=(done, total, phase) 或 (done, total)：更新进度条与状态栏。
        状态栏文案：阶段名 · 计数/总数(百分比) · 张/秒 · 预计剩余。"""
        done, total = payload[0], payload[1]
        phase = payload[2] if len(payload) > 2 else ""
        label = self._PHASE_LABELS.get(phase, phase or "处理中")
        if total <= 0:
            return
        # 阶段切换时重置该阶段速率估算基准
        if phase != self._prog_phase:
            self._prog_phase = phase
            self._prog_t0 = time.monotonic()
            self._prog_done0 = done
        self.progress.configure(maximum=total, value=done)
        pct = done / total * 100.0
        dt = time.monotonic() - self._prog_t0
        rate = (done - self._prog_done0) / dt if dt > 0.1 else 0.0
        if rate > 0 and done < total:
            eta = _fmt_eta((total - done) / rate)
        else:
            eta = "--"
        speed = f"{rate:.0f} 张/秒" if rate > 0 else "…"
        self._status(f"{label}：{done:,}/{total:,} 张（{pct:.0f}%）| "
                     f"{speed} | 预计剩余 {eta}")

    def _on_scan_done(self, data: dict):
        self.all_images = data["paths"]
        # 刷新列表
        self.tree.delete(*self.tree.get_children())
        for i, p in enumerate(self.all_images):
            try:
                sz = os.path.getsize(p)
            except OSError:
                sz = 0
            self.tree.insert("", "end",
                             values=("·", human_bytes(sz), p))
        self.tree.tag_configure("idx", background="#e8f4e0")
        self.tree.tag_configure("new", background="#ffffff")
        fmt = "  ".join(f"{k}:{v}" for k, v in
                        sorted(data["formats"].items(), key=lambda x: -x[1])[:8])
        broken = f"，跳过损坏 {data['broken']} 张" if data.get("broken") else ""
        self.scan_info_var.set(
            f"共 {data['count']} 张（{fmt}） 耗时 {data['elapsed']:.2f}s{broken}")
        self._load_indexed_set()
        self._apply_indexed_tag()
        self._set_busy(False, f"扫描完成：{data['count']} 张图片（索引内 "
                              f"{len(self.indexed_set)} 张）")

    def _on_index_done(self, data: dict):
        self.progress.configure(value=0)
        n = data["n"]
        # 明确“结束状态”：粗筛与精排都在同一条流水线里，op 返回即全部落盘
        self._set_busy(False, f"✅ {data['title']}完成：{n} 张"
                              f"（粗筛 + ResNet 均已落盘）-> {self.prefix}.*")
        self._log(f"✅ {data['title']}完成：{n} 张已落盘（{self.prefix}.*）")
        self._load_indexed_set()
        self._apply_indexed_tag()
        self._drop_engines("索引已改写，缓存失效", silent=True)
        # 可视化：进入快放模式，把排队帧 1 秒内播完（不再像“还在处理 ResNet”）
        if self._viz_busy():
            self._viz_fast = True
            label = "① 粗筛·二值点阵" if self._viz_phase == "coarse" \
                else "② ResNet·采样象限"
            self.viz_status_var.set(
                f"{label} 快放中 · 建库已完成，剩余排队帧正在快速播完"
                f"（累计 {self._viz_frames} 帧）")
        else:
            self.viz_status_var.set(f"✅ 已完成（累计 {self._viz_frames} 帧）")
        self._perf_report_ready(data.get("perf", ""), modal=True)

    def _on_search_done(self, data: dict):
        self.progress.configure(value=0)
        hits = data["hits"]
        self.last_hits = hits
        no_preview = self._show_results(hits)
        times = data["times"]
        method = data.get("method", "full")
        ms = lambda k: f"{times.get(k, 0) * 1000:.1f}ms"
        if method == "full":
            mode = "仅粗筛" if data["coarse_only"] else \
                f"粗筛 {ms('粗筛(特征+扫描)')} + 精排 {ms('精排(查询特征+打分)')}"
        elif method == "tiles":
            parts = []
            for k in ("查询图指纹", "LSH/候选获取", "hash复核(指纹查验)",
                      "原图聚合(组max)", "收敛精排(整图切片×q)"):
                if k in times:
                    parts.append(f"{k} {ms(k)}")
            mode = "局部索引(瓦片)" + ("；" + " | ".join(parts) if parts else "")
        else:
            mode = "混合(整图+瓦片，去重取高)" if not data["coarse_only"] \
                else "混合(仅粗筛)"
        info = (f"库 {data['db']} 张 | 候选 {data['kept']} 张 | "
                f"{mode} | 合计 {ms('total')}"
                + (" | 已剔除查询图自身" if data["self_excluded"] else "")
                + (f" | {no_preview} 个无预览" if no_preview else ""))
        self.result_info_var.set(f"{len(hits)} 个结果  |  {info}")
        self.detail_var.set("点击结果缩略图查看详情（局部命中缩略图带红框）")
        self._set_busy(False, f"检索完成，返回 {len(hits)} 个结果")
        self._perf_report_ready(data.get("perf", ""))

    def _on_tiles_done(self, data: dict):
        self.progress.configure(value=0)
        n = data["n"]
        self._set_busy(False, f"{data['title']}完成：{n} 个瓦片 -> {data['prefix']}.*")
        unit = "瓦片(已有索引增量为空时=0)"
        self.tiles_state_var.set(
            f"子图索引：{data['prefix'].split(os.sep)[-1]}"
            f"（最近一次 {data['title']} +{n} {unit}）"
            if data["exists"] else
            f"子图索引已构建：{n} 个瓦片 -> {data['prefix']}.*")
        self._log(f"{data['title']}完成：{n} 个瓦片 -> {data['prefix']}.*")
        self._drop_engines("瓦片索引已改写，缓存失效", silent=True)
        self._perf_report_ready(data.get("perf", ""), modal=True)

    # ------------------------------------------------------------------
    def _log(self, text: str):
        self.log_txt.configure(state="normal")
        self.log_txt.insert("end", text + "\n")
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")

    def _status(self, text: str):
        self.status_var.set(text)

    def _set_busy(self, busy: bool, msg: str = ""):
        self.busy = busy
        self._status(msg)
        state = "disabled" if busy else "normal"
        for b in (self.btn_scan, self.btn_build_all, self.btn_build_sel,
                  self.btn_add_new, self.btn_refresh_idx, self.btn_q,
                  getattr(self, "btn_tiles", None),
                  getattr(self, "btn_dedup", None)):
            if b is not None:
                b.configure(state=state)
        if getattr(self, "btn_peer_switch", None) is not None:
            # 切换按钮：busy 或切换目标校验不过 → 禁用
            self._peer_apply_state()
        if not busy:
            self._prog_phase = None
            self.progress.configure(value=0)

    def _require_scan(self) -> bool:
        if not self.all_images:
            messagebox.showwarning("提示", "请先“扫描图库”")
            return False
        return True


def _fmt_eta(seconds: float) -> str:
    """秒数 -> 人类可读的预计剩余时间。"""
    if seconds < 0 or seconds != seconds:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


# ==========================================================================
def main(argv=None) -> int:
    # --auto-handoff <path>：自动交接模式（增量建库），不影响无参数正常启动
    auto_handoff = None
    args = list(sys.argv[1:] if argv is None else argv)
    if "--auto-handoff" in args:
        i = args.index("--auto-handoff")
        if i + 1 < len(args):
            auto_handoff = args[i + 1]
    elif any(s.startswith("--auto-handoff=") for s in args):
        auto_handoff = next(s.split("=", 1)[1]
                            for s in args if s.startswith("--auto-handoff="))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
    root = tk.Tk()
    App(root, auto_handoff=auto_handoff)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n已通过 Ctrl+C 中断，正在退出…")
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
