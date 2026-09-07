# -*- coding: utf-8 -*-
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
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from hybrid_search.config import Config
from hybrid_search.engine import HybridEngine
from hybrid_search.io_utils import human_bytes
from hybrid_search.store import IndexFiles

import peer_launcher as PL      # “切换启动”全栈图库管理器（校验/登记/启动）

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
    PREVIEW_MAX_PX = 30_000_000

    def __init__(self, cap=240):
        self.cap = cap
        self._cache: dict = {}
        self._order: list = []

    def get(self, path: str, edge: int) -> "ImageTk.PhotoImage | None":
        key = (path, edge)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        try:
            with Image.open(path) as im:
                w, h = im.size
                if w * h > self.PREVIEW_MAX_PX:
                    return None          # 巨型图跳过预览（仍可索引与检索）
                im.load()
            im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
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
    def __init__(self, parent, thumb, text, cmd):
        self.frame = ttk.Frame(parent, padding=2, style="Tile.TFrame")
        self.btn = tk.Label(self.frame, image=thumb, cursor="hand2",
                            background="#2b2b31", bd=0)
        self.btn.image = thumb
        self.btn.pack()
        self.label = tk.Label(self.frame, text=text,
                              background="#2b2b31", foreground="#dcdcdc",
                              font=("Microsoft YaHei UI", 8), anchor="w")
        self.label.pack(fill="x")
        for w in (self.btn, self.label, self.frame):
            w.bind("<Button-1>", lambda _e, c=cmd: c())

    def highlight(self, on: bool):
        bg = "#3d5a80" if on else "#2b2b31"
        for w in (self.frame,):
            w.configure(style="TileSel.TFrame" if on else "Tile.TFrame")


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
        绘制绝不拖慢主流程。粗筛帧队列播完前不切 ResNet 帧（顺序观感）。"""
        q = (self._viz_queue["coarse"] or self._viz_queue["fine"])
        if not q:
            return
        now = time.monotonic()
        if now - self._viz_last_draw < 1.0 / 24.0:
            return
        self._viz_last_draw = now
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
        self.btn_refresh_idx = ttk.Button(row3, text="刷新已索引标记",
                                          command=self._refresh_indexed_state)
        for b in (self.btn_build_all, self.btn_build_sel, self.btn_add_new,
                  self.btn_refresh_idx):
            b.pack(side="left", padx=(0, 8))

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
        self._add_check(page_build, "MD5 内容去重", "dedup", True)
        self._add_int(page_build, "粗筛并行线程(0=自动≤8)", "workers", 0, 0, 64,
                      tooltip="0=自动(不超过8且不超过CPU核数)，1=串行最省内存；\n"
                              "解码在C层释放GIL，多线程可线性提速")
        self._add_int(page_build, "精排解码线程(0=自动≤8)", "decode_workers", 0,
                      0, 64,
                      tooltip="图像读盘+解码+预处理的多线程数；\n"
                              "解码与GPU前向重叠，GPU场景建议4~8")
        self._add_int(page_build, "torch推理线程(0=默认)", "torch_threads", 0,
                      0, 128,
                      tooltip="CPU前向线程数。多数机型1线程最快\n"
                              "(实测多线程反而慢)，GPU场景无需设置")
        self._add_text(page_build, "图片格式(逗号分隔)",
                       "extensions", "jpg,jpeg,png,bmp,tif,tiff,webp",
                       tooltip="扫描与建索引支持的扩展名，改完重新扫描生效")

        self._sep(page_search, "检索流程")
        self._add_int(page_search, "粗筛候选数 coarse_k", "coarse_k", 300, 10, 100000)
        self._add_int(page_search, "最终返回数 top_k", "top_k", 10, 1, 100)
        self._add_check(page_search, "剔除查询图自身", "exclude_self", True)
        self._sep(page_search, "索引位置")
        self._add_text(page_search, "索引前缀(留空=自动放图库内)",
                       "prefix", "",
                       tooltip="例如 D:/idx/my_gallery\n留空自动为 <图库目录>/.gallery_index/gallery")
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

    def _run_index(self, title: str, cfg: Config, op):
        # 阶段进度状态复位
        self._prog_phase = None
        self._prog_t0 = time.monotonic()
        self._prog_done0 = 0
        self._set_busy(True, f"{title} …（操作期间请勿重复点击）")
        threading.Thread(target=self._index_worker, args=(title, cfg, op),
                         daemon=True).start()

    def _index_worker(self, title: str, cfg: Config, op):
        try:
            eng = HybridEngine(cfg)
            n = op(eng)
            self.q.put(("indexed", {"n": n, "prefix": self.prefix, "title": title}))
        except Exception as e:  # noqa: BLE001
            self.q.put(("error", f"{title}失败：{e}\n{traceback.format_exc()}"))

    def _prog_cb(self):
        """engine 进度回调 -> 消息队列（worker 线程调用，主线程 pump 渲染）。"""
        def cb(done, total, phase):
            self.q.put(("progress", (done, total, phase)))
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

    def _do_search(self):
        q = self.query_var.get().strip()
        if not q or not os.path.exists(q):
            messagebox.showwarning("提示", "请选择存在的查询图片")
            return
        if not self._index_exists():
            messagebox.showwarning("提示", "索引不存在，请先在“图片列表”页建索引")
            return
        cfg = self.cfg_from_widgets(require_prefix=True)
        if cfg is None:
            return
        self._set_busy(True, f"检索中：{os.path.basename(q)} …")
        self.last_query = q
        threading.Thread(target=self._search_worker, args=(q, cfg), daemon=True).start()

    def _search_worker(self, q: str, cfg: Config):
        try:
            eng = HybridEngine(cfg)
            eng.open(self.prefix)
            out = eng.search(q, coarse_k=cfg.coarse_k, top_k=cfg.top_k)
            hits = [(h.rank, h.path, h.fine_score, h.coarse_score, h.d_fp)
                    for h in out.hits]
            self.q.put(("search_done", {
                "hits": hits, "db": out.db_size, "kept": out.coarse_kept,
                "coarse_only": out.coarse_only,
                "self_excluded": out.self_excluded, "times": out.times}))
        except Exception as e:  # noqa: BLE001
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
        for i, (rank, path, fine, coarse, dfp) in enumerate(hits):
            photo = self.thumbs.get(path, 128)
            if photo is None:
                continue
            name = os.path.basename(path)
            label_txt = f"{rank}. {name[:10]}  cos={fine:.3f}" \
                if fine == fine else f"{rank}. {name[:10]}  (仅粗筛)"
            tile = ResultTile(self.grid_host.inner, photo, label_txt,
                              lambda i=i: self._select_hit(i))
            r, c = divmod(i, cols)
            tile.frame.grid(row=r, column=c, padx=4, pady=4, sticky="n")
            self.tiles.append(tile)
            self._tile_photos.append(photo)

    def _select_hit(self, i: int):
        for t in self.tiles:
            t.highlight(False)
        self.tiles[i].highlight(True)
        self.sel_tile = self.tiles[i]
        rank, path, fine, coarse, dfp = self.last_hits[i]
        fine_txt = f"{fine:.4f}" if fine == fine else "n/a(未精排)"
        self.detail_var.set(
            f"#{rank}  {path}\n"
            f"ResNet余弦相似度={fine_txt}    粗筛综合分={coarse:.4f}    "
            f"指纹汉明比例={dfp:.4f}")

    def _selected_hit_path(self):
        if self.sel_tile is None:
            return None
        idx = self.tiles.index(self.sel_tile)
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
        self._set_busy(False, f"{data['title']}完成：{n} 张 -> {self.prefix}.*")
        self._load_indexed_set()
        self._apply_indexed_tag()
        # 可视化收尾：不强制清空——让排队中的帧按 24fps 自然播完
        # （快速任务两段动画也能完整回放），仅更新状态提示
        if self._viz_phase:
            label = "① 粗筛·二值点阵" if self._viz_phase == "coarse" \
                else "② ResNet·采样象限"
            self.viz_status_var.set(
                f"{label} 绘制中 · 任务已完成，排队帧继续回放至结束"
                f"（已采样 {self._viz_frames} 帧）")

    def _on_search_done(self, data: dict):
        self.progress.configure(value=0)
        hits = data["hits"]
        self.last_hits = hits
        self._show_results(hits)
        times = data["times"]
        ms = lambda k: f"{times.get(k, 0) * 1000:.1f}ms"
        mode = "仅粗筛" if data["coarse_only"] else \
            f"粗筛 {ms('粗筛(特征+扫描)')} + 精排 {ms('精排(查询特征+打分)')}"
        info = (f"库 {data['db']} 张 | 候选 {data['kept']} 张 | "
                f"{mode} | 合计 {ms('total')}"
                + (" | 已剔除查询图自身" if data["self_excluded"] else ""))
        self.result_info_var.set(f"{len(hits)} 个结果  |  {info}")
        self.detail_var.set("点击结果缩略图查看详情")
        self._set_busy(False, f"检索完成，返回 {len(hits)} 个结果")

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
                  self.btn_add_new, self.btn_refresh_idx, self.btn_q):
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
