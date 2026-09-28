# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — Web 界面壳（pywebview）：只读 WSGI 静态/缩略图 + js_api 转发服务层 + 事件推送到 JS
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
Web 界面壳（pywebview + 自建只读 WSGI）。本 docstring 即契约（架构 + 通道 + 路由），
设计取舍见 `frontend/README.md`：

    前端 dist ──WebView2──┬── 只读 HTTP：/ 静态、/thumb/<key> 缩略图、/image/<key> 原图、
                          │              /ui_strings.json、/ui_theme.css
                          └── js_api（Windows 走原生 WebMessageReceived，**不经 HTTP**）
                                  │  同进程直接调用（零 IPC 序列化开销）
                              Api ──┤ 每个方法先校验 `webview.token`
                                  ▼
                        hybrid_search/service.py（唯一编排层）

**通道分工**：一切变异/查询走 `Api`；HTTP 只承载只读资源（路径白名单只暴露 dist、缩略图缓存与
原图缓存，且都要过「索引内路径集合」校验）。
**窗口标题**：取外置 `ui_strings.json` 的 `app.title`（缺省回落内置 `TITLE`）——**页内不重复显示
产品名**（`App.vue` 里没有标题栏），改标题不用重新构建。
**事件**：`service` 事件 → 队列 → 后台线程批量 `window.evaluate_js("window.__ise_event([...])")`；
可视化帧按“最近一帧优先”节流（等价 tkinter 版的有界队列 + 丢中间帧）。
**缩略图**：复用 `hybrid_search/thumbs.py` 的 96px 磁盘缓存；`/thumb/<key>` 命中直接发，
未命中时按「索引内路径集合」校验来源后就地生成（`?p=` 只用于生成，路径永不回显给前端）。

用法（需先构建前端：`cd frontend && pnpm install && pnpm build`）::

    python -E gui_web.py [--gallery <图库根>] [--prefix <索引前缀>] [--debug]

页面（P1–P3 已全量接完）：`LibraryList` / `Search` / `Dedup` / `Params` / `Logs` 五页 + 事件通道，
外加页内大图对比覆盖层（`components/CompareOverlay.vue`）；只读路由另有 `/image/<key>`（原图，
来源校验同缩略图）。维护说明见 `frontend/README.md`，端到端回归 `devtools/verify_web_gui.py`。
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import queue
import re
import sys
import threading
import time

import bottle
import webview

from hybrid_search.io_utils import LOGGER
from hybrid_search.service import (  # noqa: F401  (常量即契约，前端照抄)
    EVENT_LOG, EVENT_PERF_REPORT, EVENT_PHASE_BOUNDARY, EVENT_PROGRESS,
    EVENT_TASK_DONE, EVENT_TASK_ERROR, EVENT_VIZ_FRAME, HIT_FIELDS, OP_ADD,
    OP_BUILD, OP_COMPACT, OP_DEDUP_APPLY, OP_DEDUP_SCAN, OP_SCAN, OP_SEARCH,
    OP_STATS, OP_TILES, PHASE_LABELS, SearchService)
from hybrid_search.store import IndexFiles
from hybrid_search.thumbs import ThumbCache

__all__ = ["Api", "EventBridge", "ThumbService", "make_app", "main"]

TITLE = "二值法粗筛 + ResNet精排 · 混合图库检索 (Web)"
THUMB_KEY_RE = re.compile(r"^[0-9a-f]{32}$")
EVENT_FN = "__ise_event"          # 前端注册的事件入口：window.__ise_event(events[])
UI_EXTERNAL = ("ui_strings.json", "ui_theme.css")   # 外置文案/主题（放包根即可覆盖，优先于内置默认）
THUMB_WAIT_SEC = 1.5              # 缩略图未命中时的“就地生成”等待上限（超时返回 202，前端重试）


def base_dir() -> str:
    """包根：PyInstaller onedir 下为 `_internal`，源码运行为本文件所在目录。"""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def resolve_dist() -> str:
    """定位前端产物 `frontend/dist`（源码运行 / 冻结包两种形态）。"""
    cands = [os.path.join(base_dir(), "frontend", "dist"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "frontend", "dist"),
             os.path.join(os.getcwd(), "frontend", "dist")]
    for c in cands:
        if os.path.isfile(os.path.join(c, "index.html")):
            return c
    return cands[0]


def external_ui_strings() -> dict:
    """读外置文案 `ui_strings.json`（放包根 / 仓库根即可覆盖；缺失或损坏时返回空 dict）。"""
    try:
        with open(os.path.join(base_dir(), UI_EXTERNAL[0]), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:                       # noqa: BLE001 —— 缺文件 / 坏 JSON 都不影响启动
        return {}


def window_title() -> str:
    """系统窗口标题：优先用外置文案的 `app.title`（改它**不需要重新构建**），否则用内置 `TITLE`。

    页面里**不再重复**显示产品名（窗口标题已经有一份，页内重复是冗余），所以这里是产品名的
    唯一展示位，也是"免 Node 改文案"能覆盖到的一处。
    """
    v = external_ui_strings().get("app.title")
    return v.strip() if isinstance(v, str) and v.strip() else TITLE


def b64u(text: str) -> str:
    """URL 安全的 base64（缩略图生成用的源路径参数）。"""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def unb64u(text: str) -> str:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode(
            "utf-8", "replace")
    except Exception:                                    # noqa: BLE001
        return ""


def _image_mime(name: str) -> str:
    """按扩展名给 Content-Type（浏览器对图片还会自行嗅探，这里只求稳妥）。"""
    ext = os.path.splitext(str(name))[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
            ".tif": "image/tiff", ".tiff": "image/tiff"}.get(
                ext, "application/octet-stream")


def viz_payload(ev: dict) -> dict:
    """把 service 的 `viz_frame`（numpy 帧）转成 JSON 安全的 base64 载荷。

    粗筛帧：64×64 二值点阵（1 字节/点）；精排帧：16×16×3 原始 RGB。
    """
    import numpy as np
    kind = str(ev.get("kind") or "")
    arr = np.asarray(ev.get("data"))
    out = {"event": EVENT_VIZ_FRAME, "task_id": ev.get("task_id", ""),
           "kind": kind}
    if kind == "fine" and arr.ndim == 3:
        out["w"], out["h"] = int(arr.shape[1]), int(arr.shape[0])
        raw = arr.astype(np.uint8, copy=False).tobytes()
    else:
        n = int(arr.shape[0]) if arr.ndim == 2 else int(round(arr.size ** 0.5))
        out["w"] = out["h"] = n
        raw = arr.reshape(-1).astype(np.uint8, copy=False).tobytes()
    out["b64"] = base64.b64encode(raw).decode("ascii")
    return out


def json_default(obj):
    """事件载荷的 JSON 兜底：numpy 数组/标量、dataclass、集合、元组都能安全出网。

    ⚠️ 事件是**整批**序列化的（`EventBridge._flush`），任何一个不可序列化的值都会
    让整批事件被丢掉 —— 所以这里必须兜住（如 `box` 的 numpy 数组、`DupReport`）。
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "tolist"):          # numpy 数组 / 标量
        return obj.tolist()
    if hasattr(obj, "item"):            # numpy 标量（老式）
        return obj.item()
    return str(obj)


def enrich_dedup_report(svc: SearchService, res) -> dict:
    """去重报告加工（前端只做渲染，语义计算留在 Python 侧）：

    * 组：`n_members`、`wasted_bytes`（除首张外可释放的字节）、`all_exact`、`title`；
    * 成员：`name`（文件名）、`thumb`（`/thumb/<key>?p=…`，key 优先用组内已有 md5）；
    * 顶层：`n_exact` / `n_near` / `n_images` / `wasted_bytes`。

    `res` 既可能是**服务层原样的 `DupReport` dataclass**（事件桥在 JSON 化之前加工），
    也可能是已经序列化过的 dict（历史事件/自检），两种都接。
    """
    if dataclasses.is_dataclass(res) and not isinstance(res, type):
        res = dataclasses.asdict(res)
    if not res or not isinstance(res, dict) or "groups" not in res:
        return res
    groups = res.get("groups") or []
    paths: list = []
    for g in groups:
        for m in (g.get("members") or []):
            paths.append(m.get("path") or "")
    keys = svc.thumb_keys(paths) if paths else []
    out_groups = []
    k = 0
    for gi, g in enumerate(groups, 1):
        members = []
        for m in (g.get("members") or []):
            md5 = m.get("md5") or ""
            path = m.get("path") or ""
            key = md5 or (keys[k] if k < len(keys) else ThumbCache.key_for(path))
            k += 1
            item = dict(m)
            item["name"] = os.path.basename(path)
            item["thumb"] = f"/thumb/{key}?p={b64u(path)}" if path else ""
            members.append(item)
        all_exact = bool(members) and all(m.get("exact_copy") for m in members)
        wasted = sum(int(m.get("size") or 0) for m in members[1:])
        kind_txt = "完全重复" if all_exact else "近似重复"
        out = dict(g)
        out["members"] = members
        out["n_members"] = len(members)
        out["all_exact"] = all_exact
        out["wasted_bytes"] = wasted
        out["title"] = f"组 {g.get('gid', gi)} · {kind_txt} · {len(members)} 张"
        out_groups.append(out)
    out = dict(res)
    out["groups"] = out_groups
    out["n_exact"] = sum(1 for g in out_groups if g["all_exact"])
    out["n_near"] = sum(1 for g in out_groups if not g["all_exact"])
    out["n_images"] = sum(g["n_members"] for g in out_groups)
    out["wasted_bytes"] = sum(g["wasted_bytes"] for g in out_groups)
    return out


def enrich_search_result(svc: SearchService, res: dict | None) -> dict | None:
    """检索结果加工：命中元组 → 具名字段 + `thumb`（缩略图 URL，供前端 <img>）。"""
    if not res or not isinstance(res, dict):
        return res
    hits = res.get("hits") or []
    if not hits or isinstance(hits[0], dict):
        return res
    paths = [h[1] for h in hits]
    keys = svc.thumb_keys(paths)
    out_hits = []
    for i, h in enumerate(hits):
        item = dict(zip(HIT_FIELDS, h))
        item["thumb"] = (f"/thumb/{keys[i]}?p={b64u(h[1])}"
                         if i < len(keys) else "")
        out_hits.append(item)
    out = dict(res)
    out["hits"] = out_hits
    return out


# ==========================================================================
# 事件桥：service 事件 -> JS（批量 + 节流；viz 帧“最近一帧优先”）
# ==========================================================================
class EventBridge:
    """把服务层事件汇成 `window.__ise_event(events)` 调用（任意线程可投递）。"""

    def __init__(self, window_getter, service: SearchService | None = None,
                 interval: float = 0.04, max_logs: int = 200) -> None:
        self._get_window = window_getter
        self.svc = service
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._interval = max(0.01, float(interval))
        self._max_logs = int(max_logs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ready = threading.Event()      # 前端调用 Api.ready() 后置位
        self.sent = 0                       # 已推送事件数（自检/回归用）
        self.dropped = 0                    # 未就绪期间丢弃的事件数

    # ---- 服务层订阅入口 ----
    def on_event(self, ev: dict) -> None:
        self._q.put(ev)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ise-events")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- 内部 ----
    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self._interval)
            if not self.ready.is_set():
                self._drain_dropped()
                continue
            batch = self._collect()
            if batch:
                self._flush(batch)

    def _drain_dropped(self) -> None:
        while True:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                return

    def _collect(self) -> list:
        """取空队列并合并：log 全留（截断）、progress 每任务留最新、
        viz 每类型留最新、阶段边界/任务收尾/性能图全留。"""
        got = []
        while True:
            try:
                got.append(self._q.get_nowait())
            except queue.Empty:
                break
        logs, prog, viz, rest = [], {}, {}, []
        for ev in got:
            kind = ev.get("event")
            if kind == EVENT_LOG:
                logs.append(ev)
            elif kind == EVENT_PROGRESS:
                prog[ev.get("task_id")] = ev
            elif kind == EVENT_VIZ_FRAME:
                viz[ev.get("kind")] = ev
            else:
                rest.append(ev)
        if len(logs) > self._max_logs:
            logs = logs[-self._max_logs:]
        out = logs + list(prog.values())
        for kind in ("coarse", "fine"):          # 顺序观感：粗筛帧先于精排帧
            if kind in viz:
                out.append(viz_payload(viz[kind]))
        for ev in rest:                          # 结果加工（检索命中 / 去重报告）
            if ev.get("event") == EVENT_TASK_DONE and self.svc is not None:
                op = ev.get("op")
                if op == OP_SEARCH:
                    ev = dict(ev)
                    ev["result"] = enrich_search_result(self.svc, ev.get("result"))
                elif op == OP_DEDUP_SCAN:
                    ev = dict(ev)
                    ev["result"] = enrich_dedup_report(self.svc, ev.get("result"))
            out.append(ev)
        return out

    def _flush(self, batch: list) -> None:
        win = self._get_window()
        if win is None:
            return
        script = ("window.%s && window.%s(%s);" % (
            EVENT_FN, EVENT_FN,
            json.dumps(batch, ensure_ascii=False, separators=(",", ":"),
                       default=json_default)))
        try:
            win.evaluate_js(script)
            self.sent += len(batch)
        except Exception:                                # noqa: BLE001
            pass                    # 窗口正在关闭/尚未就绪：丢这一批，绝不打断任务


# ==========================================================================
# 缩略图：命中直接发；未命中按“索引内路径集合”校验后就地生成
# ==========================================================================
class ThumbService:
    """`<索引目录>/thumbs/<前2位>/<key>.jpg`（复用 `hybrid_search.thumbs`）。"""

    def __init__(self, service: SearchService, workers: int = 4) -> None:
        self.svc = service
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="ise-thumb")
        self._futures: dict = {}
        self._lock = threading.Lock()
        self._caches: dict = {}
        self.generated = 0

    def cache_for(self, prefix: str) -> ThumbCache:
        root = os.path.dirname(prefix) if prefix else "."
        with self._lock:
            cache = self._caches.get(root)
            if cache is None:
                cache = ThumbCache(root)
                self._caches[root] = cache
            return cache

    def path_of(self, prefix: str, key: str) -> str:
        return self.cache_for(prefix).file_of(key)

    def ensure(self, prefix: str, key: str, src_path: str) -> bool:
        """确保缩略图存在；已存在立即 True，否则排队生成并最多等 THUMB_WAIT_SEC。"""
        cache = self.cache_for(prefix)
        if cache.exists(key):
            return True
        if not src_path:
            return False
        with self._lock:
            fut = self._futures.get(key)
            if fut is None:
                fut = self._pool.submit(self._make, cache, key, src_path)
                self._futures[key] = fut
        try:
            if fut.result(timeout=THUMB_WAIT_SEC):
                self.generated += 1
                return True
        except Exception:                                # noqa: BLE001
            pass
        return False

    @staticmethod
    def _make(cache: ThumbCache, key: str, src_path: str) -> bool:
        return cache.make(src_path, key) is not None


# ==========================================================================
# 只读 WSGI：前端 dist + 缩略图 + 外置文案/主题（**没有任何写操作**）
# ==========================================================================
def make_app(service: SearchService, dist: str, thumbs: ThumbService,
             package_dir: str | None = None) -> bottle.Bottle:
    app = bottle.Bottle()
    pkg = package_dir or os.path.dirname(os.path.abspath(__file__))

    def _src_allowed(src: str) -> bool:
        """缩略图生成源只允许：图库根目录之下、或当前索引里的路径（防目录穿越）。"""
        try:
            norm = os.path.normcase(os.path.abspath(src))
        except Exception:                                # noqa: BLE001
            return False
        if not os.path.isfile(norm):
            return False
        root = os.path.normcase(os.path.abspath(service.root)) if service.root else ""
        if root and (norm == root or norm.startswith(root + os.sep)):
            return True
        try:
            return norm in service.index_md5_map()
        except Exception:                                # noqa: BLE001
            return False

    def static_or_index(path: str):
        full = os.path.join(dist, path)
        if os.path.isfile(full):
            return bottle.static_file(path, root=dist)
        if path.endswith((".json", ".css", ".js", ".png", ".svg", ".ico", ".map")):
            return bottle.HTTPResponse(status=404, body="not found")
        return bottle.static_file("index.html", root=dist)   # SPA 路由回落

    @app.get("/")
    def index():                                        # noqa: D401
        return static_or_index("index.html")

    @app.get("/ui_strings.json")
    def ui_strings():                                   # noqa: D401
        p = os.path.join(pkg, UI_EXTERNAL[0])
        if os.path.isfile(p):
            return bottle.static_file(UI_EXTERNAL[0], root=pkg,
                                      mimetype="application/json")
        return bottle.HTTPResponse(status=404, body="no override")

    @app.get("/ui_theme.css")
    def ui_theme():                                     # noqa: D401
        p = os.path.join(pkg, UI_EXTERNAL[1])
        if os.path.isfile(p):
            return bottle.static_file(UI_EXTERNAL[1], root=pkg,
                                      mimetype="text/css")
        return bottle.HTTPResponse(status=404, body="no override")

    @app.get("/thumb/<key>")
    def thumb(key: str):                                # noqa: D401
        if not THUMB_KEY_RE.match(key):
            return bottle.HTTPResponse(status=400, body="bad key")
        cache = thumbs.cache_for(service.current_prefix())
        rel = key[:2] + "/" + key + ".jpg"
        if cache.exists(key):
            return bottle.static_file(rel, root=cache.dir, mimetype="image/jpeg")
        src = unb64u(bottle.request.query.get("p") or "")
        if src and not _src_allowed(src):
            return bottle.HTTPResponse(status=403, body="path not in gallery")
        if thumbs.ensure(service.current_prefix(), key, src) and cache.exists(key):
            return bottle.static_file(rel, root=cache.dir, mimetype="image/jpeg")
        bottle.response.set_header("Retry-After", "0.3")   # 正在生成：前端稍后重试
        return bottle.HTTPResponse(status=202, body="generating")

    @app.get("/image/<key>")
    def image(key: str):                                # noqa: D401
        """原图（大图对比用）：只读、且必须通过与缩略图同一套来源校验。"""
        if not THUMB_KEY_RE.match(key):
            return bottle.HTTPResponse(status=400, body="bad key")
        src = unb64u(bottle.request.query.get("p") or "")
        if not src or not _src_allowed(src):
            return bottle.HTTPResponse(status=403, body="path not in gallery")
        head, name = os.path.split(src)
        bottle.response.set_header("Cache-Control", "private, max-age=600")
        return bottle.static_file(name, root=head, mimetype=_image_mime(name))

    @app.get("/health")
    def health():                                       # noqa: D401
        st = service.index_status()
        return {"ok": True, "prefix": st["prefix"],
                "full_exists": st["full_exists"],
                "tiles_exists": st["tiles_exists"],
                "thumbs_generated": thumbs.generated, "dist": dist}

    # ⚠️ 通配规则必须**最后注册**：bottle 的动态路由按注册顺序匹配，
    #    放在前面会把 /thumb/<key> 等规则全部吃掉（静态规则走精确匹配、不受影响）。
    @app.get("/<path:path>")
    def static_files(path: str):                        # noqa: D401
        return static_or_index(path)
    return app


# ==========================================================================
# js_api：所有变异/查询入口（Windows 上经原生 postMessage，不经 HTTP）
# ==========================================================================
class Api:
    """前端调用面。约定：

    * **长任务方法**（scan / build_index / add_index / tiles_index / compact /
      search / dedup_scan / stats）立即返回 `{"ok":True,"accepted":True,"task_id":…}`，
      结果通过 `task_done` / `task_error` 事件回传（前端按 task_id 关联）；
      任务体在独立线程里跑，**绝不阻塞 UI 线程**。
    * **短方法**（配置/状态/对话框/打开原图/导出）同步返回结果。
    * 每个方法都先校验 `token == webview.token`（Gate 4：副作用防护）。
    """

    def __init__(self, service: SearchService, window_getter=None,
                 require_token: bool = True) -> None:
        self.svc = service
        self._get_window = window_getter or (lambda: None)
        self.require_token = bool(require_token)
        self.ready_flag = threading.Event()
        self.selftest_event = threading.Event()   # 前端自检回传（P1 Gate 5）
        self.selftest_result: dict = {}

    # ---- 校验 / 就绪 ----
    def _guard(self, token) -> None:
        if not self.require_token:
            return
        expected = getattr(webview, "token", None)
        if not expected or str(token or "") != str(expected):
            raise PermissionError("bad token")

    def ready(self, token=None) -> dict:
        """前端注册好事件处理器后调用（之后事件才开始推送）。"""
        self._guard(token)
        self.ready_flag.set()
        return {"ok": True, "ready": True}

    def ping(self, token=None) -> str:
        self._guard(token)
        return "pong"

    def version(self, token=None) -> dict:
        self._guard(token)
        return {"title": window_title(), "phases": PHASE_LABELS,
                "hits": list(HIT_FIELDS), "dist": resolve_dist(),
                "frozen": bool(getattr(sys, "frozen", False))}

    # ---- 参数 ----
    def get_config_schema(self, token=None) -> dict:
        self._guard(token)
        return self.svc.get_config_schema()

    def get_config(self, token=None) -> dict:
        self._guard(token)
        return self.svc.get_config()

    def set_config(self, values: dict | None = None, token=None) -> dict:
        self._guard(token)
        return {"ok": True, "applied": self.svc.set_config(values or {})}

    def cli_command(self, op: str = "build", token=None) -> dict:
        """等价 CLI 命令提示（T1）——界面把当前参数交给服务层拼。"""
        self._guard(token)
        return {"ok": True, "command": self.svc.cli_command(op)}

    def set_location(self, root: str | None = None,
                     prefix: str | None = None, token=None) -> dict:
        self._guard(token)
        return {"ok": True, "location": self.svc.set_location(root, prefix),
                "status": self.svc.index_status()}

    def index_status(self, prefix: str | None = None, token=None) -> dict:
        self._guard(token)
        return self.svc.index_status(prefix or None)

    def indexed_paths(self, prefix: str | None = None, token=None) -> dict:
        self._guard(token)
        return {"ok": True, "paths": self.svc.indexed_paths(prefix or None)}

    def latest_perf_report(self, token=None) -> dict:
        self._guard(token)
        return {"ok": True, "path": self.svc.latest_perf_report()}

    # ---- 对话框（pywebview 原生）----
    def confirm(self, title: str = "确认", message: str = "",
                token=None) -> dict:
        """原生确认框（不依赖页面 JS 对话框，与 `console=False` 冻结包兼容）。"""
        self._guard(token)
        win = self._get_window()
        if win is None:
            return {"ok": False, "error": "无窗口（无头模式）"}
        try:
            return {"ok": bool(win.create_confirmation_dialog(title, message))}
        except Exception as e:                           # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def choose_directory(self, title: str = "选择目录", token=None) -> dict:
        self._guard(token)
        win = self._get_window()
        res = win.create_file_dialog(webview.FOLDER_DIALOG, directory="",
                                     allow_multiple=False) if win else None
        return {"ok": True, "path": (res[0] if res else "")}

    def choose_file(self, title: str = "选择图片", token=None) -> dict:
        self._guard(token)
        win = self._get_window()
        res = win.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False,
            file_types=("图片 (*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp)",
                        "所有文件 (*.*)")) if win else None
        return {"ok": True, "path": (res[0] if res else "")}

    def choose_save_file(self, title: str = "保存为", default_name: str = "",
                         token=None) -> dict:
        self._guard(token)
        win = self._get_window()
        res = win.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=default_name or "",
            file_types=("PNG 图片 (*.png)",)) if win else None
        return {"ok": True, "path": (res[0] if res else "")}

    # ---- 缩略图 URL（前端只拿 URL，不碰文件系统）----
    def thumb_url(self, path: str = "", token=None) -> dict:
        self._guard(token)
        keys = self.svc.thumb_keys([path])
        key = keys[0] if keys else ThumbCache.key_for(path)
        return {"ok": True, "key": key,
                "url": f"/thumb/{key}?p={b64u(path)}"}

    def image_url(self, path: str = "", token=None) -> dict:
        """原图 URL（大图对比用；服务端仍按“图库内/索引内”校验来源）。"""
        self._guard(token)
        keys = self.svc.thumb_keys([path])
        key = keys[0] if keys else ThumbCache.key_for(path)
        return {"ok": True, "key": key,
                "url": f"/image/{key}?p={b64u(path)}"}

    # ---- 杂项（只读/系统动作）----
    def open_in_shell(self, path: str = "", token=None) -> dict:
        self._guard(token)
        ok, err = SearchService.open_in_shell(path)
        return {"ok": ok, "error": err}

    def export_sheet(self, paths: list | None = None, out_path: str = "",
                     token=None) -> dict:
        self._guard(token)
        return {"ok": SearchService.export_sheet(list(paths or []), out_path)}

    def cancel(self, task_id: str = "", token=None) -> dict:
        self._guard(token)
        self.svc.cancel(str(task_id))
        return {"ok": True}

    # ---- 自检（P1 Gate 5：真实前端跑一遍并把测量结果回传）----
    def selftest_report(self, metrics=None, token=None) -> dict:
        self._guard(token)
        self.selftest_result = metrics or {}
        self.selftest_event.set()
        return {"ok": True}

    # ---- 长任务：线程里跑，结果走 task_done / task_error 事件 ----
    # ⚠️ js_api 的方法是被 `func(*params)` **按位置调用**的（pywebview util.js_bridge_call），
    #    因此这里一律不用 keyword-only（`*`）参数；前端按下面的顺序传位置参数。
    def _dispatch(self, op: str, fn, task_id: str = "") -> dict:
        tid = str(task_id or f"{op}-{int(time.time() * 1000)}")

        def _job() -> None:
            fn(tid)

        threading.Thread(target=_job, daemon=True,
                         name=f"ise-{op}").start()
        return {"ok": True, "accepted": True, "task_id": tid, "op": op}

    def scan(self, task_id="", root="", recursive=True, verify=False,
             token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_SCAN, lambda t: self.svc.scan(
            t, root, recursive=bool(recursive), verify=bool(verify)), task_id)

    def build_index(self, task_id="", prefix=None, paths=None, img_dir=None,
                    force=True, perf=False, title="", token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_BUILD, lambda t: self.svc.build_index(
            t, prefix or None, paths=list(paths) if paths else None,
            img_dir=img_dir or None, force=bool(force), perf=bool(perf),
            title=title), task_id)

    def add_index(self, task_id="", prefix=None, paths=None, img_dir=None,
                  perf=False, title="", token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_ADD, lambda t: self.svc.add_index(
            t, prefix or None, paths=list(paths) if paths else None,
            img_dir=img_dir or None, perf=bool(perf), title=title), task_id)

    def tiles_index(self, task_id="", tiles_prefix=None, paths=None,
                    img_dir=None, exists=None, perf=False, token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_TILES, lambda t: self.svc.tiles_index(
            t, tiles_prefix or None, paths=list(paths) if paths else None,
            img_dir=img_dir or None,
            exists=None if exists is None else bool(exists),
            perf=bool(perf)), task_id)

    def compact(self, task_id="", prefixes=None, perf=False, token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_COMPACT, lambda t: self.svc.compact(
            t, list(prefixes) if prefixes else None, perf=bool(perf)), task_id)

    def search(self, task_id="", query="", mode="full", prefix=None,
               coarse_k=None, top_k=None, perf=False, token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_SEARCH, lambda t: self.svc.search(
            t, query, mode, prefix=prefix or None, coarse_k=coarse_k,
            top_k=top_k, perf=bool(perf)), task_id)

    def dedup_scan(self, task_id="", paths=None, threshold=0.02, prefix=None,
                   token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_DEDUP_SCAN, lambda t: self.svc.dedup_scan(
            t, list(paths or []), threshold=float(threshold),
            prefix=prefix or None), task_id)

    def dedup_delete(self, task_id="", paths=None, prefix=None, sync=True,
                     token=None) -> dict:
        """把选中的重复图移入回收站，并按 `sync` 决定是否同步索引（prune）。"""
        self._guard(token)
        return self._dispatch(OP_DEDUP_APPLY, lambda t: self.svc.dedup_delete(
            t, list(paths or []), prefix=prefix or None,
            sync=bool(sync)), task_id)

    def dedup_move(self, task_id="", paths=None, dest="", base_root=None,
                   prefix=None, sync=True, token=None) -> dict:
        """把选中的重复图移动到目标图库（保留相对结构），并按 `sync` 同步索引。"""
        self._guard(token)
        return self._dispatch(OP_DEDUP_APPLY, lambda t: self.svc.dedup_move(
            t, list(paths or []), dest, base_root=base_root or None,
            prefix=prefix or None, sync=bool(sync)), task_id)

    def stats(self, task_id="", prefix=None, token=None) -> dict:
        self._guard(token)
        return self._dispatch(OP_STATS, lambda t: self.svc.stats(
            t, prefix or None), task_id)

    def release_engines(self, reason="", token=None) -> dict:
        self._guard(token)
        return {"ok": True, "freed_mb": self.svc.release_engines(reason)}


# ==========================================================================
# 窗口 / 入口
# ==========================================================================
def launch(gallery: str = "", prefix: str = "", debug: bool = False,
           block: bool = True, on_start=None) -> dict:
    """建窗口并进入 GUI 循环；返回上下文。

    `on_start`：GUI 起来后要执行的探针（pywebview 会在独立线程里调它，用于回归/Gate 探针）；
    探针收到本函数返回的上下文字典，可自行 `window.evaluate_js(...)` 并结束窗口。
    """
    dist = resolve_dist()
    if not os.path.isfile(os.path.join(dist, "index.html")):
        raise FileNotFoundError(
            f"未找到前端产物：{dist}\n请先在 frontend/ 下执行 pnpm install && "
            "pnpm build（详见 frontend/README.md）")

    svc = SearchService()
    if gallery:
        svc.set_location(root=gallery, prefix=prefix or None)
    thumbs = ThumbService(svc)
    holder: dict = {}

    def _win():
        return holder.get("window")

    bridge = EventBridge(_win, svc)
    svc.subscribe(bridge.on_event)
    api = Api(svc, _win)
    app = make_app(svc, dist, thumbs)
    win = webview.create_window(window_title(), url=app, js_api=api, width=1500,
                                height=940, min_size=(1180, 720),
                                text_select=True, confirm_close=False)
    holder["window"] = win

    def _on_loaded() -> None:
        # 前端 boot() 会在注册好事件入口后调 ready()；这里只负责起事件线程，
        # **不要**清 ready_flag（loaded 可能晚于 ready，清了就永远收不到事件）。
        bridge.start()
        try:
            LOGGER.info("Web 界面已加载：origin=%s（dist=%s）",
                        getattr(win, "real_url", ""), dist)
        except Exception:                                # noqa: BLE001
            pass

    win.events.loaded += _on_loaded

    def _watch_ready() -> None:
        """前端 ready() 后放行事件推送（兜底：60s 后无条件放行并记一条日志）。"""
        t0 = time.time()
        while not api.ready_flag.is_set():
            if time.time() - t0 > 60:
                LOGGER.warning("前端 60s 内未调用 ready()：仍放行事件推送（可能丢首帧日志）")
                break
            time.sleep(0.05)
        bridge.ready.set()

    threading.Thread(target=_watch_ready, daemon=True, name="ise-ready").start()

    ctx = {"window": win, "service": svc, "api": api, "bridge": bridge,
           "thumbs": thumbs, "app": app, "dist": dist}
    if block:
        try:
            webview.start(func=on_start, args=(ctx,), debug=bool(debug))
        finally:
            bridge.stop()
            svc.close()
    return ctx


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Web 界面（pywebview + Vue3）；检索逻辑全在 hybrid_search/service.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gallery", default="", help="图库根目录（预填）")
    ap.add_argument("--prefix", default="", help="索引前缀（留空=按图库根推导）")
    ap.add_argument("--debug", action="store_true", help="pywebview 调试模式")
    return ap.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:                            # noqa: BLE001
                pass
    a = _parse_args(argv)
    launch(gallery=a.gallery, prefix=a.prefix, debug=a.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())

    # ---- 命中结果加工：补 `thumb`（缩略图 URL）与字段名 ----
    def _enrich(self, res: dict | None) -> dict | None:
        if not res:
            return res
        hits = res.get("hits") or []
        paths = [h[1] for h in hits]
        keys = self.svc.thumb_keys(paths) if paths else []
        out_hits = []
        for i, h in enumerate(hits):
            item = dict(zip(HIT_FIELDS, h))
            item["thumb"] = (f"/thumb/{keys[i]}?p={b64u(h[1])}"
                             if i < len(keys) else "")
            out_hits.append(item)
        res = dict(res)
        res["hits"] = out_hits
        return res


# @@TAIL@@
