# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 验证：Web 界面（gui_web + frontend）无头回归与 Gate 1/3/4/5/6
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
验证 Web 界面层（`gui_web.py` + `frontend/`）的端到端回归。

判据（Gate）：
  * **Gate 1** 起窗 + `js_api` 往返（真窗口模式）；
  * **Gate 3** HTTP 只监听回环 `127.0.0.1`（真窗口模式）；
  * **Gate 4** 错误 token 的 `Api` 调用被拒（无头 + 真窗口）；
  * **Gate 5** 一屏缩略图全部落定（吞吐/失败计数，真窗口模式）；
  * **Gate 6** 亮/暗色切换：两套配色都真的生效（body 背景色不同）+ 点击按钮即切（真窗口模式）；
  * 无头部分另断言：只读路由（`/`、`/thumb/<key>`、`/image/<key>`、`/ui_strings.json`）、
    命令面（scan/build/search/dedup_*）、事件契约（`log`/`progress`/`phase_boundary`/…）、
    去重删除/移动后的索引同步。
设计取舍与页面↔命令对照见 `frontend/README.md`。

两种模式：

* **默认（无头）**：不起窗口，直接对只读 WSGI 路由 + `Api` 命令面 + 事件桥做断言
  （HTTP 用进程内 wsgiref，事件用假窗口抓 `evaluate_js` 脚本）——
  适合无桌面会话的 CI/远程环境；
* **`--window`**：起**真实 WebView2 窗口 + 真实前端**，用前端自检钩子 `__ise_selftest`
  测“一屏 60 张缩略图首屏”，并复跑 Gate 1（桥/静态）、Gate 3（只监听回环）、
  Gate 4（错误 token 被拒）。

判据：全部断言 ✓ 且退出码 0。
用法::

    python -E devtools/verify_web_gui.py [--db 120] [--no-window] [--keep]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import gui_web  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.service import SearchService  # noqa: E402

FAILED = []


def check(cond, text: str, detail: str = "") -> bool:
    tag = "✓" if cond else "✗"
    print(f"  {tag} {text}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(text)
    return bool(cond)


class ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class FakeWin:
    """假窗口：只实现事件桥需要的 `evaluate_js`（把脚本抓下来）。"""

    def __init__(self) -> None:
        self.scripts = []
        self.lock = threading.Lock()

    def evaluate_js(self, script: str):
        with self.lock:
            self.scripts.append(script)
        return None


class Recorder:
    """服务层事件记录器（带等待）。"""

    def __init__(self) -> None:
        self.events = []
        self.cv = threading.Condition()

    def __call__(self, ev: dict) -> None:
        with self.cv:
            self.events.append(ev)
            self.cv.notify_all()

    def wait(self, pred, timeout: float = 600.0):
        t0 = time.time()
        with self.cv:
            while time.time() - t0 < timeout:
                for ev in self.events:
                    if pred(ev):
                        return ev
                self.cv.wait(0.1)
        return None

    def of(self, kind: str):
        return [e for e in self.events if e.get("event") == kind]


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get(url: str, timeout: float = 30.0, retries: int = 5,
             delay: float = 0.3):
    """GET 回环 URL（**绕开系统代理**）。

    连接级失败会重试 `retries` 次（本机防火墙/杀软偶尔会拦新起的监听套接字），
    最终仍失败则返回 `(-1, 错误文本, {})` —— 不吞掉失败，调用方按状态码断言。
    """
    last = (-1, b"", {})
    for i in range(max(1, retries)):
        try:
            with _OPENER.open(url, timeout=timeout) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers or {})
        except Exception as e:                           # noqa: BLE001
            last = (-1, f"{type(e).__name__}: {e}".encode("utf-8", "replace"), {})
            if i + 1 < retries:
                time.sleep(delay)
    return last


def wait_http_ready(base: str, timeout: float = 15.0) -> float:
    """等服务真的能连上（返回耗时秒）；超时返回 -1（调用方据此报失败）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, _b, _h = http_get(base + "/health", timeout=3.0, retries=1)
        if st == 200:
            return time.time() - t0
        time.sleep(0.25)
    return -1.0


def start_wsgi(app):
    srv = make_server("127.0.0.1", 0, app, server_class=ThreadedWSGIServer)
    port = srv.server_port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------- 无头部分
def headless(gallery: str, dist: str) -> int:
    print("\n== A) 只读 WSGI 路由（HTTP，进程内 wsgiref）==")
    svc = SearchService(cfg=Config(), capture_log=False)
    svc.set_location(root=gallery, prefix=None)
    rec = Recorder()
    svc.subscribe(rec)
    fake = FakeWin()
    bridge = gui_web.EventBridge(lambda: fake, svc)
    svc.subscribe(bridge.on_event)
    thumbs = gui_web.ThumbService(svc)
    app = gui_web.make_app(svc, dist, thumbs)
    srv, base = start_wsgi(app)
    ready = wait_http_ready(base)
    check(ready >= 0, "进程内只读 HTTP 服务可连通", f"{ready * 1000:.0f} ms" if ready >= 0 else "超时")
    try:
        st, body, _h = http_get(base + "/")
        check(st == 200 and b'id="app"' in body, "GET / 返回前端 index.html",
              f"HTTP {st} / {len(body)} 字节")
        m = re.search(rb'src="(\./assets/[^"]+\.js)"', body)
        asset = (base + "/" + m.group(1).decode().lstrip("./")) if m else ""
        st2, body2, _ = http_get(asset) if asset else (-1, b"", {})
        check(st2 == 200 and len(body2) > 1000, "GET 前端 JS 资源 200",
              f"HTTP {st2} / {len(body2)} 字节")
        st3, body3, _ = http_get(base + "/ui_strings.json")
        check(st3 == 200 and json.loads(body3).get("_comment"),
              "GET /ui_strings.json 外置文案可读", f"HTTP {st3}")
        st4, body4, _ = http_get(base + "/health")
        check(st4 == 200 and json.loads(body4)["ok"], "GET /health 正常")
        st5, _b5, _ = http_get(base + "/nope.js")
        check(st5 == 404, "缺失静态资源返回 404（不回落 index.html）", f"HTTP {st5}")
        st6, body6, _ = http_get(base + "/some/spa/route")
        check(st6 == 200 and b'id="app"' in body6, "SPA 路由回落 index.html")
        st7, body7, _ = http_get(base + "/../LICENSE")
        check(b"GNU AFFERO" not in body7, "目录穿越被拒（拿不到仓库文件）",
              f"HTTP {st7} / {len(body7)} 字节")

        paths = svc.scan("t-scan", gallery)["paths"]
        probe = paths[0]
        key = svc.thumb_keys([probe])[0]
        st8, _b8, _ = http_get(base + "/thumb/not-a-key")
        check(st8 == 400, "缩略图 key 非法 → 400", f"HTTP {st8}")
        outside = gui_web.b64u(os.path.join(gallery, "oops.jpg"))
        st9, _b9, _ = http_get(f"{base}/thumb/{key}?p={outside}")
        check(st9 in (403, 202), "缩略图生成源不在图库内 → 拒绝/未生成",
              f"HTTP {st9}")
        url = f"{base}/thumb/{key}?p={gui_web.b64u(probe)}"
        st10, body10, hdr10 = http_get(url)
        if st10 == 202:
            time.sleep(0.4)
            st10, body10, hdr10 = http_get(url)
        check(st10 == 200 and body10[:2] == b"\xff\xd8",
              "缩略图就地生成并按 JPEG 返回",
              f"HTTP {st10} / {len(body10)} 字节 / {hdr10.get('Content-Type')}")
    except Exception as e:                               # noqa: BLE001
        import traceback
        check(False, "WSGI 路由断言异常（不掩盖）", repr(e))
        traceback.print_exc()
    return 0, svc, rec, fake, bridge, srv, base


def api_checks(gallery: str, svc, rec, fake, bridge, base: str) -> int:
    print("\n== B) Api 命令面（token 校验 + 长任务事件关联）==")
    api_bad = gui_web.Api(svc, require_token=True)
    try:
        api_bad.ping("wrong-token")
        check(False, "错误 token 应被拒绝")
    except PermissionError as e:
        check(True, "错误 token 被拒绝（PermissionError）", str(e))
    api = gui_web.Api(svc, lambda: fake, require_token=False)
    check(api.ping() == "pong", "ping 往返")
    schema = api.get_config_schema()
    check(bool(schema["pages"]), "get_config_schema 可用",
          str([p["key"] for p in schema["pages"]]))
    cfg = api.get_config()
    check("coarse_k" in cfg and "png_decoder" in cfg, "get_config 可用",
          f"{len(cfg)} 项")
    loc = api.set_location(gallery, None)
    check(loc["location"]["prefix"].endswith("gallery"),
          "set_location 推导索引前缀", loc["location"]["prefix"])

    def wait_done(tid: str, timeout: float = 900.0):
        ev = rec.wait(lambda e: e.get("task_id") == tid
                      and e.get("event") in ("task_done", "task_error"), timeout)
        if ev is None:
            check(False, f"{tid} 未收到结束事件（超时）")
            return None
        if ev.get("event") == "task_error":
            check(False, f"{tid} 任务失败", str(ev.get("error"))[:200])
            return None
        return ev.get("result")

    ack = api.scan("t1", gallery, True, False)
    check(bool(ack.get("accepted")) and ack.get("task_id") == "t1",
          "scan 立即返回 accepted（不阻塞 UI）", str(ack.get("task_id")))
    scan_res = wait_done("t1")
    check(bool(scan_res) and scan_res["count"] > 0, "scan 结束事件带结果",
          f"{scan_res and scan_res['count']} 张")

    api.build_index("t2", None, scan_res["paths"], None, True, False, "验证·全量建库")
    build_res = wait_done("t2")
    bounds = [e["phase"] for e in rec.of("phase_boundary")
              if e.get("task_id") == "t2"]
    check(bool(build_res) and build_res["n"] == scan_res["count"],
          "build_index 入库张数 = 扫描张数", f"{build_res and build_res['n']}")
    check(bounds[-2:] == ["save", "done"], "phase_boundary 以 save→done 收尾",
          "→".join(bounds))

    queries = sorted(glob.glob(os.path.join(os.path.dirname(gallery), "queries",
                                            "*.jpg")))
    query = queries[0] if queries else scan_res["paths"][0]
    api.search("t3", query, "full", None, None, 60, False)
    sres = wait_done("t3")
    check(bool(sres) and len(sres["hits"]) > 0, "search 结束事件带命中",
          f"{len(sres['hits']) if sres else 0} 条")

    # 瓦片索引（走 OP_TILES），再用 tiles 模式检索：验证 numpy 框也能安全出网
    api.tiles_index("t4", None, scan_res["paths"], None, None, False)
    tres_build = wait_done("t4")
    check(bool(tres_build) and tres_build["n"] > 0, "tiles_index 完成",
          f"{tres_build and tres_build['n']} 块")

    # 事件桥：把真实事件喂给假窗口，检查出网 JSON（numpy/dataclass 兜底 + 命中加工）
    fake.scripts.clear()
    api.search("tb-1", query, "tiles", None, None, 5, False)
    wait_done("tb-1")
    bridge.ready.set()
    for ev in rec.events[-60:]:
        bridge.on_event(ev)
    bridge._flush(bridge._collect())
    payload = None
    for sc in fake.scripts:
        mm = re.search(r"window\.__ise_event\((\[.*\])\);", sc)
        if mm:
            payload = json.loads(mm.group(1))
            break
    check(payload is not None,
          "事件批量 JSON 可解析（numpy 框 / dataclass 已兜底）")
    hits_out = []
    for ev in (payload or []):
        if ev.get("event") == "task_done" and ev.get("op") == "search":
            hits_out = ev["result"]["hits"]
    check(bool(hits_out) and isinstance(hits_out[0], dict) and "thumb" in hits_out[0],
          "检索事件已加工：命中为具名字段 + 缩略图 URL",
          f"hits={len(hits_out)} keys={list(hits_out[0])[:6] if hits_out else []}")
    check(all(not isinstance(h.get("box"), str) for h in hits_out),
          "命中框（numpy 数组）已转普通数组")
    ok_thumb = bool(hits_out) and hits_out[0]["thumb"].startswith("/thumb/")
    check(ok_thumb, "缩略图 URL 形如 /thumb/<key>?p=…",
          hits_out[0]["thumb"][:64] if hits_out else "")
    if ok_thumb:
        st, body, _ = http_get(base + hits_out[0]["thumb"])
        check(st in (200, 202), "该 URL 可被 HTTP 直接取到（首屏路径成立）",
              f"HTTP {st} / {len(body)} 字节")
    api.release_engines("验证")
    return 0


def dedup_checks(gallery: str, svc, rec, fake, bridge, base: str) -> int:
    """P2：去重「应用」链路（删除/移动 + 索引同步）与报告加工（无头即可跑）。"""
    print("\n== B2) 去重应用 + 索引同步（P2）==")
    api = gui_web.Api(svc, lambda: fake, require_token=False)
    images = sorted(glob.glob(os.path.join(gallery, "*.jpg")))

    def wait_done(tid: str, timeout: float = 900.0):
        ev = rec.wait(lambda e: e.get("task_id") == tid
                      and e.get("event") in ("task_done", "task_error"), timeout)
        if ev is None:
            check(False, f"{tid} 未收到结束事件（超时）")
            return None
        if ev.get("event") == "task_error":
            check(False, f"{tid} 任务失败", str(ev.get("error"))[:200])
            return None
        return ev.get("result")

    # 造两张字节相同的副本（完全重复）。⚠️ 名字排在 img_* 之后：报告按
    # （像素数, 体积, mtime）稳定排序取“基准”，这样基准仍是已入库的原件，
    # 才符合“未入库的完全副本”这一勾选助手的语义。
    dup_a = os.path.join(gallery, "zz_dup_copy_a.jpg")
    dup_b = os.path.join(gallery, "zz_dup_copy_b.jpg")
    shutil.copy2(images[0], dup_a)
    shutil.copy2(images[0], dup_b)

    fake.scripts.clear()
    api.dedup_scan("d1", list(images) + [dup_a, dup_b], 0.02, None)
    rep = wait_done("d1")
    if dataclasses.is_dataclass(rep):        # 服务层原样给的是 DupReport，这里按桥的加工口径转
        rep = gui_web.enrich_dedup_report(svc, dataclasses.asdict(rep))
    check(bool(rep) and rep["groups"], "dedup_scan 返回分组",
          f"{rep and len(rep.get('groups') or [])} 组")
    grp = None
    for g in (rep or {}).get("groups", []):
        if any(m["path"] == dup_a for m in g["members"]):
            grp = g
            break
    check(bool(grp) and any(m["path"] == images[0] for m in (grp or {}).get("members", [])),
          "副本与原件同组")
    members = (grp or {}).get("members", [])
    md5_a = next((m.get("md5") for m in members if m["path"] == dup_a), "")
    md5_src = next((m.get("md5") for m in members if m["path"] == images[0]), "")
    check(bool(md5_a) and md5_a == md5_src, "副本与原件 MD5 相同（完全重复）",
          f"md5={bool(md5_a)}")
    check(bool(grp and grp.get("title") and grp.get("n_members", 0) >= 2
               and "wasted_bytes" in grp),
          "报告已加工：组 title / n_members / wasted_bytes / all_exact",
          str({k: (grp or {}).get(k) for k in ("title", "n_members", "wasted_bytes", "all_exact")}))
    first = (grp or {}).get("members", [{}])[0]
    check(bool(first.get("name")) and first.get("thumb", "").startswith("/thumb/"),
          "成员带 name 与缩略图 URL", str(first.get("thumb", ""))[:48])

    # 事件桥：dedup_scan 的 task_done 也必须能在 JS 侧解析（numpy/dataclass 兜底）
    bridge.ready.set()
    for ev in rec.events[-30:]:
        bridge.on_event(ev)
    bridge._flush(bridge._collect())
    payload = None
    for sc in fake.scripts:
        mm = re.search(r"window\.__ise_event\((\[.*\])\);", sc)
        if mm:
            payload = json.loads(mm.group(1))
            break
    dedup_ev = [e for e in (payload or [])
                if e.get("event") == "task_done" and e.get("op") == "dedup_scan"]
    dn = (dedup_ev[-1]["result"] if dedup_ev else {})
    check(bool(dn.get("groups")) and int(dn.get("n_images") or 0) > 0
          and "wasted_bytes" in dn,
          "dedup_scan 事件已加工（n_images / wasted_bytes）并可在 JS 侧解析",
          f"组 {len(dn.get('groups') or [])} / 成员 {dn.get('n_images')} / "
          f"可释放 {dn.get('wasted_bytes')}")

    rows0 = len(svc.indexed_paths())
    out = api.dedup_delete("d2", [dup_a], None, True)
    res = wait_done("d2")
    check(bool(res) and res["removed"] == [dup_a] and not res["failed"],
          "dedup_delete：副本移入回收站", f"removed={len((res or {}).get('removed') or [])}")
    check(not os.path.exists(dup_a), "副本文件已不在图库")
    check(bool(res and any(r.get("removed") == 0 for r in res.get("prune", []))),
          "未入库副本 → prune 剔除 0 条（不误删）", str((res or {}).get("prune")))

    # 删一张“已入库”的图：索引行数应 -1（prune 同步）
    victim = images[1]
    out2 = api.dedup_delete("d3", [victim], None, True)
    res2 = wait_done("d3")
    rows1 = len(svc.indexed_paths())
    check(bool(res2) and res2["removed"] == [victim], "已入库图片删除成功")
    check(rows1 == rows0 - 1, "prune 同步：索引行数 -1", f"{rows0} → {rows1}")
    check(not os.path.exists(victim), "被删文件已离开图库")

    # 移动一张已入库的图：索引行数再 -1（保留相对目录结构）
    dest = os.path.join(os.path.dirname(gallery), "moved_gallery")
    mover = images[2]
    out3 = api.dedup_move("d4", [mover], dest, gallery, None, True)
    res3 = wait_done("d4")
    rows2 = len(svc.indexed_paths())
    moved_to = os.path.join(dest, os.path.basename(mover))
    check(bool(res3) and res3["removed"] == [mover] and res3.get("dest") == dest,
          "dedup_move 成功", str(res3 and res3.get("dest")))
    check(os.path.exists(moved_to) and not os.path.exists(mover),
          "文件已移动到目标图库（保留相对结构）")
    check(rows2 == rows1 - 1, "移动后 prune 同步：索引行数再 -1", f"{rows1} → {rows2}")
    return 0


# ---------------------------------------------------------------- 窗口部分（Gate 1/3/4/5）
def probe_gates(ctx: dict, gallery: str, top_k: int) -> None:
    """在 pywebview 的 GUI 线程里跑 Gate 探针，最后关窗。"""
    win = ctx["window"]
    api = ctx["api"]
    try:
        print("\n== C) Gate 1：起窗 / js_api 往返 / 静态资源（真实前端）==")
        win.events.shown.wait(30)
        # 前端 boot() 会调 ready()（事件入口注册完毕）
        check(api.ready_flag.wait(60), "前端已连上 js_api（ready()）")
        check(win.evaluate_js("typeof window.__ise_event === 'function'") is True,
              "事件入口 window.__ise_event 已注册")
        ntitle = win.evaluate_js("document.querySelectorAll('.title').length")
        check(int(ntitle or 0) == 0,
              "页内不重复产品名（无 .title 元素；产品名只在系统窗口标题栏）", f"{ntitle} 个")
        ntabs = win.evaluate_js("document.querySelectorAll('.tabs button').length")
        check(int(ntabs or 0) >= 6, "页签 + 主题按钮已渲染", f"{ntabs} 个按钮")
        nbtn = win.evaluate_js("document.querySelectorAll('.toolbar button').length")
        check(int(nbtn or 0) >= 8, "工具栏按钮已渲染", f"{nbtn} 个")
        origin = win.evaluate_js("window.location.origin")
        check(str(origin).startswith("http://127.0.0.1:"),
              "窗口 origin 为回环地址", str(origin))

        win.evaluate_js("window.__g1='';"
                        "window.pywebview.api.ping(window.pywebview.token)"
                        ".then(v=>{window.__g1='OK:'+v;})"
                        ".catch(e=>{window.__g1='ERR:'+e;});")
        got = ""
        for _ in range(100):
            got = win.evaluate_js("window.__g1 || ''") or ""
            if got:
                break
            time.sleep(0.1)
        check(got == "OK:pong", "js_api 往返（正确 token）", got)

        print("\n== C2) Gate 6：亮/暗色切换（真实前端）==")
        # 自检钩子在 boot 链尾部安装（loadStrings → loadSchema → refreshStatus 之后），
        # 所以先等它就绪，避免"抢跑"报 not a function（用户看不到这个时序，按钮首屏即渲染）
        for _ in range(60):
            if win.evaluate_js("typeof window.__ise_selftest_theme === 'function'") is True:
                break
            time.sleep(0.25)
        check(win.evaluate_js("typeof window.__ise_selftest_theme === 'function'") is True,
              "主题自检钩子已就绪")
        lt = win.evaluate_js("window.__ise_selftest_theme('light')") or {}
        check(isinstance(lt, dict) and lt.get("theme") == "light" and lt.get("attr") == "light",
              "切到亮色：<html data-theme=light>", str(lt.get("attr")))
        dk = win.evaluate_js("window.__ise_selftest_theme('dark')") or {}
        check(isinstance(dk, dict) and dk.get("theme") == "dark",
              "切回暗色", str(dk.get("attr") or "(默认 :root)"))
        check(bool(lt.get("bodyBg")) and bool(dk.get("bodyBg"))
              and lt.get("bodyBg") != dk.get("bodyBg"),
              "两套配色都真的生效（body 背景色不同）",
              f"light={lt.get('bodyBg')} / dark={dk.get('bodyBg')}")
        # 再走一次真实按钮点击（验证 UI 通路，而不只是自检钩子）
        win.evaluate_js("document.querySelector('.tabs button.theme').click()")
        time.sleep(0.3)
        after = win.evaluate_js("document.documentElement.dataset.theme")
        check(after == "light", "点击主题按钮即切换（DOM 属性随之变化）", str(after))
        win.evaluate_js("window.__ise_selftest_theme('dark')")   # 复位，避免影响后续 Gate

        print("\n== D) Gate 3：只监听回环 ==")
        import psutil
        addrs = []
        for c in psutil.Process().net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN:
                addrs.append(c.laddr.ip if hasattr(c.laddr, "ip") else c.laddr[0])
        loopback_only = all(a in ("127.0.0.1", "::1") for a in addrs)
        check(loopback_only, "只绑定回环（无 0.0.0.0）", str(sorted(set(addrs))))

        print("\n== E) Gate 4：副作用防护 ==")
        st, _b, _ = http_get(f"{origin}/thumb/../../LICENSE")
        win.evaluate_js("window.__g4='';"
                        "window.pywebview.api.ping('bad-token-xxx')"
                        ".then(v=>{window.__g4='OK:'+v;})"
                        ".catch(e=>{window.__g4='ERR:'+e;});")
        got4 = ""
        for _ in range(100):
            got4 = win.evaluate_js("window.__g4 || ''") or ""
            if got4:
                break
            time.sleep(0.1)
        check(got4.startswith("ERR:") and "bad token" in got4,
              "错误 token 被服务端拒绝", got4)

        print("\n== F) Gate 5：一屏缩略图首屏（真实前端自检）==")
        js = ("window.__ise_selftest({gallery:%s, topK:%d, rebuild:true, timeoutMs:120000});"
              % (json.dumps(gallery), top_k))
        win.evaluate_js(js)
        check(api.selftest_event.wait(600), "自检已回传（冷启动：含建库 + 60 张缩略图生成）")
        cold = dict(api.selftest_result or {})
        check(bool(cold.get("ok")), "冷启动自检无异常", str(cold.get("error") or "")[:160])
        api.selftest_event.clear()
        api.selftest_result = {}
        win.evaluate_js("window.__ise_selftest({gallery:%s, topK:%d, timeoutMs:60000});"
                        % (json.dumps(gallery), top_k))
        check(api.selftest_event.wait(300), "自检已回传（热缓存：缩略图已落盘）")
        warm = dict(api.selftest_result or {})
        check(bool(warm.get("ok")), "热缓存自检无异常", str(warm.get("error") or "")[:160])
        print(f"    冷启动：命中 {cold.get('hits')} 条 / 前 10 张 {cold.get('firstVisibleMs')} ms"
              f" / 全部落定 {cold.get('firstPaintMs')} ms（成功 {cold.get('thumbsOk')}"
              f" 失败 {cold.get('thumbsFailed')}，检索 {cold.get('searchMs')} ms，"
              f"建库 {cold.get('build')} ms）")
        print(f"    热缓存：命中 {warm.get('hits')} 条 / 前 10 张 {warm.get('firstVisibleMs')} ms"
              f" / 全部落定 {warm.get('firstPaintMs')} ms（成功 {warm.get('thumbsOk')}"
              f" 失败 {warm.get('thumbsFailed')}，检索 {warm.get('searchMs')} ms）")
        if warm.get("failedSamples") or cold.get("failedSamples"):
            print(f"    失败样本：{warm.get('failedSamples') or cold.get('failedSamples')}")
        check(int(warm.get("hits") or 0) >= top_k, "热缓存命中数达到 top_k",
              f"{warm.get('hits')} >= {top_k}")
        check(int(warm.get("thumbsFailed") or 0) == 0, "缩略图无失败项",
              f"失败 {warm.get('thumbsFailed')}")
        # G5 判据：一屏缩略图首屏 < 1 s。这里区分两个量（都实测打印）：
        #   * 前 10 张可见：真正的“首屏”体验，两次都必须 < 1 s；
        #   * 全部 60 张落定：受浏览器 6 连接调度影响（服务端实测 2.2 ms/张），放宽到 1.5 s。
        best = min(int(cold.get("firstPaintMs") or 10 ** 9),
                   int(warm.get("firstPaintMs") or 10 ** 9))
        check(best < 1500, "G5 判据：一屏 60 张全部落定 < 1.5 s（两次取优）",
              f"冷 {cold.get('firstPaintMs')} ms / 热 {warm.get('firstPaintMs')} ms")
        check(int(cold.get("firstVisibleMs") or 10 ** 9) < 1000
              and int(warm.get("firstVisibleMs") or 10 ** 9) < 1000,
              "G5 判据：前 10 张缩略图两次均 < 1 s",
              f"冷 {cold.get('firstVisibleMs')} ms / 热 {warm.get('firstVisibleMs')} ms")
        bnd = cold.get("phaseBoundaries") or []      # 冷启动那次含建库
        check("save" in bnd and "done" in bnd,
              "前端可见 phase_boundary(save/done) 事件", "→".join(bnd[-4:]))
        check(not warm.get("error"), "自检未报错")

        # 服务端吞吐（同一 origin，串行 20 张，排除浏览器调度开销）：先预热（可能触发生成），再计时
        paths = ctx["service"].indexed_paths()[:20]
        if paths:
            urls = [f"{origin}/thumb/{k}?p={gui_web.b64u(p)}"
                    for k, p in zip(ctx["service"].thumb_keys(paths), paths)]
            wait_http_ready(origin + "/", timeout=5)
            for u in urls:                       # 预热：已存在的直接 200；未生成的顺带生成
                http_get(u, timeout=10, retries=1)
            time.sleep(0.8)
            t0 = time.time()
            ok_n = 0
            for u in urls:
                st, body, _ = http_get(u.split("?")[0], timeout=5, retries=1)
                ok_n += 1 if st == 200 and body[:2] == b"\xff\xd8" else 0
            dt = (time.time() - t0) * 1000
            per = dt / max(len(urls), 1)
            check(ok_n == len(urls) and per < 50,
                  "服务端缩略图吞吐（串行，热缓存）",
                  f"{per:.1f} ms/张（{ok_n}/{len(urls)} 张 200）")

        # ---- P2：去重页 + 大图对比（真实前端）-----------------------
        print("\n== G) P2 去重审查页（构建 / 首屏 / 二次打开）==")
        api.selftest_event.clear()
        api.selftest_result = {}
        win.evaluate_js("window.__ise_selftest_dedup({gallery:%s, threshold:2, "
                        "firstScreen:12, timeoutMs:120000});" % json.dumps(gallery))
        check(api.selftest_event.wait(600), "去重页自检已回传")
        dd = dict(api.selftest_result or {})
        check(bool(dd.get("ok")), "去重自检无异常", str(dd.get("error") or "")[:160])
        print(f"    扫描 {dd.get('scanned')} 张 / {dd.get('groups')} 组 "
              f"{dd.get('members')} 张；构建(scan+渲染) {dd.get('scanMs')} ms；"
              f"首屏 12 张缩略图 {dd.get('firstScreenMs')} ms（成功 {dd.get('thumbsOk')}）；"
              f"虚拟列表可见行 {dd.get('rows')}；二次打开 {dd.get('reopenMs')} ms")
        check(int(dd.get("groups") or 0) >= 1, "去重页有重复组",
              f"{dd.get('groups')} 组")
        check(int(dd.get("thumbsOk") or 0) > 0, "去重页缩略图可用",
              f"{dd.get('thumbsOk')} 张")
        check(int(dd.get("selCount") or 0) > 0, "“保留最佳”勾选生效",
              f"{dd.get('selCount')} 张")
        check(int(dd.get("reopenMs") or 10 ** 9) < 500, "二次打开 < 500 ms（报告在内存里）",
              f"{dd.get('reopenMs')} ms")

        print("\n== H) P2 大图对比（缩放/平移/拖放/全屏/Esc）==")
        api.selftest_event.clear()
        api.selftest_result = {}
        win.evaluate_js("window.__ise_selftest_compare({groupIndex:0});")
        check(api.selftest_event.wait(300), "对比窗自检已回传")
        cp = dict(api.selftest_result or {})
        check(bool(cp.get("ok")), "对比自检无异常", str(cp.get("error") or "")[:160])
        check(bool(cp.get("panesRendered")) and bool(cp.get("leftImg"))
              and bool(cp.get("rightImg")),
              "两侧原图均已渲染", f"成员 {cp.get('members')} 张 / "
              f"L status={cp.get('leftStatus')} complete={cp.get('leftComplete')} / "
              f"R status={cp.get('rightStatus')} complete={cp.get('rightComplete')} / "
              f"L={str(cp.get('leftSrc'))[-24:]}")
        check(bool(cp.get("zoomChanged")) and bool(cp.get("panChanged")),
              "滚轮缩放 / 拖动平移生效")
        check(bool(cp.get("stripDragOk")) and bool(cp.get("paneSwap")),
              "成员条拖放换区生效")
        check(bool(cp.get("escClosed")), "Esc 关闭对比窗")
        print(f"    全屏（Fullscreen API）: {'生效' if cp.get('fullscreen') else '本环境未授予（不影响）'}")

        print("\n== I) P2 去重页真实点击链路（DOM → Api → 服务层 → 事件 → 渲染）==")
        seen = []
        unsub = ctx["service"].subscribe(lambda ev: seen.append(ev))
        try:
            page_txt = win.evaluate_js(
                "(document.querySelector('.tabs button.active')||{}).textContent || ''")
            has_dedup = win.evaluate_js("!!document.querySelector('.dedup')")
            check(bool(has_dedup), "去重页在前台", f"当前页签={page_txt!r}")
            q = (".dedup")  # 状态读页面根的 data-*（虚拟滚动下 DOM 只有可见行）
            before = dict(groups=int(win.evaluate_js(f"document.querySelector('{q}').dataset.groups") or 0),
                          members=int(win.evaluate_js(f"document.querySelector('{q}').dataset.members") or 0))
            win.evaluate_js("window.__ise_confirm = () => true;")
            win.evaluate_js(
                "document.querySelector('.dedup [data-act=\"exact-unindexed\"]').click();")
            sel_n = 0
            t0 = time.time()
            while time.time() - t0 < 5:
                sel_n = int(win.evaluate_js(
                    "document.querySelector('.dedup').dataset.sel") or 0)
                if sel_n >= 1:
                    break
                time.sleep(0.2)
            check(sel_n >= 1, "点击“未入库的完全副本”后确有勾选",
                  f"勾选 {sel_n} 张（可见行 {before['members']} 成员 / {before['groups']} 组）")
            win.evaluate_js("document.querySelector('.dedup [data-act=\"delete\"]').click();")
            ev = None
            t0 = time.time()
            while time.time() - t0 < 300:
                ev = next((e for e in seen if e.get("event") == "task_done"
                           and e.get("op") == "dedup_apply"), None)
                if ev:
                    break
                time.sleep(0.1)
            removed = ((ev or {}).get("result") or {}).get("removed") or []
            check(bool(ev) and len(removed) >= 1,
                  "删除按钮触发 dedup_apply 并使服务层发出 task_done",
                  f"removed={len(removed)}")
            check(bool(removed) and not any(os.path.exists(p) for p in removed),
                  "被删文件确实移出图库", str([os.path.basename(p) for p in removed]))
            t0 = time.time()
            after_members = before["members"]
            while time.time() - t0 < 5:
                after_members = int(win.evaluate_js(
                    "document.querySelector('.dedup').dataset.members") or 0)
                if after_members < before["members"]:
                    break
                time.sleep(0.2)
            check(after_members < before["members"],
                  "报告在内存里就地摘掉被删项（无需重扫）",
                  f"成员 {before['members']} → {after_members}")
        finally:
            unsub()
    except Exception as e:                               # noqa: BLE001
        import traceback
        check(False, "窗口探针异常", f"{e!r}")
        traceback.print_exc()
    finally:
        try:
            ctx["bridge"].stop()
            win.destroy()
        except Exception:                                # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Web 界面回归：无头（WSGI/Api/事件桥）+ Gate 1/3/4/5（真实窗口）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--db", type=int, default=120, help="模拟图库图片总数")
    ap.add_argument("--per-group", type=int, default=4, help="每组变体数")
    ap.add_argument("--top-k", type=int, default=60, help="Gate 5 一屏缩略图数")
    ap.add_argument("--no-window", action="store_true", help="只跑无头部分（不需要桌面会话）")
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    a = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dist = gui_web.resolve_dist()
    if not os.path.isfile(os.path.join(dist, "index.html")):
        print(f"✗ 未找到前端产物 {dist}：请先 `cd frontend && pnpm install && pnpm build`")
        return 2

    work = tempfile.mkdtemp(prefix="webverify_")
    data = os.path.join(work, "test_data")
    gallery = os.path.join(data, "db")
    t0 = time.time()
    print(f"临时目录 {work}\n前端产物 {dist}")
    try:
        print("\n== 0) 生成模拟图库（make_test_dataset.py）==")
        rc = subprocess.run(
            [sys.executable, "-E", os.path.join(repo, "make_test_dataset.py"),
             "--out", data, "--db", str(a.db),
             "--per-group", str(a.per_group), "--queries", "10"],
            cwd=repo).returncode
        imgs = glob.glob(os.path.join(gallery, "*.jpg"))
        check(rc == 0 and len(imgs) > 0, "模拟图库就绪", f"{len(imgs)} 张")

        _rc, svc, rec, fake, bridge, srv, base = headless(gallery, dist)
        try:
            st, body, _ = http_get(base + "/health")
            health = json.loads(body) if st == 200 else None
            check(bool(health and health["ok"]),
                  "只读 HTTP 服务可独立运行（无窗口）")
            api_checks(gallery, svc, rec, fake, bridge, base)
            dedup_checks(gallery, svc, rec, fake, bridge, base)
        finally:
            srv.shutdown()
            svc.close()

        if not a.no_window:
            print("\n== C-F) 真实窗口 Gate 探针 ==")
            gui_web.launch(gallery=gallery, block=True,
                           on_start=lambda ctx: probe_gates(ctx, gallery, a.top_k))
        else:
            print("\n（--no-window：跳过 Gate 1/3/4/5 窗口探针）")
    finally:
        if a.keep:
            print(f"（--keep：保留 {work}）")
        else:
            shutil.rmtree(work, ignore_errors=True)

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    if FAILED:
        print(f"结果: 存在失败项 {len(FAILED)} 项 -> {FAILED}")
        return 1
    print("结果: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# @@TAIL@@
