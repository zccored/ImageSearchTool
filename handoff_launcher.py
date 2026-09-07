# -*- coding: utf-8 -*-
"""
handoff_launcher —— 由 img_server 按钮回调“一次性启动”的过渡进程。

职责（内存敏感顺序：img_server 先退、再开检索管理器）：
  1) 读 request JSON（见 docs/HANDOFF_PROTOCOL.md）
  2) 等待 request.expect_exit 中的进程（pid 或进程名）全部退出
     （img_server 及“上级主程序”），超时则放弃并写错误 result
  3) 把 request 文件转 working_（防重入）
  4) open_mode=gui → 启动 image-search/gui.py --auto-handoff <working>
     （GUI 内自动执行增量建库并写 result，全程可视）
     open_mode=cli  → 本进程直接执行增量建库并写 result（无窗口）

用法（img_server 按钮回调里调用）：
  python handoff_launcher.py <request.json> [--wait 90] [--cwd ...]
本脚本依赖 image-search 包；默认按自身所在目录向上定位。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))


def wait_processes_exit(req: dict, timeout: float) -> bool:
    """等待 expect_exit.pids / .names 全部消失。返回是否全部退出。"""
    import psutil

    expect = req.get("expect_exit") or {}
    pids = set(int(p) for p in (expect.get("pids") or []))
    names = set((expect.get("names") or []))
    if not pids and not names:
        return True                    # 未声明则假定已退出
    t0 = time.time()
    while time.time() - t0 < timeout:
        alive = False
        if pids:
            for pid in list(pids):
                try:
                    p = psutil.Process(pid)
                    if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                        alive = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pids.discard(pid)
            if alive:
                time.sleep(0.5)
                continue
        if names:
            # 按可执行文件名匹配（不匹配 python.exe，避免误伤本进程）
            cand = {p.info["name"].lower()
                    for p in psutil.process_iter(["name"])}
            if names & cand:
                alive = True
        if not alive:
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="img_server 交接过渡启动器")
    ap.add_argument("request", help="request JSON 路径")
    ap.add_argument("--wait", type=float, default=90.0,
                    help="等待 img_server 进程退出的最大秒数")
    a = ap.parse_args()
    sys.path.insert(0, BASE)                     # image-search 包目录

    from hybrid_search import handoff as H

    try:
        req = H.load_request(os.path.abspath(a.request))
    except Exception as e:                       # noqa: BLE001
        print(f"[launcher] request 无效: {e}", flush=True)
        return 2

    req_id = req.get("request_id", "unknown")
    print(f"[launcher] 等待 img_server 退出（≤{a.wait:.0f}s）…", flush=True)
    if not wait_processes_exit(req, a.wait):
        H.write_result(req_id, {
            "request_id": req_id, "ok": False,
            "fatal_error": "等待 img_server 退出超时，已放弃自动流程（未强制结束）",
        })
        print("[launcher] 超时放弃", flush=True)
        return 1

    mode = req.get("open_mode", "gui")
    if mode == "cli":
        # 直接在本进程执行（结果写回 handoff 目录）
        req_path = os.path.abspath(a.request)
        result = H.process_request_file(req_path)
        print(f"[launcher] ingest ok={result.get('ok')} "
              f"新增 {result.get('total_added', 0)} 张", flush=True)
        return 0 if result.get("ok") else 1

    # gui 模式：转 working 后交给 gui.py --auto-handoff
    d = os.path.dirname(os.path.abspath(a.request))
    working = H.mark_working(os.path.abspath(a.request), d)
    gui = os.path.join(BASE, "gui.py")
    print(f"[launcher] 启动图库检索管理器（自动增量）：{gui}", flush=True)
    subprocess.Popen([sys.executable, "-X", "utf8", gui,
                      "--auto-handoff", working],
                     cwd=BASE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
