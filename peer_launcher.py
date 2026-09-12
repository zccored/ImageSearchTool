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
peer_launcher —— “切换启动”全栈图库管理器（跨程序互切，配合 img_server 按钮闭环）。

本图库检索管理器（image-search）可与“全栈图库管理器”（D:\\code\\新的代码\\全栈图库
管理器 v3.2bata\\main.py，PySide6 独立程序）互相切换：
  * img_server 侧按钮：关全栈管理器 → 打开本检索管理器（见 handoff 流程）；
  * 本侧“切换启动”按钮：关本程序 → 打开全栈管理器 main.py（本模块实现）。

激活门槛（防呆/防注入）——不满足任一条即“不予激活”按钮：
  1) 绝对路径下必须真实存在 main.py（本检索管理器被当作独立数据包分发到
     其它位置/机器时，路径上不存在该文件 → 按钮保持禁用）；
  2) 文件内容哈希必须与登记白名单一致（peer_manifest.json，随本包分发）。
     内容被改动/被植入后门 → 哈希矛盾 → 禁用，并给出新/旧哈希供人工判断；
  3) 登记动作只应在用户人工确认文件可信后发生（manifest 记录登记时间留痕）。

可靠性：目标程序由独立进程启动（DETACHED，不随本程序退出而终止）；
启动后做短暂存活探测，2 秒内即退出视为启动失败并回滚（不关闭本程序）。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

# 全栈图库管理器绝对位置（可用环境变量 IMG_PEER_DIR 覆盖——换机/迁移时用）
PEER_DIR = os.environ.get("IMG_PEER_DIR") or \
    r"D:\code\新的代码\全栈图库管理器 v3.2bata"
PEER_MAIN = os.path.join(PEER_DIR, "main.py")
PEER_NAME = "main.py"

# 登记白名单：随本包分发的受信哈希记录
MANIFEST_PATH = os.path.join(BASE, "peer_manifest.json")

# 状态码（供 GUI 展示/提示）
ST_OK = "ok"
ST_MISSING = "missing"          # 绝对路径下不存在
ST_UNREGISTERED = "unregistered"  # 存在但白名单无记录（首次见到，需人工信任）
ST_MISMATCH = "mismatch"        # 内容哈希与登记不一致（可能被改动/植入）


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if os.path.isfile(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"files": {}}
    return {"files": {}}


def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MANIFEST_PATH)


def check() -> dict:
    """
    校验“切换目标”是否可激活。
    返回 {"ok": bool, "code": 状态码, "reason": str,
          "current_sha256": str|None, "registered_sha256": str|None,
          "registered_at": str|None}
    """
    out = {"ok": False, "code": ST_MISSING, "reason": "",
           "current_sha256": None, "registered_sha256": None,
           "registered_at": None}
    if not os.path.isfile(PEER_MAIN):
        out["reason"] = (f"未检测到绝对路径下的切换目标：\n{PEER_MAIN}\n"
                         f"（本程序作为独立数据包分发时不含该文件，按钮不予激活）")
        return out
    cur = _sha256_file(PEER_MAIN)
    out["current_sha256"] = cur
    manifest = load_manifest()
    rec = (manifest.get("files") or {}).get(PEER_NAME)
    if not rec:
        out["code"] = ST_UNREGISTERED
        out["reason"] = (f"切换目标存在但尚未登记信任：\n{PEER_MAIN}\n"
                         f"当前 SHA256: {cur[:16]}…\n"
                         "请人工确认文件来源可信后点击“信任并登记”")
        return out
    out["registered_sha256"] = rec.get("sha256")
    out["registered_at"] = rec.get("registered_at")
    if rec.get("sha256", "").lower() != cur.lower():
        out["code"] = ST_MISMATCH
        out["reason"] = (f"切换目标内容与登记不一致（可能被修改/植入后门）：\n"
                         f"{PEER_MAIN}\n"
                         f"当前 SHA256: {cur[:16]}…\n"
                         f"登记 SHA256: {rec.get('sha256', '')[:16]}…\n"
                         f"登记时间  : {rec.get('registered_at', '?')}\n"
                         "按钮不予激活；如文件是官方更新，请人工核对后重新登记。")
        return out
    out["ok"] = True
    out["code"] = ST_OK
    out["reason"] = f"校验通过，可切换启动：\n{PEER_MAIN}"
    return out


def register() -> dict:
    """人工确认信任后，把当前文件哈希写入登记白名单（留痕）。"""
    if not os.path.isfile(PEER_MAIN):
        return {"ok": False, "reason": f"切换目标不存在：{PEER_MAIN}"}
    cur = _sha256_file(PEER_MAIN)
    manifest = load_manifest()
    (manifest.setdefault("files", {}))[PEER_NAME] = {
        "sha256": cur,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": PEER_MAIN,
    }
    save_manifest(manifest)
    return {"ok": True, "sha256": cur,
            "reason": f"已登记信任 {PEER_MAIN}"}


def launch() -> tuple:
    """
    以独立进程启动全栈图库管理器 main.py（不随本程序退出而终止）。
    返回 (proc, None) 或 (None, 错误信息)。

    启动方式说明（实测结论，重要）：
      * **不能用 pythonw（无控制台）**——全栈管理器每秒刷新 GPU 性能、
        内部反复 spawn nvidia-smi 等控制台子进程；父进程无控制台时，
        Windows 会为这些子进程不断新建/激活控制台窗口，
        表现为“窗口一直闪弹并抢占前台”（pythonw 实测每 ~1 秒一次）。
      * **不能用 SW_HIDE 完全隐藏控制台**——实测会导致其 Qt 主窗口
        无法显示（卡在启动画面，日志停滞）。
      * 正确方式：**python.exe + CREATE_NEW_CONSOLE + SW_SHOWMINIMIZED**
        （独立且最小化的控制台）——子进程附着该控制台后不再产生任何
        窗口活动，主窗口正常显示，稳态前台零闪烁（均已实测验证）。
      * 补充：main.py 启动后会出现**启动确认弹窗**（带按钮，需人工点击
        后才进入主程序）。GUI 切换按钮的 2 秒存活探测只判断进程未崩溃，
        不等待该人工确认——弹窗出现属预期，点击确认即进入主程序。
    """
    if not os.path.isfile(PEER_MAIN):
        return None, f"切换目标不存在：{PEER_MAIN}"
    # 可执行文件必须是 python.exe（带控制台子系统）：
    #   * 源码运行：即使本进程由 pythonw 启动（无控制台），
    #     也换用同目录的 python.exe；
    #   * PyInstaller 打包运行：sys.executable 是本工具自身 exe，
    #     不能当解释器，退回 PATH 里的 python（或 IMG_PEER_PYTHON 指定）。
    if getattr(sys, "frozen", False):
        exe = os.environ.get("IMG_PEER_PYTHON") or shutil.which("python")
        if not exe:
            return None, "打包版切换启动需要本机安装 Python 并加入 PATH，" \
                         "或用环境变量 IMG_PEER_PYTHON 指定 python.exe 路径"
    else:
        exe = sys.executable
        if os.path.basename(exe).lower().startswith("pythonw"):
            alt = os.path.join(os.path.dirname(exe), "python.exe")
            if os.path.isfile(alt):
                exe = alt
            else:
                return None, f"找不到 python.exe（当前解释器：{exe}）"
    err_log = os.path.join(os.environ.get("TEMP") or ".", "peer_launch_err.log")
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 6                     # SW_SHOWMINIMIZED：最小化控制台
        proc = subprocess.Popen(
            [exe, "-X", "utf8", PEER_MAIN],
            cwd=PEER_DIR,
            startupinfo=si,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=open(err_log, "w", encoding="utf-8", errors="replace"),
        )
        return proc, None
    except OSError as e:
        return None, f"启动失败：{e}"


def main() -> int:
    """命令行入口：python peer_launcher.py check | register | launch"""
    import argparse
    ap = argparse.ArgumentParser(description="切换启动校验/登记/启动")
    ap.add_argument("action", choices=["check", "register", "launch"])
    a = ap.parse_args()
    if a.action == "check":
        st = check()
        print(f"[{st['code']}] {'OK' if st['ok'] else '不可激活'}：{st['reason']}")
        return 0 if st["ok"] else 1
    if a.action == "register":
        st = register()
        print(st["reason"])
        return 0 if st["ok"] else 1
    proc, err = launch()
    if err:
        print(err)
        return 1
    print(f"已启动：{PEER_MAIN}（pid={proc.pid}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
