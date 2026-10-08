# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — “切换启动”对端：跨程序互切 + main.py 哈希白名单校验
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
peer_launcher —— “切换启动”对端：跨程序互切（配合端口画板按钮闭环）。

本图库检索管理器（image-search）可与“全栈图库管理器 / 端口画板”互相切换：
  * 端口画板侧按钮：关自己 → 打开本检索管理器（见 handoff 流程）；
  * 本侧“切换启动”按钮：关本程序 → 打开对端（本模块实现）。

**目标优先级（2026-10-06 起，用户明确指定）**：

  ============  ==================================================  ==============
  环境里有什么   切换启动会开什么                                      要不要校验
  ============  ==================================================  ==============
  main.py       全栈图库管理器主程序（**优先且只用它**）              **要**（sha256 白名单）
  只有端口画板    端口画板（打包 exe 优先，其次源码入口）              **不要**
  都没有         按钮不予激活                                        —
  ============  ==================================================  ==============

  注意“有 main.py 就**只**用 main.py”—— 两者同时存在时不会退而求其次去开端口画板：
  主程序才是正主，端口画板只是它的一个窗口。

main.py 的激活门槛（防呆/防注入）——不满足任一条即“不予激活”按钮：
  1) 绝对路径下必须真实存在 main.py（本检索管理器被当作独立数据包分发到
     其它位置/机器时，路径上不存在该文件 → 按钮保持禁用）；
  2) 文件内容哈希必须与登记白名单一致（peer_manifest.json，随本包分发）。
     内容被改动/被植入后门 → 哈希矛盾 → 禁用，并给出新/旧哈希供人工判断；
  3) 登记动作只应在用户人工确认文件可信后发生（manifest 记录登记时间留痕）。

端口画板**不走**上面这套：它是可独立分发的工具，不带白名单，也不需要“信任登记”。

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

# 全栈图库管理器绝对位置：默认取**本仓库的上级目录**（两者的常规布局就是
# <...>\全栈图库管理器 v3.2bata\main.py 与 <...>\全栈图库管理器 v3.2bata\image-search\）。
# 需要换机/换布局时用环境变量 IMG_PEER_DIR 覆盖 —— 不要把本机绝对路径写回源码，
# 本文件随公开仓库分发。
PEER_DIR = os.environ.get("IMG_PEER_DIR") or os.path.dirname(BASE)
PEER_MAIN = os.path.join(PEER_DIR, "main.py")
PEER_NAME = "main.py"

# ---------------------------------------------------------------------------
# 候选 2：**端口画板**（原「平台整合器」，2026-10-06 起正式更名）
#
# 为什么要有第二个候选：端口画板现在会**独立打包**成 exe 分发给其它用户，
# 那个包里通常**没有**全栈图库管理器的 main.py。此时「切换启动」应该能直接开端口画板。
# 优先级由用户明确指定：
#   ① 有 main.py            → **优先且只**用 main.py（照旧走 sha256 白名单校验）
#   ② 没有 main.py，有端口画板 → 用端口画板（**不做哈希校验**：它是可独立分发的工具，
#                              不带白名单，也不需要"信任登记"这一层）
#   ③ 都没有                → 按钮不予激活
# ---------------------------------------------------------------------------
PANEL_EXE_NAMES = ("端口画板.exe", "Download_To_Draw.exe", "PortPanel.exe", "port_panel.exe")
PANEL_SRC_NAMES = ("端口画板.py", "port_panel.py")

# 目标种类
TGT_MAIN = "main"
TGT_PANEL = "panel"
TGT_LABEL = {TGT_MAIN: "全栈图库管理器", TGT_PANEL: "端口画板"}

# 登记白名单：随本包分发的受信哈希记录
MANIFEST_PATH = os.path.join(BASE, "peer_manifest.json")

# 状态码（供 GUI 展示/提示）
ST_OK = "ok"
ST_MISSING = "missing"          # 绝对路径下不存在
ST_UNREGISTERED = "unregistered"  # 存在但白名单无记录（首次见到，需人工信任）
ST_MISMATCH = "mismatch"        # 内容哈希与登记不一致（可能被改动/植入）


def _find_panel():
    """在 PEER_DIR 里找端口画板：打包 exe 优先，其次源码入口。返回 (路径, 'exe'|'py')。"""
    for name in PANEL_EXE_NAMES:
        p = os.path.join(PEER_DIR, name)
        if os.path.isfile(p):
            return p, "exe"
    for name in PANEL_SRC_NAMES:
        p = os.path.join(PEER_DIR, name)
        if os.path.isfile(p):
            return p, "py"
    return None, None


def resolve_target():
    """决定这次「切换启动」开哪个。返回 (kind, path, how)：
    kind ∈ {'main','panel',None}；how ∈ {'py','exe',None}。

    **有 main.py 就只用 main.py**（哪怕端口画板也在）—— 这是用户明确的口径：
    两者同时存在时，主程序才是"正主"，端口画板只是它的一个窗口。
    """
    if os.path.isfile(PEER_MAIN):
        return TGT_MAIN, PEER_MAIN, "py"
    panel, how = _find_panel()
    if panel:
        return TGT_PANEL, panel, how
    return None, None, None


def target_label(kind=None):
    """给界面用的人类可读目标名。"""
    if kind is None:
        kind = resolve_target()[0]
    return TGT_LABEL.get(kind) or "（未找到切换目标）"


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

    目标按 resolve_target() 的优先级决定：有 main.py 就**只**用 main.py（走哈希白名单），
    否则用端口画板（**不校验**）。

    返回 {"ok": bool, "code": 状态码, "reason": str,
          "target": 'main'|'panel'|None, "target_path": str|None, "target_how": 'py'|'exe'|None,
          "current_sha256": str|None, "registered_sha256": str|None,
          "registered_at": str|None}
    """
    kind, path, how = resolve_target()
    out = {"ok": False, "code": ST_MISSING, "reason": "",
           "target": kind, "target_path": path, "target_how": how,
           "current_sha256": None, "registered_sha256": None,
           "registered_at": None}

    # ---- 端口画板：**不需要**校验，存在即可激活 ----
    if kind == TGT_PANEL:
        out["ok"] = True
        out["code"] = ST_OK
        out["reason"] = (f"未检测到全栈图库管理器 main.py，改用独立分发的「端口画板」：\n"
                         f"{path}\n"
                         f"（端口画板是可独立运行的工具，不做哈希校验）")
        return out

    if kind is None:
        out["reason"] = (f"未检测到任何可切换的目标。\n"
                         f"查找位置：{PEER_DIR}\n"
                         f"  · 全栈图库管理器：{PEER_MAIN}\n"
                         f"  · 端口画板：{' / '.join(PANEL_EXE_NAMES + PANEL_SRC_NAMES)}\n"
                         f"（本程序作为独立数据包分发时不含这些文件，按钮不予激活）")
        return out

    # ---- main.py：照旧走 sha256 白名单 ----
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
    """人工确认信任后，把当前文件哈希写入登记白名单（留痕）。

    只对 **main.py** 有意义 —— 端口画板不参与哈希校验，无需登记。
    """
    kind, path, _how = resolve_target()
    if kind == TGT_PANEL:
        return {"ok": True, "sha256": None,
                "reason": f"当前切换目标是「端口画板」，不做哈希校验，无需登记：\n{path}"}
    if not os.path.isfile(PEER_MAIN):
        return {"ok": False, "reason": f"切换目标不存在：{PEER_MAIN}"}
    cur = _sha256_file(PEER_MAIN)
    manifest = load_manifest()
    # 注意：**不写 path 字段**。清单随本包分发（公开仓库），写进去等于把本机绝对路径
    # 带回去；而 check() 只读 sha256 / registered_at，path 从未被消费。
    (manifest.setdefault("files", {}))[PEER_NAME] = {
        "sha256": cur,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_manifest(manifest)
    return {"ok": True, "sha256": cur,
            "reason": f"已登记信任 {PEER_MAIN}"}


def launch() -> tuple:
    """
    以独立进程启动对端（不随本程序退出而终止）。
    返回 (proc, None) 或 (None, 错误信息)。

    开哪个由 resolve_target() 决定：**有 main.py 就只用 main.py**，否则用端口画板
    （打包 exe 直接起 exe；源码入口走 python）。两种情形都**不做**存活探测之外的处理。

    启动方式说明（实测结论，重要；只对 main.py 那条链路成立）：
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
      * 端口画板那条链路没有这些问题：打包 exe 自带 GUI 子系统，
        源码入口也不 spawn 控制台子进程，所以 exe 用 SW_SHOWNORMAL 正常显示。
    """
    kind, path, how = resolve_target()
    if kind is None:
        return None, (f"未检测到任何可切换的目标。\n查找位置：{PEER_DIR}\n"
                      f"  · 全栈图库管理器：{PEER_MAIN}\n"
                      f"  · 端口画板：{' / '.join(PANEL_EXE_NAMES + PANEL_SRC_NAMES)}")
    if not os.path.isfile(path):
        return None, f"切换目标不存在：{path}"


    # ---- 端口画板（打包 exe）：直接起 exe，不需要 Python ----
    if kind == TGT_PANEL and how == "exe":
        err_log = os.path.join(os.environ.get("TEMP") or ".", "peer_launch_err.log")
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 1                 # SW_SHOWNORMAL：它自带 GUI，正常显示
            proc = subprocess.Popen(
                [path],
                cwd=os.path.dirname(path) or PEER_DIR,
                startupinfo=si,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=open(err_log, "w", encoding="utf-8", errors="replace"),
            )
            return proc, None
        except OSError as e:
            return None, f"启动失败：{e}"

    # ---- main.py / 端口画板源码：走 python.exe ----
    if kind == TGT_PANEL:
        target_script = path
    else:
        target_script = PEER_MAIN

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
            [exe, "-X", "utf8", target_script],
            cwd=os.path.dirname(target_script) or PEER_DIR,
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
        print(f"目标：{target_label(st.get('target'))}")
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
    print(f"已启动：{resolve_target()[1]}（pid={proc.pid}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
