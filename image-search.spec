# -*- mode: python ; coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PyInstaller 打包配置：onedir 双入口、CPU+CUDA 全量依赖
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
image-search 工具包打包配置（onedir，三入口，CPU+CUDA 全量）。

产物（--distpath 指定输出根，默认 <项目>/dist）：
  <dist>/ImageSearch/ImageSearchGUI.exe    图形界面（tkinter 版，以图搜图管理器）
  <dist>/ImageSearch/ImageSearchWeb.exe    新界面（Web 版：pywebview + Vue3 + 只读 WSGI，见 gui_web.py）
  <dist>/ImageSearch/ImageSearchCLI.exe    命令行（build/add/search/ingest …）
  其余依赖与数据在 ImageSearch/_internal/ 下，分发时整个 ImageSearch 目录
  打包（zip）即为“独立工具包”，目标机无需安装 Python/依赖。

特性：
  * torch/torchvision 在业务代码中均为函数内延迟导入 —— 静态分析不可见，
    必须走 hiddenimports 显式收集（含 CUDA DLL，全量随包）。
  * 三个入口各自独立 Analysis（共享同一份 binaries/datas 输出，避免依赖重复占盘），
    使用各自 PYZ。
  * resnet18 预训练权重离线可用：datas 打包进 torch_home/hub/checkpoints/，
    runtime hook 设 TORCH_HOME 指向包内（无网络也能建精排索引；
    resnet50 等未内置权重仍会按需联网下载）。
  * peer_manifest.json 随包分发：切换启动按钮在目标机因绝对路径下不存在
    全栈图库管理器 main.py 而自动禁用（独立工具包语义），本机则可用。
  * Web 入口额外带：frontend/dist（Vue3 产物）+ ui_strings.json / ui_theme.css
    （存在才带）；pywebview 自带的 js 资源与 WebView2 interop DLL 由它自己的
    PyInstaller hook 收集，无需手工 --add-data；未用的 GUI 后端必须 excludes
    （否则本机装着的 PySide6 会被整包收进去）。

构建：  pyinstaller --noconfirm --clean image-search.spec
            --distpath <项目>/dist --workpath <项目>/pyi_build
        （SRC 由 spec 所在目录推导；另有环境变量 ISE_SRC 可覆盖源码根）
"""
import os

SRC = os.environ.get("ISE_SRC") or os.path.abspath(SPECPATH)
# 注：PyInstaller 注入的 SPECPATH 就是 **spec 所在目录**（见其 build_main.py:
# `CONF['specpath'], CONF['specnm'] = os.path.split(CONF['spec'])`），
# 即本仓库根 —— 不要再套一层 os.path.dirname，否则 SRC 会指到仓库的上一层。
WEIGHTS_DIR = os.path.join(os.environ.get("USERPROFILE", ""),
                           ".cache", "torch", "hub", "checkpoints")

# OpenVINO 可选后端：运行时 DLL + plugins.xml；找不到时 spec 仍可打 torch-only 包
try:
    from PyInstaller.utils.hooks import collect_all as _collect_all
    _ov_datas, _ov_binaries, _ov_hidden = _collect_all("openvino")
    print("[spec] OpenVINO: datas=%d binaries=%d hidden=%d" % (
        len(_ov_datas), len(_ov_binaries), len(_ov_hidden)))
except Exception as _e:
    print("[spec] OpenVINO 未收集：%r" % (_e,))
    _ov_datas, _ov_binaries, _ov_hidden = [], [], []

hiddenimports = [
    "torch", "torchvision", "torchvision.models",
    "cv2", "PIL", "numpy", "psutil",
    # 运行期按需 import 的本地模块（PyInstaller 静态分析可能漏收）
    "hybrid_search.distribution",
    "hybrid_search.federation",
    "hybrid_search.openvino_backend",
    "hybrid_search.feature_checkpoint",
    "hybrid_search.prep_cache_torch",
]

datas = [
    (os.path.join(SRC, "peer_manifest.json"), "."),
]
_model_dir = os.path.join(SRC, "model")
if os.path.isfile(os.path.join(_model_dir, "resnet18.xml")):
    datas.append((_model_dir, "model"))
datas += _ov_datas
hiddenimports += _ov_hidden

# ---- 原生 PNG 加速模块与 VC/OpenMP 运行库 --------------------------------
# PyInstaller 不会自动收 hybrid_search/native 下的运行期 pyd/dll；
# torch_cpu.dll 还依赖 System32 的 libomp140.x86_64.dll（干净 Windows 上不一定有）。
NATIVE_DIR = os.path.join(SRC, "hybrid_search", "native")
_SYS32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
extra_binaries = []
def _add_binary(src, dest):
    if os.path.isfile(src):
        extra_binaries.append((src, dest))
    else:
        print("[spec] 警告：缺少打包依赖 %s" % src)
_add_binary(os.path.join(NATIVE_DIR, "_pngfast.cp312-win_amd64.pyd"),
            "hybrid_search/native")
_add_binary(os.path.join(NATIVE_DIR, "libdeflate.dll"),
            "hybrid_search/native")
for _name in ("libomp140.x86_64.dll",):
    _add_binary(os.path.join(_SYS32, _name), "torch/lib")
    _add_binary(os.path.join(_SYS32, _name), ".")
for _name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
              "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
              "concrt140.dll", "vcomp140.dll"):
    _add_binary(os.path.join(_SYS32, _name), ".")
print("[spec] 额外运行库/原生模块：%d 个" % len(extra_binaries))
if os.path.isdir(WEIGHTS_DIR):
    datas.append((WEIGHTS_DIR, "torch_home/hub/checkpoints"))

# ---- Web 入口（ImageSearchWeb）专用的数据与依赖 --------------------------
# 前端产物必须存在：缺失时给出提示，避免打出一个“没有界面”的包
WEB_DIST = os.path.join(SRC, "frontend", "dist")
web_datas = list(datas)
if os.path.isfile(os.path.join(WEB_DIST, "index.html")):
    web_datas.append((WEB_DIST, "frontend/dist"))
    for _name in ("ui_strings.json", "ui_theme.css"):     # 外置文案/主题（存在才带；放包根即可覆盖）
        _p = os.path.join(SRC, _name)
        if os.path.isfile(_p):
            web_datas.append((_p, "."))
    _lic = os.path.join(WEB_DIST, "Vue-LICENSE.txt")      # 前端第三方许可（build 时生成）
    print("[spec] Web 前端产物: %s%s" % (
        WEB_DIST, "" if os.path.isfile(_lic) else "（缺 Vue-LICENSE.txt）"))
else:
    print("[spec] 警告：未找到 %s —— ImageSearchWeb.exe 将无界面可用；"
          "请先 cd frontend && pnpm install && pnpm build" % WEB_DIST)

# pywebview 的后端与 .NET 桥（仅 win32 需要；显式列出，避免静态分析漏收）
hiddenimports_web = hiddenimports + [
    "webview", "webview.platforms.winforms", "webview.platforms.edgechromium",
    "clr", "pythonnet",
]
# 未使用的 GUI 后端与大件必须排除（PyInstaller 会把机器上装着的 Qt 一并收走）
excludes_web = ["PyQt5", "PyQt6", "PySide2", "PySide6", "gi", "tkinter",
                "matplotlib", "pandas", "scipy"]

runtime_hooks = [
    os.path.join(SRC, "pyinstall_runtime_dynamo_stub.py"),   # 须在 import torch 前
    os.path.join(SRC, "pyinstall_runtime_torch_home.py"),
]

a_gui = Analysis(
    [os.path.join(SRC, "gui.py")],
    pathex=[SRC],
    binaries=extra_binaries + _ov_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz_gui = PYZ(a_gui.pure)

a_cli = Analysis(
    [os.path.join(SRC, "main.py")],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz_cli = PYZ(a_cli.pure)

# ---- Web 入口：第三个 Analysis（共享同一份 binaries/datas，不重复占盘）----
a_web = Analysis(
    [os.path.join(SRC, "gui_web.py")],
    pathex=[SRC],
    binaries=[],
    datas=web_datas,
    hiddenimports=hiddenimports_web,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=excludes_web,
    noarchive=False,
    optimize=0,
)
pyz_web = PYZ(a_web.pure)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="ImageSearchGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                  # GUI：无控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="ImageSearchCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,                   # CLI：保留控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

exe_web = EXE(
    pyz_web,
    a_web.scripts,
    [],
    exclude_binaries=True,
    name="ImageSearchWeb",          # 新界面（Web 版）：独立 exe（维护者 2026-09-27 定）
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                  # 窗口模式：stdout/stderr 为 None（包内代码不得 print）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_gui, exe_cli, exe_web,
    a_gui.binaries,
    a_gui.datas,
    a_web.binaries,
    a_web.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ImageSearch",
)
