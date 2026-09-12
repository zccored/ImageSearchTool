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
image-search 工具包打包配置（onedir，双入口，CPU+CUDA 全量）。

产物（--distpath 指定输出根，默认 <项目>/dist）：
  <dist>/ImageSearch/ImageSearchGUI.exe    图形界面（以图搜图管理器）
  <dist>/ImageSearch/ImageSearchCLI.exe    命令行（build/add/search/ingest …）
  其余依赖与数据在 ImageSearch/_internal/ 下，分发时整个 ImageSearch 目录
  打包（zip）即为“独立工具包”，目标机无需安装 Python/依赖。

特性：
  * torch/torchvision 在业务代码中均为函数内延迟导入 —— 静态分析不可见，
    必须走 hiddenimports 显式收集（含 CUDA DLL，全量随包）。
  * GUI 与 CLI 各自独立 Analysis（共享同一份 binaries/datas 输出，避免
    依赖重复占盘），两入口使用各自 PYZ。
  * resnet18 预训练权重离线可用：datas 打包进 torch_home/hub/checkpoints/，
    runtime hook 设 TORCH_HOME 指向包内（无网络也能建精排索引；
    resnet50 等未内置权重仍会按需联网下载）。
  * peer_manifest.json 随包分发：切换启动按钮在目标机因绝对路径下不存在
    全栈图库管理器 main.py 而自动禁用（独立工具包语义），本机则可用。

构建：  pyinstaller --noconfirm --clean image-search.spec
            --distpath F:\PLC\dist --workpath F:\PLC\pyi_build
"""
import os

SRC = r"D:\code\新的代码\全栈图库管理器 v3.2bata\image-search"
WEIGHTS_DIR = os.path.join(os.environ.get("USERPROFILE", ""),
                           ".cache", "torch", "hub", "checkpoints")

hiddenimports = [
    "torch", "torchvision", "torchvision.models",
    "cv2", "PIL", "numpy", "psutil",
]

datas = [
    (os.path.join(SRC, "peer_manifest.json"), "."),
]
if os.path.isdir(WEIGHTS_DIR):
    datas.append((WEIGHTS_DIR, "torch_home/hub/checkpoints"))

runtime_hooks = [
    os.path.join(SRC, "pyinstall_runtime_dynamo_stub.py"),   # 须在 import torch 前
    os.path.join(SRC, "pyinstall_runtime_torch_home.py"),
]

a_gui = Analysis(
    [os.path.join(SRC, "gui.py")],
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

coll = COLLECT(
    exe_gui, exe_cli,
    a_gui.binaries,
    a_gui.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ImageSearch",
)
