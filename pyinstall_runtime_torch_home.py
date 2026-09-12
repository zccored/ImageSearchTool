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

"""PyInstaller runtime hook：把 TORCH_HOME 指向包内自带权重目录，
使 torchvision 预训练权重离线可用（打包时 datas 已把 resnet18
权重放进 <包>/torch_home/hub/checkpoints/）。setdefault：外部若
显式设置了 TORCH_HOME 则尊重外部值。"""
import os
import sys

base = getattr(sys, "_MEIPASS", None) or os.path.dirname(
    os.path.abspath(sys.executable))
os.environ.setdefault("TORCH_HOME", os.path.join(base, "torch_home"))
