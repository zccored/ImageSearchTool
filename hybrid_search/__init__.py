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
二值法粗筛 + ResNet精排 的混合图库检索系统。

用法见 README.md；CLI 入口：python main.py <build|add|build-fine|search|eval|stats> ...
"""
from .config import Config
from .engine import Hit, HybridEngine, Outcome

__version__ = "1.0.0"
__all__ = ["Config", "HybridEngine", "Hit", "Outcome", "__version__"]
