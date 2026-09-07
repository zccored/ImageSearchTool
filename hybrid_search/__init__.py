# -*- coding: utf-8 -*-
"""
二值法粗筛 + ResNet精排 的混合图库检索系统。

用法见 README.md；CLI 入口：python main.py <build|add|build-fine|search|eval|stats> ...
"""
from .config import Config
from .engine import Hit, HybridEngine, Outcome

__version__ = "1.0.0"
__all__ = ["Config", "HybridEngine", "Hit", "Outcome", "__version__"]
