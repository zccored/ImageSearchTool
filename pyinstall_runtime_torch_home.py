# -*- coding: utf-8 -*-
"""PyInstaller runtime hook：把 TORCH_HOME 指向包内自带权重目录，
使 torchvision 预训练权重离线可用（打包时 datas 已把 resnet18
权重放进 <包>/torch_home/hub/checkpoints/）。setdefault：外部若
显式设置了 TORCH_HOME 则尊重外部值。"""
import os
import sys

base = getattr(sys, "_MEIPASS", None) or os.path.dirname(
    os.path.abspath(sys.executable))
os.environ.setdefault("TORCH_HOME", os.path.join(base, "torch_home"))
