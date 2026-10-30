# -*- coding: utf-8 -*-
"""同源预处理算子：CPU/GPU 共用同一套 torch 数学（Resize 256 + CenterCrop 224）。

与旧 PIL 路径的区别：旧路径用 PIL 的双线性实现，GPU 用 torch 的实现，两者
采样结果相差约 0.3% 余弦；本模块让 CPU 与 GPU 都走 torch 算子，仅 device 不同，
从而把"两路混跑"建立在同一空间里。
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

SIDE = 256
CROP = 224


def lane() -> str:
    """运行时选择执行设备：HERMES_PREP_LANE=cpu|cuda（默认 cpu）。"""
    return os.environ.get("HERMES_PREP_LANE", "cpu").strip().lower() or "cpu"


def transform_torch(rgb: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """HWC uint8 -> [3,224,224] float(0..1)，与 torchvision Resize/CenterCrop 等价。

    返回 CPU 张量（`_forward` 会再搬到模型设备），GPU 车道只多做一次 150 KB 的 D2H。
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    with torch.inference_mode():          # 不建 autograd 图，CPU/GPU 都省一块内存
        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    s = SIDE / float(min(h, w))
    nh = max(SIDE, int(round(h * s)))
    nw = max(SIDE, int(round(w * s)))
    t = F.interpolate(t, size=(nh, nw), mode="bilinear",
                          antialias=True, align_corners=False)
    top = (nh - CROP) // 2
    left = (nw - CROP) // 2
    t = t[:, :, top:top + CROP, left:left + CROP]
    return t[0].to("cpu")


def transform_to_device(rgb: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """与 transform_torch 同一套数学，但把结果留在 device 上，省一次 D2H。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    with torch.inference_mode():
        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device, non_blocking=True)
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        s = SIDE / float(min(h, w))
        nh = max(SIDE, int(round(h * s)))
        nw = max(SIDE, int(round(w * s)))
        t = F.interpolate(t, size=(nh, nw), mode="bilinear",
                          antialias=True, align_corners=False)
        top = (nh - CROP) // 2
        left = (nw - CROP) // 2
        t = t[:, :, top:top + CROP, left:left + CROP]
    return t[0]
