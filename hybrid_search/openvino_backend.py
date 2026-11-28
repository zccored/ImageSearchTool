# -*- coding: utf-8 -*-
"""OpenVINO 推理后端（可选插件，默认由 Config.backend=auto 选择）。

NAS: i5-1235U Iris Xe 80EU，OpenVINO GPU batch=8 实测约 465 张/s（仅前向）。
本机: RTX 4060 时 auto 仍优先 torch CUDA；UHD 核显不会比 4060 快。

约束：
  * 预处理/解码沿用 ResNetExtractor 的流水线，只替换前向；
  * OpenVINO 设备名可能是 GPU 或 GPU.0；
  * 模型 IR 默认打包在 model/resnet18.xml；
  * 找不到 OpenVINO/模型时由 factory 回退 torch。
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .io_utils import LOGGER

_MODEL_NAME = "resnet18.xml"


def _ov():
    import openvino as ov
    return ov


def available_devices() -> List[str]:
    try:
        return list(_ov().Core().available_devices)
    except Exception:
        return []


def has_openvino() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except Exception:
        return False


def resolve_model_path(cfg) -> str:
    cands = []
    p = getattr(cfg, "openvino_model", "") or ""
    if p:
        cands.append(p)
    env = os.environ.get("IMAGE_SEARCH_OPENVINO_MODEL", "")
    if env:
        cands.append(env)
    base = getattr(sys, "_MEIPASS", "") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    cands += [
        os.path.join(base, "model", _MODEL_NAME),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "model", _MODEL_NAME),
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError("找不到 OpenVINO 模型 IR：%s（可用 openvino_model 指定）"
                            % _MODEL_NAME)


def resolve_device(cfg, override: str = "") -> str:
    want = (override or getattr(cfg, "openvino_device", "auto") or "auto").strip()
    devs = available_devices()
    if want.lower() == "auto":
        for d in ("GPU", "GPU.0", "GPU.1", "CPU"):
            if d in devs:
                return d
        return "CPU"
    if want in devs:
        return want
    # 允许写 GPU.0，但机器只暴露 GPU
    if want.startswith("GPU"):
        for d in ("GPU", "GPU.0", "GPU.1"):
            if d in devs:
                return d
    LOGGER.warning("OpenVINO 设备 %s 不可用（%s），回退 CPU", want, devs)
    return "CPU"


class OpenVINOExtractor:
    """只替换 ResNetExtractor 的前向；其余接口由 mixin/子类提供。"""

    def __init__(self, cfg):
        # 延迟 import，避免 auto 时无 OpenVINO 也去加载
        from .fine import ResNetExtractor
        self._base_cls = ResNetExtractor
        self.cfg = cfg
        self.device = "openvino"
        self.model = None
        self.feature_dim = 512
        self.batch = int(cfg.batch) if cfg.batch and cfg.batch > 0 else 16
        from .fine import _auto_decode_workers
        self.decode_workers = _auto_decode_workers(cfg, False)
        self.use_fp16 = False
        self.norm_on_gpu = False
        self._compiled: Dict[int, object] = {}
        self._compile_lock = threading.Lock()
        self._model_path = resolve_model_path(cfg)
        self._device = resolve_device(cfg)
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        # 与 torch 路径一致的 CPU 预处理（Resize 256/CenterCrop 224/ToTensor/Normalize）
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.tile_transform = self.transform
        self._tile_scale_lut = None
        LOGGER.info("精排：OpenVINO 设备=%s 模型=%s 批=%d",
                    self._device, os.path.basename(self._model_path), self.batch)
        self._compile(self.batch)

    def _compile(self, batch: int):
        with self._compile_lock:
            if batch in self._compiled:
                return self._compiled[batch]
            ov = _ov()
            core = ov.Core()
            m = core.read_model(self._model_path)
            m.reshape({0: ov.PartialShape([int(batch), 3, 224, 224])})
            t0 = time.time()
            cm = core.compile_model(m, self._device)
            self._compiled[batch] = cm
            LOGGER.info("精排：OpenVINO 编译 batch=%d 设备=%s 耗时 %.2fs",
                        batch, self._device, time.time() - t0)
            return cm

    def prep(self, rgb):
        mode = getattr(self.cfg, "prep", "cpu")
        if mode == "torch":
            from .prep_torch import transform_torch
            return transform_torch(rgb, getattr(self.cfg, "prep_lane", "cpu"))
        from PIL import Image
        return self.transform(Image.fromarray(rgb))

    def _forward(self, tensors) -> Optional[np.ndarray]:
        import torch
        try:
            if (len(tensors) == 1 and torch.is_tensor(tensors[0])
                    and tensors[0].dim() == 4):
                arr = tensors[0].detach().cpu().numpy()
            else:
                arr = torch.stack(tensors, dim=0).detach().cpu().numpy()
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            cm = self._compile(arr.shape[0])
            out = cm([arr])
            return np.asarray(out[cm.outputs[0]], dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("OpenVINO 批次前向失败（%d 张）: %r", len(tensors), e)
            return None

    # 以下方法从 ResNetExtractor 借用（不调用其 __init__）
    stream_decode = None
    forward_rgb = None
    forward_gpu = None

    def extract_batch(self, paths, progress=None, frame_sink=None):
        """用同一条 stream_decode 流水线，避免继承 torch 的 extract_batch。"""
        def on_batch(ok, tensors, _payloads):
            feats = self._forward(tensors)
            if feats is None:
                return [], None
            n = feats.astype(np.float32)
            norms = np.linalg.norm(n, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            n /= norms
            return ok, n
        return self.stream_decode(paths, prep=None, progress=progress,
                                  on_batch=on_batch)

    def extract_one(self, path):
        """查询图用与建库相同的 batch 形状（重复填充），避免 batch1 核/布局差异。"""
        n = max(1, int(self.batch))
        ok, feats = self.stream_decode([path] * n)
        if not ok or feats is None or len(feats) == 0:
            return None
        out = np.asarray(feats[0], dtype=np.float32)
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-8 else out


def create_extractor(cfg):
    """auto：本机 CUDA 优先；无 CUDA 且 OpenVINO 可用时走 iGPU/CPU。"""
    backend = (getattr(cfg, "backend", "auto") or "auto").strip().lower()
    if backend in ("torch", "pytorch"):
        from .fine import ResNetExtractor
        return ResNetExtractor(cfg)
    if backend in ("openvino", "ov"):
        if not has_openvino():
            raise RuntimeError("backend=openvino 但未安装 openvino")
        return OpenVINOExtractor(cfg)
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            from .fine import ResNetExtractor
            return ResNetExtractor(cfg)
    except Exception:
        pass
    if has_openvino():
        try:
            return OpenVINOExtractor(cfg)
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("OpenVINO 初始化失败，回退 torch：%r", e)
    from .fine import ResNetExtractor
    return ResNetExtractor(cfg)


# 绑定继承方法（避免复制整段解码流水线）
def _bind_methods():
    from .fine import ResNetExtractor
    for name in ("stream_decode", "forward_rgb", "forward_gpu"):
        setattr(OpenVINOExtractor, name, getattr(ResNetExtractor, name))


_bind_methods()
