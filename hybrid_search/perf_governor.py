# -*- coding: utf-8 -*-
"""性能预算器：启动预热/进程优先级/构建期 GPU keep-alive + 机器画像。

目的不是“超频”，而是在不要求管理员权限的前提下，让 GPU 在构建期间维持
boost 区间，避免批间空隙掉到 210 MHz；同时把机器画像（CPU/GPU/带宽）写入
日志，供自动参数选择使用。
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time

LOGGER = None


def _log():
    global LOGGER
    if LOGGER is None:
        from .io_utils import LOGGER as _L
        LOGGER = _L
    return LOGGER


def apply_process_budget() -> None:
    """Windows: 提高进程优先级；其他平台 no-op。失败不影响构建。"""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        HIGH_PRIORITY_CLASS = 0x00000080
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
    except Exception as e:                       # noqa: BLE001
        _log().debug("设置进程优先级失败: %r", e)


def gpu_profile() -> dict:
    """轻量机器画像：GPU 名称/显存、CPU 核数、H2D 带宽（失败则 None）。"""
    prof = {"cpu": os.cpu_count() or 0, "cuda": False}
    try:
        import torch
        prof["cuda"] = bool(torch.cuda.is_available())
        if prof["cuda"]:
            prop = torch.cuda.get_device_properties(0)
            prof["gpu"] = prop.name
            prof["vram_mb"] = int(prop.total_memory / (1024 * 1024))
            prof["cc"] = f"{prop.major}.{prop.minor}"
            # H2D 带宽（32 MB）
            buf = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
            gpu = torch.empty_like(buf, device="cuda")
            torch.cuda.synchronize()
            t = time.perf_counter()
            gpu.copy_(buf, non_blocking=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            prof["h2d_gbps"] = round(32 / 1024 / max(dt, 1e-9), 1)
            del buf, gpu
            torch.cuda.empty_cache()
    except Exception as e:                       # noqa: BLE001
        prof["error"] = repr(e)
    return prof


class PerfGovernor:
    """构建期间维持 GPU 性能区间；off 时完全 no-op。"""

    def __init__(self, cfg):
        self.mode = str(getattr(cfg, "gpu_governor", "auto") or "auto").lower()
        self.cfg = cfg
        self.profile = {}
        self._stop = threading.Event()
        self._thread = None
        self._stream = None

    def __enter__(self):
        if self.mode == "off":
            return self
        apply_process_budget()
        try:
            import torch
            if not torch.cuda.is_available():
                return self
            self.profile = gpu_profile()
            self._tune()
            # 预热：把 GPU 从 210 MHz 拉到 boost（约 100~200 ms 的算力）
            a = torch.randn(1024, 1024, device="cuda")
            for _ in range(40):
                a = a @ a
            torch.cuda.synchronize()
            del a
            if self.mode == "keepalive":
                # 显式开启才跑常驻小核：本机实测会让批间竞争、慢约 5%
                self._stream = torch.cuda.Stream()
                self._thread = threading.Thread(target=self._keepalive, daemon=True)
                self._thread.start()
            _log().info("性能预算：%s，%s MB 显存，H2D %s GB/s，keep-alive=%s",
                        self.profile.get("gpu"), self.profile.get("vram_mb"),
                        self.profile.get("h2d_gbps"),
                        "on" if self._thread else "off")
        except Exception as e:                   # noqa: BLE001
            _log().warning("性能预算初始化失败（继续构建）：%r", e)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    def _tune(self) -> None:
        """按机器画像收敛参数：低核/小显存自动降档，高端机保持实测最优值。"""
        cfg = self.cfg
        prof = self.profile
        cpu = int(prof.get("cpu") or 0)
        vram = int(prof.get("vram_mb") or 0)
        changes = []
        if cpu and cpu < 12:
            tw = max(8, cpu)
            if int(getattr(cfg, "tile_workers", 0) or 0) > tw:
                cfg.tile_workers = tw
                changes.append(f"tile_workers={tw}")
            if int(getattr(cfg, "tile_decode_slots", 0) or 0) > tw:
                cfg.tile_decode_slots = tw
                changes.append(f"tile_decode_slots={tw}")
            mw = max(4, min(12, cpu // 2))
            if int(getattr(cfg, "tile_mp_workers", 0) or 0) not in (0, mw):
                cfg.tile_mp_workers = mw
                changes.append(f"tile_mp_workers={mw}")
        if vram and vram < 4096 and int(getattr(cfg, "tile_fwd_batch", 64) or 64) > 32:
            cfg.tile_fwd_batch = 32
            changes.append("tile_fwd_batch=32")
        if changes:
            _log().info("性能预算自适应：%s", ", ".join(changes))

    def _keepalive(self):
        import torch
        try:
            a = torch.randn(512, 512, device="cuda")
            with torch.cuda.stream(self._stream):
                while not self._stop.wait(0.04):
                    for _ in range(2):           # ~2~4 ms 低占空比算力，维持 boost
                        a = torch.mm(a, a)
        except Exception as e:                   # noqa: BLE001
            _log().debug("GPU keep-alive 退出: %r", e)
