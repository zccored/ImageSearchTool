# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — CLI 阶段化进度渲染：计数/百分比/吞吐/ETA，非 TTY 自动降级
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
CLI 阶段化进度渲染器。

接收 engine 的 progress(done, total, phase) 回调（phase ∈ {"coarse","fine"}），
实时输出：当前阶段（第几阶段/共几阶段）· 计数/总数 · 百分比 · 吞吐 · ETA。

终端（TTY）：同一阶段内用 \r 原地刷新单行，阶段切换时换行输出新标题；
管道/重定向（非 TTY）：不刷行，改为每跨越 10% 打印一行，便于留存日志。
"""
from __future__ import annotations

import sys
import time
from typing import Optional

# 阶段 key -> 展示名（中文阶段编号在创建渲染器时按顺序补上）
PHASE_TITLES = {
    "fused": "粗筛 + ResNet 融合提取（单遍解码，CPU/GPU 并行）",
    "coarse": "二值法粗筛（CPU 特征提取 + 去重）",
    "fine": "ResNet 全库特征提取",
    "verify": "图片解码校验",
}

STEP_LINE = 0.20          # 非 TTY 时每跨过 20% 输出一行（阶段中途最多 4 行）
THROTTLE_TTY = 0.12       # TTY 单行刷新最小间隔（秒）


def fmt_eta(seconds: float) -> str:
    """把秒数格式化为人类可读的预计剩余时间。"""
    if seconds < 0 or seconds != seconds:          # 负数或 NaN
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class CliPhaseProgress:
    """把 engine 的 (done,total,phase) 回调渲染成人类可读的进度输出。"""

    def __init__(self, total_phases: int, enabled: bool = True,
                 stream=None, logger=None):
        self.total_phases = max(total_phases, 1)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.logger = logger          # 非 TTY 时进度行走 logger.info
        try:
            self.is_tty = bool(self.stream.isatty())
        except Exception:             # noqa: BLE001
            self.is_tty = False
        self._cur_phase: Optional[str] = None
        self._phase_no = 0
        # 每阶段自己的计时起点（吞吐按阶段统计，跨阶段速率不混淆）
        self._t0 = 0.0
        self._d0 = 0
        self._last_draw = 0.0
        self._last_pct = -1.0
        self._line_len = 0
        self._done = False

    # ------------------------------------------------------------------
    def __call__(self, done: int, total: int, phase: str = "coarse") -> None:
        if not self.enabled:
            return
        if phase != self._cur_phase:
            self._begin_phase(phase, total)
        if total <= 0:
            return
        now = time.time()
        pct = done / total * 100.0
        # TTY：节流刷新；非 TTY：每跨过 STEP_LINE 打一行
        if self.is_tty:
            if now - self._last_draw < THROTTLE_TTY and done < total:
                return
            self._last_draw = now
            self._draw_tty(done, total, pct, now)
        else:
            rate = self._rate(done, now)
            if rate > 0 and done < total:
                eta_txt = fmt_eta((total - done) / rate)
            else:
                eta_txt = "--"
            if pct >= self._last_pct + STEP_LINE * 100:
                self._last_pct = pct
                self._line_log(f"进度 {pct:5.1f}% | {done}/{total} 张 | "
                               f"{rate:.0f} 张/秒 | 预计剩余 {eta_txt}")
            elif done >= total:
                self._line_log(f"完成 100% | {done}/{total} 张 | "
                               f"平均 {rate:.0f} 张/秒")

    # ------------------------------------------------------------------
    def _begin_phase(self, phase: str, total: int) -> None:
        if self._cur_phase is not None and not self.is_tty:
            self._line_log("-- 阶段切换 --")
        self._cur_phase = phase
        self._phase_no += 1
        self._t0 = time.time()
        self._d0 = 0
        self._last_pct = -1.0
        title = PHASE_TITLES.get(phase, phase)
        header = f"[阶段 {self._phase_no}/{self.total_phases}] {title}"
        total_txt = f"，共 {total} 张" if total else ""
        if self.is_tty:
            if self._phase_no > 1:
                self.stream.write("\n")
            self._write_raw(f"{header}{total_txt} …")
        else:
            self._line_log(f"{header}{total_txt}")

    def _rate(self, done: int, now: float) -> float:
        dt = now - self._t0
        return (done - self._d0) / dt if dt > 0.1 else 0.0

    # ------------------------------------------------------------------
    def _draw_tty(self, done: int, total: int, pct: float, now: float) -> None:
        rate = self._rate(done, now)
        speed = f"{rate:.0f} 张/秒" if rate > 0 else "…"
        if rate > 0 and done > self._d0:
            eta = fmt_eta((total - done) / rate)
        else:
            eta = "--"
        line = (f"  {done:,}/{total:,} 张 ({pct:5.1f}%)  |  {speed}  |  "
                f"预计剩余 {eta}")
        # 覆盖上一行：写满整行再回退，避免残留旧字符
        pad = max(self._line_len - len(line), 0)
        self._write_raw(line + " " * pad)

    def _line_log(self, text: str) -> None:
        if self.logger is not None:
            self.logger.info(text)
        else:
            self.stream.write(text + "\n")
            self.stream.flush()

    def _write_raw(self, text: str) -> None:
        self.stream.write("\r" + text)
        self.stream.flush()
        self._line_len = len(text)

    # ------------------------------------------------------------------
    def finish(self) -> None:
        """阶段/命令收尾：清掉单行进度，回到行首输出新行。"""
        if not self.enabled:
            return
        if self.is_tty and self._line_len:
            self.stream.write("\r" + " " * self._line_len + "\r")
            self.stream.flush()
        self._done = True
