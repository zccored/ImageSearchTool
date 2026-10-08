# -*- coding: utf-8 -*-
# ImageSearchTool · 进程级计算资源初始化
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""在首个引擎启动工作线程前配置；不在图片处理期间切换 OpenCV 全局状态。"""
from __future__ import annotations

import operator
import threading

_opencv_lock = threading.Lock()
_opencv_requested = None
_opencv_effective = None


def _checked_threads(value: int) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError("opencv_threads 必须是 0..128 的整数") from exc
    if not 0 <= value <= 128:
        raise ValueError("opencv_threads 必须在 0..128 范围内")
    return value


def _check_locked(value: int) -> None:
    if _opencv_requested is not None and value != _opencv_requested:
        raise ValueError(
            f"OpenCV 线程配置已在本进程固定为 {_opencv_requested}"
            f"（实际 {_opencv_effective}）；不能运行中改为 {value}。"
            "请重启应用，在首次任务(含扫描)前设置；CLI 可用 --opencv-threads。")


def check_opencv_threads(value: int) -> None:
    """参数页预检，不初始化 OpenCV，也不改变进程状态。"""
    value = _checked_threads(value)
    with _opencv_lock:
        _check_locked(value)


def opencv_thread_policy():
    """返回已固定的请求策略（含 0）；尚未初始化则 None，供内部新引擎继承。"""
    with _opencv_lock:
        return _opencv_requested


def configure_opencv_threads(value: int) -> int:
    """一次性配置并返回实际值。0=不接管外部设置，不等于 cv2 的参数 0。

    同进程的多个引擎必须使用同一策略。嵌入其他 OpenCV 应用时，应在其工作
    线程启动前创建首个引擎，或用 0 由宿主负责调度。本模块不协调宿主的直接
    setNumThreads 调用，初始化后宿主也不应并发改动该全局状态。
    """
    global _opencv_requested, _opencv_effective
    value = _checked_threads(value)
    with _opencv_lock:
        _check_locked(value)
        if _opencv_requested is None:
            import cv2
            if value:
                cv2.setNumThreads(value)
            effective = cv2.getNumThreads()
            # 只有成功后才锁定，初始化异常允许重试。
            _opencv_effective = effective
            _opencv_requested = value
        return _opencv_effective
