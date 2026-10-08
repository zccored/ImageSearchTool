# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 建库耗时分解探针（定位"消失的核"）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE.
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""零侵入耗时分解：用 monkey-patch 给"解码"与"主线程前向"装计时器，然后跑真实建库。

要回答的问题：解码探针显示"解码池有效核数 15.5"，而建库只跑到 8.7~9.8 核 —— 差的那
6~7 个核被谁占了？本探针给出：
  t_decode_total   解码池累计忙时 → 与 (线程数 × 墙钟) 比 = 池忙碌率
  t_forward_total  主线程累计 GPU/前向耗时 → 与墙钟比 = 主线程占比
  wall             总墙钟
  ⇒ 未归因 = wall − t_forward_total（主线程若在等，就是生产端供给不足或屏障等待）

不修改任何主逻辑；只在 import 之后替换两个函数引用，退出即失效。
用法: python -E devtools/probe_pipeline_attrib.py [--mode whole|tiles] [--n 540] [--dup 60]
"""
import os
import statistics as st
import sys
import threading
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

_LOCK = threading.Lock()
ACC = {"decode": 0.0, "decode_n": 0, "forward": 0.0, "forward_n": 0,
       "read": 0.0, "read_n": 0, "decode_ms": []}
PATCHED = {}


def _wrap(mod, name, key, rec_list=False):
    fn = getattr(mod, name, None)
    if fn is None:
        PATCHED[name] = "缺失"
        return False

    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            with _LOCK:
                if key + "_n" in ACC:
                    ACC[key] += dt
                    ACC[key + "_n"] += 1
                else:
                    ACC[key] = ACC.get(key, 0.0) + dt
                if rec_list:
                    ACC["decode_ms"].append(dt * 1e3)
    setattr(mod, name, wrapper)
    PATCHED[name] = "已装"
    return True


def main() -> int:
    import hybrid_search.io_utils as iu
    _wrap(iu, "decode_rgb", "decode", rec_list=True)
    _wrap(iu, "decode_gray", "decode")
    try:
        import hybrid_search.fine as fine
        for cand in ("_forward", "forward"):
            if _wrap(fine.ResNetExtractor, cand, "forward"):
                break
    except Exception as e:                       # noqa: BLE001
        PATCHED["ResNetExtractor"] = "异常 %s" % e
    print("补丁状态:", PATCHED)

    argv = sys.argv[1:] or ["--mode", "whole", "--n", "540", "--dup", "60"]
    sys.argv = ["ab_build_bench.py"] + argv
    t0 = time.perf_counter()
    import runpy
    try:
        runpy.run_path(os.path.join(_HERE, "devtools", "ab_build_bench.py"),
                       run_name="__main__")
    except SystemExit:                 # 基准台以 SystemExit 结束，必须吞掉才能打印统计
        pass
    wall = time.perf_counter() - t0

    dec_s, dec_n = ACC["decode"], ACC["decode_n"]
    fwd_s, fwd_n = ACC["forward"], ACC["forward_n"]
    ms = sorted(ACC["decode_ms"])
    print("\n==== 分解结果（wall %.2f s，含模型加载与索引落盘）====" % wall)
    if dec_n:
        print("  解码：%d 张 / %.2f 核秒 → 单张均值 %.1f ms（p50 %.1f，p95 %.1f，max %.1f）"
              % (dec_n, dec_s, dec_s / dec_n * 1e3, ms[len(ms) // 2],
                 ms[int(len(ms) * .95)], ms[-1]))
        print("        解码累计占单线程时间 %.1f%%（= 核秒/墙钟 = %.2f 核）"
              % (100 * dec_s / wall, dec_s / wall))
    if fwd_n:
        print("  前向：%d 批 / %.2f 核秒 → 占单线程时间 %.1f%%（主线程若在此阻塞即为 GPU 瓶颈）"
              % (fwd_n, fwd_s, 100 * fwd_s / wall))
    other = wall - dec_s - fwd_s
    print("  未归因（读盘/预处理/队列/屏障/落盘等）：%.2f s = %.1f%% 单线程时间"
          % (other, 100 * other / wall))
    print("  判读：解码占比高→生产端 CPU 受限；前向占比高→GPU 受限；"
          "两者都不高→屏障/队列/IO 开销")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
