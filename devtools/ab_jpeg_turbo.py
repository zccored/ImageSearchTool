# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — C2 · JPEG 解码器建库级配对 A/B（零侵入）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""用 monkey-patch 跑"换 JPEG 解码器"的**建库级配对 A/B**，不动任何主程序代码。

为什么这么做：
  * 项目纪律：性能结论只认 ab_build_bench 配对 A/B（探针的整进程墙钟波动可达 15%）；
  * 但"先改代码再测"有两个风险：改错了不知道，以及改完才发现收益不够要回退。
    所以先**零侵入**验证收益，再决定要不要动主程序。
  * 参照 devtools/probe_pipeline_attrib.py 的做法：import 之后替换函数引用，退出即失效。
    注意：fine.py / tile_index.py 用的是 `from .io_utils import decode_rgb`（**绑定引用**），
    只改 io_utils 属性覆盖不到它们，必须一并替换 —— 否则 A/B 两侧只改了一半的调用点，
    测出来的差异会被稀释（这是本脚本相对旧探针的关键修正）。

同时统计 **按格式的解码 CPU 归属**（JPEG / PNG 各占多少核秒），
这样"解码级 1.12×"才能换算成"建库级百分之几"。

用法:
  python -E devtools/ab_jpeg_turbo.py                       # whole, 540+60, 4 轮
  python -E devtools/ab_jpeg_turbo.py --mode tiles --rounds 4 --n 540 --dup 60
"""
import argparse
import json
import os
import statistics as st
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "devtools"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import ab_build_bench as AB                                    # noqa: E402

DLL = os.environ.get("TURBOJPEG_DLL") or os.path.join(
    os.environ.get("TEMP", ""), "c2_jpeg", "turbojpeg.dll")
OUT_DIR = os.path.join(_HERE, "perf_reports")
_TARGET = 2048

_LOCK = threading.Lock()
ACC = {"jpeg_s": 0.0, "jpeg_n": 0, "png_s": 0.0, "png_n": 0.0,
       "other_s": 0.0, "other_n": 0, "tj_hit": 0, "tj_fb": 0}
STATE = {"use_tj": False}
_TL = threading.local()
_ORIG = {}


def _acc(key, dt):
    with _LOCK:
        ACC[key + "_s"] = ACC.get(key + "_s", 0.0) + dt
        ACC[key + "_n"] = ACC.get(key + "_n", 0) + 1


def wrap_stage(mod, name, key):
    """给某个阶段函数装计时器（两侧都装，保证对称），用于归因多出来的 CPU。"""
    fn = getattr(mod, name, None)
    if fn is None or getattr(fn, "_ab_wrapped", False):
        return False

    def w(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            _acc(key, time.perf_counter() - t0)
    w._ab_wrapped = True
    setattr(mod, name, w)
    return True


def scale_of(w, h):
    ms = max(w, h)
    if ms <= _TARGET * 1.25:
        return (1, 1)
    if ms <= _TARGET * 2.5:
        return (1, 2)
    if ms <= _TARGET * 5:
        return (1, 4)
    return (1, 8)


def tj():
    if getattr(_TL, "obj", None) is None:
        from turbojpeg import TJPF_RGB, TurboJPEG
        _TL.obj = TurboJPEG(DLL)
        _TL.rgb = TJPF_RGB
    return _TL.obj, _TL.rgb


def make_wrapper(iu, orig):
    probe_fn = iu._probe

    def wrapper(data):
        t0 = time.perf_counter()
        try:
            if STATE["use_tj"]:
                probe = probe_fn(data)
                if (probe is not None and probe[0] == "JPEG" and probe[2] == 1
                        and data[-2:] == b"\xff\xd9"):    # 缺 EOI=疑似截断，交回 cv2
                    try:
                        obj, TJPF_RGB = tj()
                        arr = obj.decode(data, pixel_format=TJPF_RGB,
                                         scaling_factor=scale_of(*probe[1]))
                        with _LOCK:
                            ACC["tj_hit"] += 1
                        return arr
                    except Exception:                     # noqa: BLE001
                        with _LOCK:
                            ACC["tj_fb"] += 1
            return orig(data)
        finally:
            dt = time.perf_counter() - t0
            ext = data[6:10] if data[:2] == b"\xff\xd8" else None
            with _LOCK:
                if data[:8] == b"\x89PNG\r\n\x1a\n":
                    ACC["png_s"] += dt
                    ACC["png_n"] += 1
                elif data[:2] == b"\xff\xd8":
                    ACC["jpeg_s"] += dt
                    ACC["jpeg_n"] += 1
                else:
                    ACC["other_s"] += dt
                    ACC["other_n"] += 1
                del ext
    return wrapper


def install(use_tj, big_conc=0):
    """把 wrapper 装到所有 decode_rgb 的调用点上；use_tj 控制是否走 TurboJPEG。"""
    import hybrid_search.io_utils as iu
    import hybrid_search.fine as fine
    import hybrid_search.tile_index as ti
    if not _ORIG:
        _ORIG["iu"] = iu.decode_rgb
        _ORIG["fine"] = getattr(fine, "decode_rgb", None)
        _ORIG["ti"] = getattr(ti, "decode_rgb", None)
        _ORIG["wrapper"] = make_wrapper(iu, _ORIG["iu"])
        # 阶段归因：读盘 / 前向（两侧都装，对称）
        for mod, nm, key in ((iu, "read_bytes", "read"), (fine, "read_bytes", "read"),
                             (ti, "read_bytes", "read")):
            wrap_stage(mod, nm, key)
        for cand in ("_forward", "forward"):
            if wrap_stage(fine.ResNetExtractor, cand, "fwd"):
                break
        if big_conc:
            _orig_sbl = iu.set_big_decode_limit

            def _sbl(_n, _orig=_orig_sbl):
                _orig(big_conc)
            iu.set_big_decode_limit = _sbl
            print("   [实验] 大图解码并发上限强制为 %d" % big_conc)
    STATE["use_tj"] = use_tj
    iu.decode_rgb = _ORIG["wrapper"]
    for mod, key in ((fine, "fine"), (ti, "ti")):
        if _ORIG[key] is not None:
            setattr(mod, "decode_rgb", _ORIG["wrapper"])
    # 让"补丁是否真的生效"可观测：engine 在函数体内 import，取的就是当前属性
    return iu.decode_rgb is _ORIG["wrapper"]


def reset_acc():
    with _LOCK:
        for k in ACC:
            ACC[k] = 0.0 if k.endswith("_s") else 0


def mem_snapshot():
    """页错误/私有提交量快照（定位"解码器换了却拖慢无关阶段"的内存机制）。"""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        d = {"pf": int(getattr(mi, "num_page_faults", 0)),
             "private_mb": round(getattr(mi, "private", 0) / 2 ** 20, 1),
             "peak_wset_mb": round(getattr(mi, "peak_wset", 0) / 2 ** 20, 1)}
        return d
    except Exception:                                     # noqa: BLE001
        return {"pf": 0, "private_mb": 0.0, "peak_wset_mb": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["whole", "tiles"], default="whole")
    ap.add_argument("--n", type=int, default=540)
    ap.add_argument("--dup", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--work", default=AB.DEFAULT_WORK)
    ap.add_argument("--tag", default="c2j")
    ap.add_argument("--png-decoder", default="libdeflate",
                    choices=["cv2", "imagecodecs", "pillow", "libdeflate"],
                    help="PNG 解码器（**生产默认是 libdeflate**；ab_build_bench 自己的"
                         "默认是 cv2，两边不一致会让结论跑偏，故这里显式给默认值）")
    ap.add_argument("--big-conc", type=int, default=0,
                    help="把大图解码并发上限改成该值（0=按 Config），用于定位并发交互")
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(DLL):
        print("**缺 turbojpeg.dll**：请先跑 devtools/fetch_turbojpeg.py"); return 2
    import turbojpeg
    print("TurboJPEG %s · DLL %s" % (turbojpeg.__version__, DLL))

    print("构建固定样本（n=%d dup=%d seed=%d）…" % (a.n, a.dup, a.seed))
    sample = AB.build_sample(a.n, a.dup, a.seed)
    print("样本 %d 张" % len(sample))

    # 关掉预处理缓存，测的就是解码/特征流水线本身
    common = dict(mode=a.mode, sample=sample, work=a.work, prep_cache=False,
                  dedup_prefilter=True, tile_flush_ms=0, png_decoder=a.png_decoder)
    print("PNG 解码器 = %s（生产默认 libdeflate）" % a.png_decoder)

    runs = []
    for r in range(1, a.rounds + 1):
        order = [("cv2", False), ("turbo", True)] if r % 2 else [("turbo", True), ("cv2", False)]
        for arm, use_tj in order:
            label = "%s_%s_r%d" % (a.tag, arm, r)
            install(use_tj, a.big_conc)
            reset_acc()
            print("\n--- 轮 %d | %s（TurboJPEG=%s）---" % (r, label, use_tj))
            m0 = mem_snapshot()
            res = AB.run_one(label=label, **common)
            m1 = mem_snapshot()
            res["mem"] = {"page_faults": m1["pf"] - m0["pf"],
                          "private_mb_end": m1["private_mb"],
                          "peak_wset_mb": m1["peak_wset_mb"]}
            res["arm"] = arm
            res["round"] = r
            res["decode_cpu"] = {k: round(v, 3) for k, v in ACC.items()}
            runs.append(res)
            print("   wall %.2f s | CPU %.1f 核秒 | %.2f 核 | %.1f 张/s | 解码JPEG %.1f s/%d张 "
                  "| 解码PNG %.1f s/%d张 | TJ命中 %d 回退 %d"
                  % (res["wall_s"], res["cpu_s"], res["cores"], res["img_per_s"],
                     ACC["jpeg_s"], ACC["jpeg_n"], ACC["png_s"], ACC["png_n"],
                     ACC["tj_hit"], ACC["tj_fb"]))
            print("   阶段归因：读盘 %.2f s/%d 次 | 前向 %.2f s/%d 次 | 页错误 +%d | 私有提交 %.0f MB"
                  % (ACC.get("read_s", 0.0), ACC.get("read_n", 0),
                     ACC.get("fwd_s", 0.0), ACC.get("fwd_n", 0),
                     res["mem"]["page_faults"], res["mem"]["private_mb_end"]))

    # ---------------- 配对汇总
    print("\n===== 配对 A/B 汇总（%s 模式，%d 轮）=====" % (a.mode, a.rounds))
    print("   %-6s %10s %10s %9s %10s %9s" % ("轮", "cv2 wall", "turbo wall", "Δwall",
                                             "cv2 CPU", "turbo CPU"))
    dw, dc, dp = [], [], []
    for r in range(1, a.rounds + 1):
        cv = next(x for x in runs if x["round"] == r and x["arm"] == "cv2")
        tb = next(x for x in runs if x["round"] == r and x["arm"] == "turbo")
        w = 100 * (tb["wall_s"] - cv["wall_s"]) / cv["wall_s"]
        c = 100 * (tb["cpu_s"] - cv["cpu_s"]) / cv["cpu_s"]
        p = 100 * (tb["img_per_s"] - cv["img_per_s"]) / cv["img_per_s"]
        dw.append(w)
        dc.append(c)
        dp.append(p)
        print("   %-6d %9.2fs %9.2fs %+8.2f%% %9.1f %9.1f  (CPU %+.2f%%, 吞吐 %+.2f%%)"
              % (r, cv["wall_s"], tb["wall_s"], w, cv["cpu_s"], tb["cpu_s"], c, p))
    print("\n   Δwall 中位 %+.2f%%（各轮 %s）" % (st.median(dw), [round(x, 2) for x in dw]))
    print("   ΔCPU  中位 %+.2f%%（各轮 %s）" % (st.median(dc), [round(x, 2) for x in dc]))
    print("   Δ吞吐 中位 %+.2f%%" % st.median(dp))

    # ---------------- 解码 CPU 归属（换算式）
    cv = next(x for x in runs if x["arm"] == "cv2")
    tb = next(x for x in runs if x["arm"] == "turbo")
    dcpu = cv["decode_cpu"]
    tot_dec = dcpu["jpeg_s"] + dcpu["png_s"] + dcpu["other_s"]
    print("\n===== 解码 CPU 归属（cv2 侧，%d 张样本）=====" % len(sample))
    print("   JPEG %.2f 核秒 / %d 张（%.2f ms/张）→ 占解码 %.1f%%"
          % (dcpu["jpeg_s"], dcpu["jpeg_n"], dcpu["jpeg_s"] / max(dcpu["jpeg_n"], 1) * 1e3,
             100 * dcpu["jpeg_s"] / max(tot_dec, 1e-9)))
    print("   PNG  %.2f 核秒 / %d 张（%.2f ms/张）→ 占解码 %.1f%%"
          % (dcpu["png_s"], dcpu["png_n"], dcpu["png_s"] / max(dcpu["png_n"], 1) * 1e3,
             100 * dcpu["png_s"] / max(tot_dec, 1e-9)))
    if dcpu["other_n"]:
        print("   其它 %.2f 核秒 / %d 张" % (dcpu["other_s"], dcpu["other_n"]))
    print("   TurboJPEG 侧：命中 %d，回退 %d（回退率 %.2f%%）"
          % (tb["decode_cpu"]["tj_hit"], tb["decode_cpu"]["tj_fb"],
             100.0 * tb["decode_cpu"]["tj_fb"]
             / max(tb["decode_cpu"]["tj_hit"] + tb["decode_cpu"]["tj_fb"], 1)))
    jcv, jtb = cv["decode_cpu"]["jpeg_s"], tb["decode_cpu"]["jpeg_s"]
    if jtb:
        print("   **JPEG 解码本段加速 %.3fx**（%.2f → %.2f 核秒；占建库 CPU 的 %.1f%%）"
              % (jcv / jtb, jcv, jtb,
                 100 * jcv / max(cv["cpu_s"], 1e-9)))
    # 换算：JPEG 解码在建库总 CPU 中的占比 × 其降幅 = 建库级 CPU 降幅上限
    share = jcv / max(cv["cpu_s"], 1e-9)
    gain = (1 - jtb / jcv) if jcv else 0
    print("   ⇒ 换算建库级 CPU 降幅上限 ≈ %.2f%%（%.1f%% × %.1f%%；与实际 ΔCPU 互为校验）"
          % (100 * share * gain, 100 * share, 100 * gain))

    # ---------------- 索引位级比对
    r1cv = "%s_cv2_r1" % a.tag
    r1tb = "%s_turbo_r1" % a.tag
    print("\n===== 索引位级比对（%s vs %s）=====" % (r1cv, r1tb))
    try:
        AB.compare_arrays(a.work, r1cv, r1tb)
    except Exception as e:                                    # noqa: BLE001
        print("   compare_arrays 失败：%r" % e)

    meta = {"tool": "ab-jpeg-turbo", "ts": time.strftime("%Y%m%d-%H%M%S"),
            "mode": a.mode, "n": len(sample), "rounds": a.rounds, "dll": DLL,
            "delta_wall_pct_median": round(st.median(dw), 3),
            "delta_cpu_pct_median": round(st.median(dc), 3),
            "delta_thr_pct_median": round(st.median(dp), 3),
            "delta_wall_pct": [round(x, 3) for x in dw],
            "delta_cpu_pct": [round(x, 3) for x in dc],
            "decode_cpu_cv2": cv["decode_cpu"], "decode_cpu_turbo": tb["decode_cpu"],
            "jpeg_decode_share_of_build_cpu": round(share, 4),
            "jpeg_decode_gain": round(gain, 4),
            "runs": [{k: v for k, v in r.items() if k != "arrays"} for r in runs]}
    jf = os.path.join(OUT_DIR, "ab_jpeg_turbo_%s_%s.json" % (a.mode, meta["ts"]))
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\nJSON:", jf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
