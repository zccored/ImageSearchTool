# -*- coding: utf-8 -*-
# ImageSearchTool · 正式瓦片流水线的阶段与系统负载诊断
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""不替换前向算法：一轮参考、两轮带调用计时，均新进程且同等预热。

父进程采样系统/子进程/NVML；所有输出进入新命名归档。并发阶段墙钟不能相加；
forward 是主线程同步调用的墙钟，不是 GPU kernel 活跃时间。此工具不是优化 A/B。
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from archive_batch import create_archive, digest
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from paths import GALLERY_ROOT

LABELS = ("reference", "observed_1", "observed_2")
SERIAL_STAGES = ("forward", "coarse_add", "export", "save_coarse", "save_fine", "save_meta")


class Recorder:
    def __init__(self):
        self.origin = time.perf_counter()
        self.rows = defaultdict(list)
        self.lock = threading.Lock()

    def wrap(self, fn, name, batch=False):
        def measured(*args, **kwargs):
            start, cpu = time.perf_counter(), time.thread_time()
            succeeded = False
            try:
                value = fn(*args, **kwargs)
                succeeded = True
                return value
            finally:
                end, cpu_end = time.perf_counter(), time.thread_time()
                row = {"start_s": start - self.origin, "end_s": end - self.origin,
                       "thread_cpu_s": cpu_end - cpu, "raised": not succeeded,
                       "units": len(args[1]) if batch else 1}
                with self.lock:
                    self.rows[name].append(row)
        return measured

    def summary(self, duration):
        stages = {}
        for name, rows in self.rows.items():
            stages[name] = {"calls": len(rows), "units": sum(r["units"] for r in rows),
                "wall_s": sum(r["end_s"] - r["start_s"] for r in rows),
                "thread_cpu_s": sum(r["thread_cpu_s"] for r in rows),
                "raised": sum(r["raised"] for r in rows)}
        forward = self.rows.get("forward", [])
        full = [r["end_s"] - r["start_s"] for r in forward if r["units"] == 64]
        quarters = []
        for i in range(4):
            left, right = duration*i/4, duration*(i+1)/4
            rows = [r for r in forward if left <= r["end_s"] < right or
                    (i == 3 and r["end_s"] == right)]
            quarters.append({"start_s": left, "end_s": right, "completed_batches": len(rows),
                "completed_rows": sum(r["units"] for r in rows),
                "completed_forward_wall_s": sum(r["end_s"]-r["start_s"] for r in rows)})
        ordered = sorted(full)
        return {"stages": stages, "build_s": duration,
                "main_unattributed_s": duration - sum(stages.get(k, {}).get("wall_s", 0)
                                                        for k in SERIAL_STAGES),
                "full_batch_p95_ms": 1000*ordered[min(len(ordered)-1, int(.95*len(ordered)))]
                                      if ordered else None,
                "quarters_by_completion": quarters,
                "note": "Worker stages overlap main stages. Residual includes queue wait, normalization, assembly and cleanup; not GPU idle time."}


def child(work, label):
    import cv2
    import torch
    from hybrid_search import tile_index as ti
    from hybrid_search.fine import ResNetExtractor
    from hybrid_search.coarse import CoarseIndex
    from hybrid_search.store import IndexFiles
    from hybrid_search.config import Config
    from hybrid_search import png_fast
    original_init, original_build = ResNetExtractor.__init__, ti.build_tiles
    recorder = Recorder()
    boundary = {}

    def warmed(ex, cfg):
        original_init(ex, cfg)
        if ex.device != "cuda":
            raise RuntimeError("此诊断要求 CUDA")
        data = torch.zeros((64, 3, 224, 224), device=ex.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
            for _ in range(3):
                ex.model(data)
        torch.cuda.synchronize()
        boundary["effective"] = {"opencv_threads": cv2.getNumThreads(),
            "requested": cfg.opencv_threads, "workers": ex.decode_workers,
            "torch_threads": torch.get_num_threads(), "device": ex.device, "warmed": True}

    def build(*args, **kwargs):
        recorder.origin = time.perf_counter()
        boundary["start_epoch"] = time.time()
        boundary["start_monotonic"] = time.monotonic()
        try:
            return original_build(*args, **kwargs)
        finally:
            boundary["build_s"] = time.perf_counter() - recorder.origin
            boundary["end_epoch"] = time.time()

    sample = json.loads((work / "sample.json").read_text(encoding="utf-8"))
    with ExitStack() as stack:
        stack.enter_context(patch.object(ResNetExtractor, "__init__", warmed))
        stack.enter_context(patch.object(ti, "build_tiles", build))
        if label != "reference":
            for owner, method, name, batch in (
                (ti, "_decode_to_crops", "decode_crop", False),
                (ti, "_feature_one_tile", "tile_feature", False),
                (ResNetExtractor, "_forward", "forward", True),
                (CoarseIndex, "add_results", "coarse_add", True),
                (CoarseIndex, "export_state", "export", False),
                (IndexFiles, "save_coarse", "save_coarse", False),
                (IndexFiles, "save_fine", "save_fine", False),
                (IndexFiles, "save_meta", "save_meta", False)):
                stack.enter_context(patch.object(owner, method,
                    recorder.wrap(getattr(owner, method), name, batch)))
        result = run_one("tiles", label, sample, str(work), False, True, 0,
                         png_decoder="libdeflate")
    assert boundary["effective"]["opencv_threads"] == Config().opencv_threads
    assert boundary["effective"]["warmed"] and png_fast.scratch_mb() == 0
    if label != "reference":
        assert sum(r["units"] for r in recorder.rows["forward"]) == result["items"]
        assert not any(r["raised"] for rows in recorder.rows.values() for r in rows)
    report = {"result": result, "boundary": boundary,
              "summary": recorder.summary(boundary["build_s"]), "events": recorder.rows}
    (work / f"{label}.json").write_text(json.dumps(report), encoding="utf-8")
    print("RUN", label, result["wall_s"], flush=True)


def read_gpu(nv, handle):
    if nv is None:
        return {"available": False}
    result = {}
    for name, fn, extra in (
        ("util_percent", nv.nvmlDeviceGetUtilizationRates, ()),
        ("temperature_c", nv.nvmlDeviceGetTemperature, (nv.NVML_TEMPERATURE_GPU,)),
        ("graphics_mhz", nv.nvmlDeviceGetClockInfo, (nv.NVML_CLOCK_GRAPHICS,)),
        ("power_mw", nv.nvmlDeviceGetPowerUsage, ()),
        ("pstate", nv.nvmlDeviceGetPerformanceState, ())):
        try:
            value = fn(handle, *extra)
            result[name] = value.gpu if name == "util_percent" else int(value)
        except Exception as exc:
            result[name] = None
            result[name + "_error"] = type(exc).__name__
    return result


def observe(work, label, nv, handle):
    samples = []
    start = time.monotonic()
    with (work / f"{label}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-E", "-B", "-X", "utf8", __file__,
            "--work", str(work), "--label", label], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        tracked = psutil.Process(process.pid)
        tracked.cpu_percent(None)
        psutil.cpu_percent(None)
        ignored = {os.getpid(), process.pid} | {p.pid for p in psutil.Process().parents()}
        while process.poll() is None:
            row = {"epoch": time.time(), "elapsed_s": time.monotonic()-start}
            try:
                with tracked.oneshot():
                    row.update(cpu_process_percent=tracked.cpu_percent(None),
                        cpu_system_percent=psutil.cpu_percent(None), rss_mib=tracked.memory_info().rss/2**20,
                        threads=tracked.num_threads(), io=tracked.io_counters()._asdict())
                row["gpu"] = read_gpu(nv, handle)
                row["other_python_pids"] = [p.info["pid"] for p in psutil.process_iter(["pid", "name"])
                    if p.info["pid"] not in ignored and (p.info["name"] or "").lower().startswith("python")]
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                row["sample_error"] = type(exc).__name__
            samples.append(row)
            time.sleep(.5)
        code = process.wait()
    (work / f"{label}_system.json").write_text(json.dumps(samples), encoding="utf-8")
    if code:
        raise RuntimeError(f"{label} 退出码 {code}；查看独立日志，不自动重启")
    report = json.loads((work / f"{label}.json").read_text(encoding="utf-8"))
    a, b = report["boundary"]["start_epoch"], report["boundary"]["end_epoch"]
    inside = [r for r in samples if a <= r["epoch"] <= b]
    metrics = {}
    for name in ("cpu_system_percent", "rss_mib", "threads"):
        values = [r[name] for r in inside if name in r]
        metrics[name] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)} if values else None
    for name in ("util_percent", "temperature_c", "graphics_mhz", "power_mw", "pstate"):
        values = [r["gpu"][name] for r in inside if r.get("gpu", {}).get(name) is not None]
        metrics["gpu_"+name] = {"mean": statistics.mean(values), "min": min(values), "max": max(values)} if values else None
    metrics["other_python_pids"] = sorted({pid for r in inside for pid in r.get("other_python_pids", [])})
    metrics["samples_in_build"] = len(inside)
    metrics["sample_error_count"] = sum("sample_error" in r for r in inside)
    return {"label": label, "result": report["result"], "phases": report["summary"], "system": metrics}


def self_test():
    recorder = Recorder()
    token = object()
    assert recorder.wrap(lambda: token, "ok")() is token
    try:
        recorder.wrap(lambda: 1/0, "error")()
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("异常被探针吞掉")
    assert recorder.rows["error"][0]["raised"]
    wrapped = recorder.wrap(lambda *args: 7, "forward", True)
    workers = [threading.Thread(target=lambda: wrapped(None, [1]*64)) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    summary = recorder.summary(time.perf_counter()-recorder.origin)
    assert summary["stages"]["forward"]["calls"] == 8
    assert summary["stages"]["forward"]["units"] == 512
    assert read_gpu(None, None) == {"available": False}
    print("PASS probe identity / exceptions / concurrent accounting / unavailable NVML")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--n", type=int, default=8640)
    parser.add_argument("--work")
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.work:
        work, gallery = Path(args.work).resolve(), Path(GALLERY_ROOT).resolve()
        if work == gallery or gallery in work.parents or not args.label:
            raise ValueError("非法输出目录或标签")
        if (work / f"tiles_{args.label}").exists() or (work / f"{args.label}.json").exists():
            raise FileExistsError("拒绝覆写已有实验")
        child(work, args.label)
        return
    if not args.out or args.n <= 0:
        parser.error("需要 --out 与正数 --n")
    sources = ["devtools/profile_runtime_load.py", "devtools/ab_build_bench.py",
               "hybrid_search/config.py", "hybrid_search/runtime.py", "hybrid_search/fine.py",
               "hybrid_search/tile_index.py", "hybrid_search/coarse.py", "hybrid_search/store.py"]
    work = create_archive(args.out, parameters={"script": Path(__file__).name, "n": args.n,
        "seed": 20260915, "labels": LABELS, "sampler_interval_s": .5,
        "source_sha256": {p: digest(ROOT / p) for p in sources}})
    print("REPORT", work, flush=True)
    sample = build_sample(args.n, 0, 20260915)
    assert len(sample) == args.n, "样本数量不足"
    (work / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    nv, handle = None, None
    try:
        import pynvml
        pynvml.nvmlInit()
        nv, handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        pass
    reports = []
    try:
        for label in LABELS:
            reports.append(observe(work, label, nv, handle))
            (work / "progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            print("DONE", label, reports[-1]["result"]["wall_s"], flush=True)
        with (work / "comparisons.log").open("x", encoding="utf-8") as output, redirect_stdout(output):
            for label in LABELS[1:]:
                assert compare_tile_arrays(str(work), "reference", label)
        summary = {"comparisons_passed": True, "runs": reports,
                   "note": "Same production configuration throughout; timings with probes are diagnostic, not optimization speedups."}
        (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("PASS", "REPORT", work, flush=True)
    except BaseException as exc:
        (work / "failure.json").write_text(json.dumps({"error": repr(exc), "completed": len(reports)}), encoding="utf-8")
        raise
    finally:
        if nv is not None:
            nv.nvmlShutdown()


if __name__ == "__main__":
    main()
