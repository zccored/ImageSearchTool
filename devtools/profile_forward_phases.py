# -*- coding: utf-8 -*-
# ImageSearchTool · 原前向函数的有界阶段探针与 ABBA 扰动校验
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""保留原 _forward，用当前线程行事件和 CUDA 事件测阶段，不重写计算。

GPU 时间是同一 stream 上的事件间隔，包含主机供给间隙，不是 kernel 忙时。
仅每第 8 个调用的完整 64 行批采样，最多 24 批；A/B 是无/有探针，不是优化。
"""
import argparse
import ast
from contextlib import redirect_stdout
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from archive_batch import create_archive, digest
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from paths import GALLERY_ROOT
from profile_runtime_load import read_gpu

LABELS = ("plain_1", "probe_1", "probe_2", "plain_2")
STAGES = ("stack", "h2d", "lut", "normalize", "model", "return_cpu")


class ForwardProbe:
    def __init__(self, original):
        self.original = original
        source, first = inspect.getsourcelines(original)
        tree = ast.parse(textwrap.dedent("".join(source)))
        # 结构不再匹配就拒绝测量，避免重构后悄悄把不同代码记为同一阶段。
        body = tree.body[0].body[2].body[0].body
        assert len(body) == 5, "_forward 结构改变，请重新审核阶段边界"
        assignment, dtype, norm, model, result = body
        assert isinstance(assignment, ast.Assign)
        assert ast.unparse(assignment.value) == "torch.stack(tensors, dim=0).to(self.device)"
        assert ast.unparse(dtype.test) == "batch_t.dtype == torch.uint8"
        assert ast.unparse(norm.test) == "self.norm_on_gpu"
        assert ast.unparse(model.test) == "self.use_fp16"
        assert isinstance(result, ast.Return)
        assert ast.unparse(result.value) == "feats.float().cpu().numpy()"
        self.lines = {first+n.lineno-1: stage for n, stage in zip(
            body, ("stack", "lut", "normalize", "model", "return_cpu"))}
        self.rows = []

    def measure(self, ex, tensors):
        import torch
        if sys.gettrace() is not None:
            raise RuntimeError("当前线程已有跟踪器，拒绝覆盖")
        owner, original_stack = threading.get_ident(), torch.stack
        marks = []

        def transition(name):
            if marks and marks[-1]["stage"] == name:
                return
            now = time.perf_counter()
            event = None
            if name != "stack":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
            marks.append({"stage": name, "time": now, "event": event})

        def trace(frame, kind, arg):
            if frame.f_code is not self.original.__code__:
                return None
            if kind == "line" and frame.f_lineno in self.lines:
                transition(self.lines[frame.f_lineno])
            return trace

        def stack(*args, **kwargs):
            value = original_stack(*args, **kwargs)
            if threading.get_ident() == owner and marks and marks[-1]["stage"] == "stack":
                transition("h2d")
            return value

        start = time.perf_counter()
        try:
            with patch.object(torch, "stack", stack):
                sys.settrace(trace)
                try:
                    value = self.original(ex, tensors)
                finally:
                    sys.settrace(None)
        finally:
            # 即使原函数返回 None 或抛出异常，也先恢复跟踪器/模块函数。
            end = time.perf_counter()
            transition("end")
            marks[-1]["event"].synchronize()
            phases = []
            for left, right in zip(marks, marks[1:]):
                phases.append({"stage": left["stage"], "host_s": right["time"]-left["time"],
                    "cuda_stream_ms": left["event"].elapsed_time(right["event"])
                        if left["event"] is not None else None})
            self.rows.append({"rows": len(tensors), "forward_wall_s": end-start,
                "probe_total_s": time.perf_counter()-start, "phases": phases})
        return value

    def summary(self):
        result = {}
        for stage in STAGES:
            rows = [p for r in self.rows for p in r["phases"] if p["stage"] == stage]
            gpu = [p["cuda_stream_ms"] for p in rows if p["cuda_stream_ms"] is not None]
            result[stage] = {"calls": len(rows),
                "host_mean_ms": 1000*statistics.mean(p["host_s"] for p in rows) if rows else None,
                "cuda_stream_mean_ms": statistics.mean(gpu) if gpu else None}
        return result


def child(work, label):
    import cv2
    import torch
    from hybrid_search.fine import ResNetExtractor
    from hybrid_search.config import Config
    from hybrid_search import png_fast
    init, forward = ResNetExtractor.__init__, ResNetExtractor._forward
    probe = ForwardProbe(forward)
    counts = {"calls": 0, "rows": 0, "wall_s": 0.0}
    effective = {}

    def warmed(ex, cfg):
        init(ex, cfg)
        assert ex.device == "cuda" and ex.norm_on_gpu
        data = torch.zeros((64, 3, 224, 224), device=ex.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
            for _ in range(3):
                ex.model(data)
        torch.cuda.synchronize()
        effective.update(cv_threads=cv2.getNumThreads(), workers=ex.decode_workers,
                         torch_threads=torch.get_num_threads(), warmed=True)

    def measured(ex, tensors):
        counts["calls"] += 1
        counts["rows"] += len(tensors)
        selected = label.startswith("probe") and counts["calls"] % 8 == 0 and len(tensors) == 64 and len(probe.rows) < 24
        start = time.perf_counter()
        value = probe.measure(ex, tensors) if selected else forward(ex, tensors)
        counts["wall_s"] += time.perf_counter()-start
        assert value is not None, "前向失败，不把丢失数据的运行当作性能样本"
        return value

    sample = json.loads((work / "sample.json").read_text(encoding="utf-8"))
    before_trace, before_stack = sys.gettrace(), torch.stack
    with patch.object(ResNetExtractor, "__init__", warmed), patch.object(ResNetExtractor, "_forward", measured):
        result = run_one("tiles", label, sample, str(work), False, True, 0, png_decoder="libdeflate")
    assert sys.gettrace() is before_trace and torch.stack is before_stack
    assert counts["rows"] == result["items"] and png_fast.scratch_mb() == 0
    assert effective["cv_threads"] == Config().opencv_threads
    assert not label.startswith("probe") or probe.rows, "没有采到完整批"
    assert all(tuple(p["stage"] for p in r["phases"]) == STAGES for r in probe.rows)
    report = {"result": result, "effective": effective, "all_forward": counts,
        "sampled_batches": probe.rows, "phase_summary": probe.summary(), "scratch_mb": png_fast.scratch_mb(),
        "restored": True}
    (work / f"{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS", label, result["wall_s"], flush=True)


def foreign_python(exclude=()):
    ignored = {os.getpid(), *exclude} | {p.pid for p in psutil.Process().parents()}
    return [p.info["pid"] for p in psutil.process_iter(["pid", "name"])
        if p.info["pid"] not in ignored and (p.info["name"] or "").lower().startswith("python")]


def observe(work, label, nv, handle):
    other = foreign_python()
    if other:
        raise RuntimeError(f"发现其他 Python 进程 {other}，未启动新基准；不终止其他任务")
    samples = []
    with (work / f"{label}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-E", "-B", "-X", "utf8", __file__,
            "--work", str(work), "--label", label], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            samples.append({"epoch": time.time(), "gpu": read_gpu(nv, handle),
                            "other_python_pids": foreign_python((process.pid,))})
            time.sleep(.5)
        code = process.wait()
    (work / f"{label}_system.json").write_text(json.dumps(samples), encoding="utf-8")
    if code:
        raise RuntimeError(f"{label} 退出 {code}，保留日志；不自动重启或继续后续轮次")
    return json.loads((work / f"{label}.json").read_text(encoding="utf-8"))


def self_test():
    import torch
    from hybrid_search.fine import ResNetExtractor
    probe = ForwardProbe(ResNetExtractor._forward)
    ex = SimpleNamespace(device="cuda", norm_on_gpu=True, use_fp16=True,
        _tile_scale_lut=torch.arange(256, dtype=torch.float32).div_(255).cuda(),
        _mean_t=torch.tensor([.485, .456, .406], device="cuda").view(3, 1, 1),
        _std_t=torch.tensor([.229, .224, .225], device="cuda").view(3, 1, 1), model=torch.nn.Identity())
    inputs = [torch.arange(3*8*8, dtype=torch.uint8).reshape(3, 8, 8)]*2
    before_stack = torch.stack
    expected = probe.original(ex, inputs)
    np.testing.assert_array_equal(expected, probe.measure(ex, inputs))
    assert tuple(p["stage"] for p in probe.rows[0]["phases"]) == STAGES
    assert sys.gettrace() is None and torch.stack is before_stack
    assert probe.measure(ex, [None]) is None  # 保留生产函数既有回退，不吞成“成功”
    assert sys.gettrace() is None and torch.stack is before_stack
    entered, finished, errors = threading.Event(), threading.Event(), []

    def worker():
        entered.wait()
        try:
            torch.stack([torch.ones(2), torch.ones(2)])
        except BaseException as exc:
            errors.append(repr(exc))
        finally:
            finished.set()

    def model(data):
        entered.set()
        assert finished.wait(5), "后台测试未完成"
        return data

    ex.model = model
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        np.testing.assert_array_equal(expected, probe.measure(ex, inputs))
    finally:
        entered.set()
        thread.join(5)
    assert not errors and not thread.is_alive()
    assert tuple(p["stage"] for p in probe.rows[-1]["phases"]) == STAGES
    trace = lambda *args: None
    sys.settrace(trace)
    try:
        try:
            probe.measure(ex, inputs)
        except RuntimeError:
            pass
        else:
            raise AssertionError("覆盖了既有跟踪器")
        assert sys.gettrace() is trace
    finally:
        sys.settrace(None)
    assert torch.stack is before_stack
    print("PASS exact values / phase coverage / failure cleanup / other thread / existing tracer")


def calibrate(parent):
    """固定输入无解码竞争；全批开探针，观察单次采样扰动，不作并发开销上界。"""
    import torch
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    from hybrid_search.fine import ResNetExtractor
    if foreign_python():
        raise RuntimeError("其他 Python 仍在运行，未启动标定")
    work = create_archive(parent, parameters={"script": Path(__file__).name, "mode": "calibrate",
        "labels": LABELS, "calls_per_label": 24, "batch": 64, "seed": 20261003,
        "source_sha256": {p: digest(ROOT / p) for p in (
            "devtools/profile_forward_phases.py", "hybrid_search/fine.py", "hybrid_search/config.py")}})
    print("REPORT", work, flush=True)
    try:
        ex = HybridEngine(Config())._get_extractor()
        assert ex.device == "cuda" and ex.norm_on_gpu
        generator = torch.Generator().manual_seed(20261003)
        inputs = list(torch.randint(0, 256, (64, 3, 224, 224), dtype=torch.uint8, generator=generator))
        for _ in range(3):
            expected = ex._forward(inputs)
        assert expected is not None and np.isfinite(expected).all()
        before_stack = torch.stack
        reports = []
        for label in LABELS:
            probe = ForwardProbe(ResNetExtractor._forward)
            times, cosine = [], []
            for _ in range(24):
                start = time.perf_counter()
                value = probe.measure(ex, inputs) if label.startswith("probe") else ex._forward(inputs)
                times.append(time.perf_counter()-start)
                assert value is not None and np.isfinite(value).all()
                cos = np.sum(value*expected, axis=1)/(np.linalg.norm(value, axis=1)*np.linalg.norm(expected, axis=1))
                cosine.append(float(cos.min()))
            assert min(cosine) > .99999 and torch.stack is before_stack and sys.gettrace() is None
            reports.append({"label": label, "mean_ms": 1000*statistics.mean(times),
                "times_s": times, "min_cosine": min(cosine), "phase_summary": probe.summary()})
        plain = statistics.mean(r["mean_ms"] for r in (reports[0], reports[3]))
        observed = statistics.mean(r["mean_ms"] for r in reports[1:3])
        report = {"runs": reports, "observed_vs_plain_percent": 100*(observed/plain-1),
            "extra_mean_ms_per_sample": observed-plain,
            "note": "Fixed tensors without decode workers; probe overhead calibration, not production speedup or bound under contention."}
        (work / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("PASS", "extra_ms", observed-plain, "REPORT", work, flush=True)
    except BaseException as exc:
        (work / "failure.json").write_text(json.dumps({"error": repr(exc)}), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--n", type=int, default=1080)
    parser.add_argument("--work")
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.calibrate:
        if not args.out:
            parser.error("--calibrate 需要 --out")
        calibrate(args.out)
        return
    if args.work:
        work, gallery = Path(args.work).resolve(), Path(GALLERY_ROOT).resolve()
        if work == gallery or gallery in work.parents or not args.label:
            raise ValueError("非法输出目录或标签")
        if (work / f"tiles_{args.label}").exists() or (work / f"{args.label}.json").exists():
            raise FileExistsError("拒绝覆写已有实验")
        child(work, args.label)
        return
    if not args.out or args.n < 120:
        parser.error("需要 --out，--n 至少 120")
    sources = ["devtools/profile_forward_phases.py", "devtools/ab_build_bench.py",
        "devtools/profile_runtime_load.py", "hybrid_search/config.py", "hybrid_search/runtime.py",
        "hybrid_search/fine.py", "hybrid_search/tile_index.py", "hybrid_search/coarse.py", "hybrid_search/store.py"]
    work = create_archive(args.out, parameters={"script": Path(__file__).name, "n": args.n,
        "seed": 20260915, "labels": LABELS, "sample_every_calls": 8, "max_sampled_batches": 24,
        "source_sha256": {p: digest(ROOT / p) for p in sources}})
    print("REPORT", work, flush=True)
    nv, handle, reports = None, None, []
    try:
        sample = build_sample(args.n, 0, 20260915)
        assert len(sample) == args.n
        (work / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
        try:
            import pynvml
            pynvml.nvmlInit()
            nv, handle = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass
        for label in LABELS:
            reports.append(observe(work, label, nv, handle))
            (work / "progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            print("DONE", label, reports[-1]["result"]["wall_s"], flush=True)
        with (work / "comparisons.log").open("x", encoding="utf-8") as output, redirect_stdout(output):
            for label in LABELS[1:]:
                assert compare_tile_arrays(str(work), LABELS[0], label)
        finite = {}
        for label in LABELS:
            directory = work / f"tiles_{label}"
            with np.load(directory / "idx_tiles.coarse.npz", allow_pickle=True) as coarse, np.load(
                    directory / "idx_tiles.fine.npz", allow_pickle=True) as fine:
                assert np.array_equal(coarse["paths"], fine["paths"])
                assert np.isfinite(fine["features"]).all() and np.isfinite(coarse["hu"]).all()
                finite[label] = len(fine["paths"])
        plain = statistics.median(r["result"]["wall_s"] for r in (reports[0], reports[3]))
        observed = statistics.median(r["result"]["wall_s"] for r in reports[1:3])
        summary = {"comparisons_passed": True, "finite_aligned_rows": finite, "runs": reports,
            "observed_vs_plain_median_percent": 100*(observed/plain-1),
            "note": "Probe calibration, not optimization. CUDA stream intervals include host launch gaps; not kernel busy time."}
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
