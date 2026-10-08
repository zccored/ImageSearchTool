# -*- coding: utf-8 -*-
# ImageSearchTool · 就绪队列诊断（不修改生产调度）
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""前向边界队列快照不是 GPU 活跃率，也不是异步旁路的性能承诺。"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "devtools")]
from ab_build_bench import build_sample, run_one, compare_tile_arrays
from hybrid_search.fine import ResNetExtractor
from paths import GALLERY_ROOT


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1080)
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("输出必须在图库外")
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ready_queue_", dir=base))
    sample = build_sample(args.n, 0, 20260915)
    original_init = ResNetExtractor.__init__
    original_forward = ResNetExtractor._forward
    rows = []

    def init(ex, cfg):
        original_init(ex, cfg)
        if ex.device != "cuda":
            raise RuntimeError("本诊断需要 CUDA")
        data = torch.zeros((64, 3, 224, 224), device=ex.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=ex.use_fp16):
            for _ in range(3):
                ex.model(data)
        torch.cuda.synchronize()

    def forward(ex, tensors):
        # 只读取当前调用栈持有的队列；不取出、重排或保留数据张量。
        frame = sys._getframe(1)
        try:
            while frame is not None and frame.f_code.co_name != "_ingest_tiles":
                frame = frame.f_back
            if frame is None:
                raise RuntimeError("未找到瓦片入口，拒绝生成误导性报告")
            ready = frame.f_locals["ready_q"]
        finally:
            del frame
        start = time.perf_counter()
        before = ready.qsize()
        result = original_forward(ex, tensors)
        end = time.perf_counter()
        rows.append({"batch": len(tensors), "ready_before": before,
                     "ready_after": ready.qsize(), "forward_s": end - start,
                     "between_forward_s": start - rows[-1]["end"] if rows else None,
                     "end": end})
        return result

    reports = []
    for label in ("reference", "observed"):
        with patch.object(ResNetExtractor, "__init__", init), patch.object(
                ResNetExtractor, "_forward", forward if label == "observed" else original_forward):
            reports.append(run_one("tiles", label, sample, str(work), False, True, 0,
                                   png_decoder="libdeflate"))
        print("RUN", label, reports[-1]["wall_s"], flush=True)
    equal = compare_tile_arrays(str(work), "reference", "observed")
    assert rows and equal
    summary = {"batches": len(rows), "tiles_forwarded": sum(r["batch"] for r in rows),
               "start_at_least_64": sum(r["ready_before"] >= 64 for r in rows),
               "end_at_least_64": sum(r["ready_after"] >= 64 for r in rows),
               "start_empty": sum(r["ready_before"] == 0 for r in rows),
               "end_empty": sum(r["ready_after"] == 0 for r in rows),
               "forward_s": sum(r["forward_s"] for r in rows),
               "between_forward_s": sum(r["between_forward_s"] or 0 for r in rows),
               "index_comparison_passed": equal,
               "caveat": "qsize is an instantaneous item count, may include the final sentinel; no GPU idle time or speedup inferred"}
    (work / "diagnostic.json").write_text(json.dumps(
        {"summary": summary, "runs": reports, "batches": rows}, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), "REPORT", work, flush=True)


if __name__ == "__main__":
    main()
