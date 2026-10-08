# -*- coding: utf-8 -*-
# ImageSearchTool · GPU 发射开销与解码竞争诊断
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""合成 PNG/JPEG 解码负载下比较 eager/graph；不修改生产模型或图库。"""
import io
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_search.config import Config
from hybrid_search.engine import HybridEngine
from hybrid_search.io_utils import decode_rgb


def main():
    import torch
    ex = HybridEngine(Config())._get_extractor()
    if ex.device != "cuda":
        raise RuntimeError("需要 CUDA")
    data = torch.rand((64, 3, 224, 224), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        for _ in range(3):
            ex.model(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph), torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        captured = ex.model(data)
    graph.replay()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        reference = ex.model(data)
    np.testing.assert_allclose(reference.float().cpu().numpy(), captured.float().cpu().numpy(), atol=1e-5, rtol=0)
    pixels = np.random.default_rng(10).integers(0, 256, (1024, 1024, 4), dtype=np.uint8)
    payloads = []
    for fmt in ("PNG", "JPEG"):
        buffer = io.BytesIO()
        image = Image.fromarray(pixels)
        if fmt == "JPEG":
            image = image.convert("RGB")
        image.save(buffer, format=fmt)
        payloads.append(buffer.getvalue())
    stop = threading.Event()

    def decode_loop(index):
        count = 0
        while not stop.is_set():
            assert decode_rgb(payloads[index % 2]) is not None
            count += 1
        return count

    def measure(label):
        results = []
        for mode in ("eager", "graph", "graph", "eager"):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(30):
                    if mode == "graph":
                        graph.replay()
                        result = captured
                    else:
                        result = ex.model(data)
                    result.float().cpu().numpy()
            results.append({"mode": mode, "seconds": time.perf_counter() - start})
        print(json.dumps({"load": label, "runs": results}), flush=True)

    measure("idle")
    with ThreadPoolExecutor(max_workers=20) as pool:
        jobs = [pool.submit(decode_loop, i) for i in range(20)]
        try:
            measure("20_png_jpeg_decoders")
        finally:
            stop.set()
        print("DECODED", sum(job.result() for job in jobs), flush=True)


if __name__ == "__main__":
    main()
