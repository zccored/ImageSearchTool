# -*- coding: utf-8 -*-
# ImageSearchTool · 瓦片任务生命周期与结束竞争回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""故障注入在隔离子进程内执行，超时仅终止本脚本创建的子进程。不写索引。"""
import os
import queue
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search import tile_index as ti  # noqa: E402
from hybrid_search.coarse import CoarseRecord  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.engine import HybridEngine  # noqa: E402


def run_case(case):
    cfg = Config(tile_decode_slots=1, silence_png_warnings=False)
    engine = HybridEngine(cfg)
    extractor = SimpleNamespace(decode_workers=2, transform=None,
                                _forward=lambda batch: np.ones((len(batch), 4), np.float32))
    engine._get_extractor = lambda: extractor
    seen = []

    def decode(path, *args, **kwargs):
        seen.append(path)
        if case == "image_failure" and path == "bad":
            raise ValueError("injected image failure")
        if case == "all_images_fail":
            raise ValueError("injected image failure")
        return [(np.zeros((4, 4, 3), np.uint8), (0, 0, 4, 4), b"data", True)]

    def feature(path, *args, **kwargs):
        if case == "tile_failure" and path == "bad":
            raise ValueError("injected tile failure")
        rec = CoarseRecord(path=path, md5=path, hu=np.zeros(7, np.float32),
                           fp=np.zeros(512, np.uint8), box=(0, 0, 4, 4))
        return object(), rec

    class DelayedQueue(queue.Queue):
        def put(self, item, *args, **kwargs):
            # 放大“完成计数归零，但结果尚未进入 ready_q”的竞争窗口。
            if case == "last_result_delay" and self.maxsize and item is not None:
                time.sleep(0.12)
            return super().put(item, *args, **kwargs)

    files = Mock(coarse_path=os.path.abspath("unused.coarse.npz"))
    paths = ["good"] if case == "last_result_delay" else ["bad", "good"]
    expected = 0 if case == "all_images_fail" else 1
    with patch.object(ti, "IndexFiles", return_value=files), \
         patch.object(ti, "_decode_to_crops", side_effect=decode), \
         patch.object(ti, "_feature_one_tile", side_effect=feature), \
         patch.object(ti.queue, "Queue", DelayedQueue):
        count = ti._ingest_tiles(engine, "unused", paths)
    assert count == expected, (case, count, expected)
    assert seen == paths, (case, seen)
    assert engine.coarse.size == expected
    if expected:
        assert engine.coarse.paths == ["good"]
        assert engine._fine_feats.shape == (expected, 4)
    print("PASS", case, flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        run_case(sys.argv[2])
    else:
        failed = []
        for case in ("image_failure", "all_images_fail", "tile_failure", "last_result_delay"):
            try:
                result = subprocess.run(
                    [sys.executable, "-E", "-B", "-X", "utf8", __file__, "--case", case],
                    capture_output=True, text=True, encoding="utf-8", timeout=8)
                print(result.stdout, end="")
                if result.returncode:
                    print(result.stderr)
                    failed.append(case)
            except subprocess.TimeoutExpired:
                print("FAIL timeout:", case)
                failed.append(case)
        print("FAILED", failed)
        sys.exit(bool(failed))
