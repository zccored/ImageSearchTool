# -*- coding: utf-8 -*-
# ImageSearchTool · 瓦片前向后的批次计时回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""用虚拟时钟放大前向耗时；只验证内存流水线，索引落盘 mock。"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search import tile_index as ti
from hybrid_search.config import Config
from hybrid_search.coarse import CoarseRecord
from hybrid_search.engine import HybridEngine


def main():
    engine = HybridEngine(Config(tile_decode_slots=1, silence_png_warnings=False))
    clock = [1.0]
    batches = []

    def forward(tensors):
        batches.append(len(tensors))
        clock[0] += 1.0  # 一秒前向；不应计入下一批的收集等待时间
        return np.ones((len(tensors), 4), np.float32)

    def feature(path, crop, box, *args):
        rec = CoarseRecord(path=path, md5=str(box), box=box,
                           hu=np.ones(7, np.float32), fp=np.ones(512, np.uint8))
        return object(), rec

    ex = SimpleNamespace(decode_workers=2, transform=None, _forward=forward)
    engine._get_extractor = lambda: ex
    crops = [(None, (i, 0, i + 4, 4), None, i == 0) for i in range(128)]
    files = Mock(coarse_path=os.path.abspath("unused.coarse.npz"))
    with patch.object(ti, "IndexFiles", return_value=files), \
         patch.object(ti, "_decode_to_crops", return_value=crops), \
         patch.object(ti, "_feature_one_tile", side_effect=feature), \
         patch.object(ti.time, "monotonic", side_effect=lambda: clock[0]):
        assert ti._ingest_tiles(engine, "unused", ["image"]) == 128
    assert batches == [64, 64], batches
    assert engine.coarse.size == len(engine._fine_feats) == 128
    print("PASS batches", batches)


if __name__ == "__main__":
    main()
