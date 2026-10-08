# -*- coding: utf-8 -*-
# ImageSearchTool · 融合批次失败回滚回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""内存故障注入；解码与索引写入均 mock，不读写图库。"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search.engine import HybridEngine
from hybrid_search.coarse import CoarseRecord
from hybrid_search.config import Config
from hybrid_search import tile_index as ti


def record(path, tile=False):
    return CoarseRecord(path, path, np.ones(7, np.float32),
                        np.ones(512, np.uint8), (0, 0, 4, 4) if tile else None)


class RollbackTests(unittest.TestCase):
    def test_partial_coarse_failure_preserves_pending_parts(self):
        engine = HybridEngine(Config(silence_png_warnings=False))
        coarse = engine.coarse
        coarse.add_results(["old"], [record("old")])
        pending = coarse._hu_parts[0]
        broken = record("broken")
        broken.hu = np.ones(8, np.float32)
        with self.assertRaises(ValueError):
            with coarse.append_transaction():
                coarse.add_results(["good", "broken"], [record("good"), broken])
        self.assertEqual(coarse.paths, ["old"])
        self.assertEqual(coarse.md5s, ["old"])
        self.assertEqual(coarse.boxes, [None])
        self.assertEqual(coarse._md5_set, {"old"})
        self.assertIs(coarse._hu_parts[0], pending)
        self.assertEqual(len(coarse._fp_parts), 1)

    def test_successful_transaction_keeps_dedup_and_empty_digest(self):
        engine = HybridEngine(Config(silence_png_warnings=False))
        coarse = engine.coarse
        old = record("old")
        old.md5 = ""
        coarse.add_results(["old"], [old])
        with self.assertRaises(RuntimeError):
            with coarse.append_transaction():
                coarse.add_results(["bad"], [record("bad")])
                raise RuntimeError("injected")
        self.assertEqual(coarse._md5_set, {""})
        with coarse.append_transaction():
            mask = coarse.add_results(["old", "new", "new"],
                                      [old, record("new"), record("new")])
        self.assertEqual(mask, [False, True, False])
        self.assertEqual(coarse.paths, ["old", "new"])

    def check_whole(self, failure):
        engine = HybridEngine(Config(silence_png_warnings=False))
        engine.coarse.add_results(["old"], [record("old")])
        engine.coarse.finalize()
        old_hu = engine.coarse._hu
        outcomes = iter([failure, np.ones((1, 4), np.float32)])

        def forward(batch):
            value = next(outcomes)
            if isinstance(value, BaseException):
                raise value
            return value

        def stream(paths, prep, progress, on_batch):
            results = []
            for name in ("bad", "good"):
                try:
                    result = on_batch([name], [object()], [record(name)])
                    if result[0]:
                        results.append(result)
                except Exception:
                    pass  # 与真实 stream_decode 的单批失败跳过策略相同
            return results

        ex = SimpleNamespace(_forward=forward, stream_decode=stream)
        engine._get_extractor = lambda: ex
        engine._make_fused_prep = lambda *args, **kwargs: None
        results = engine._fused_ingest(["bad", "good"])
        self.assertEqual(engine.coarse.paths, ["old", "good"])
        self.assertEqual([p for paths, _ in results for p in paths], ["good"])
        self.assertIs(engine.coarse._hu, old_hu)
        self.assertNotIn("bad", engine.coarse._md5_set)
        self.assertNotIn(os.path.normcase(os.path.abspath("bad")), engine.coarse._path_set)
        # 失败条目不应被去重集合永久占用，下一次可重试。
        self.assertEqual(engine.coarse.add_results(["bad"], [record("bad")]), [True])
        engine.coarse.finalize()
        self.assertEqual(engine.coarse.hu.shape, (3, 7))
        self.assertEqual(engine.coarse.fp.shape, (3, 512))

    def test_whole_exception(self):
        self.check_whole(RuntimeError("injected forward failure"))

    def test_whole_none(self):
        self.check_whole(None)

    def test_whole_wrong_rows(self):
        self.check_whole(np.ones((2, 4), np.float32))

    def test_whole_wrong_rank(self):
        self.check_whole(np.ones(4, np.float32))

    def check_tiles(self, failure):
        engine = HybridEngine(Config(tile_decode_slots=1, silence_png_warnings=False))
        engine.coarse.add_results(["old"], [record("old", True)])
        engine.coarse.finalize()
        engine._keep_fine(np.ones((1, 4), np.float32))
        outcomes = iter([failure, np.ones((64, 4), np.float32)])

        def forward(batch):
            value = next(outcomes)
            if isinstance(value, BaseException):
                raise value
            return value

        ex = SimpleNamespace(decode_workers=2, transform=None, _forward=forward)
        engine._get_extractor = lambda: ex
        files = Mock(coarse_path=os.path.abspath("unused.coarse.npz"))
        counters = {}

        def feature(path, *args, **kwargs):
            index = counters.get(path, 0)
            counters[path] = index + 1
            rec = record(path, True)
            rec.md5 = f"{path}:{index}"
            rec.box = (index, 0, index + 4, 4)
            return object(), rec

        with patch.object(ti, "IndexFiles", return_value=files), \
             patch.object(ti, "_decode_to_crops", return_value=[(None, None, None, True)] * 64), \
             patch.object(ti, "_feature_one_tile", side_effect=feature):
            count = ti._ingest_tiles(engine, "unused", ["first", "second"])
        self.assertEqual(count, 64)
        self.assertEqual(engine.coarse.size, 65)
        self.assertEqual(engine._fine_feats.shape, (65, 4))
        saved_paths = files.save_fine.call_args.args[0]
        self.assertEqual(saved_paths, engine.coarse.paths)
        self.assertEqual(engine.coarse.hu.shape, (65, 7))
        self.assertEqual(len(engine.coarse._md5_set), 65)

    def test_tiles_exception(self):
        self.check_tiles(RuntimeError("injected forward failure"))

    def test_tiles_none(self):
        self.check_tiles(None)

    def test_tiles_wrong_rows(self):
        self.check_tiles(np.ones((2, 4), np.float32))

    def test_tiles_all_fail_preserve_existing_index(self):
        engine = HybridEngine(Config(tile_decode_slots=1, silence_png_warnings=False))
        engine.coarse.add_results(["old"], [record("old", True)])
        engine.coarse.finalize()
        old_feats = np.ones((1, 4), np.float32)
        engine._keep_fine(old_feats)
        ex = SimpleNamespace(decode_workers=2, transform=None,
                             _forward=lambda batch: None)
        engine._get_extractor = lambda: ex
        files = Mock(coarse_path=os.path.abspath("unused.coarse.npz"))
        with patch.object(ti, "IndexFiles", return_value=files), \
             patch.object(ti, "_decode_to_crops", return_value=[(None, None, None, True)]), \
             patch.object(ti, "_feature_one_tile", side_effect=lambda path, *a, **k: (object(), record(path, True))):
            self.assertEqual(ti._ingest_tiles(engine, "unused", ["bad"]), 0)
        self.assertEqual(engine.coarse.paths, ["old"])
        self.assertIs(engine._fine_feats, old_feats)
        self.assertEqual(files.save_coarse.call_args.args[0], ["old"])
        files.save_fine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
