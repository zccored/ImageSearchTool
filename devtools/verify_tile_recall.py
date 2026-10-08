# -*- coding: utf-8 -*-
# ImageSearchTool · 瓦片召回与查询几何回归 / Tile recall regression
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""Deterministic in-memory tests: no model downloads, no gallery/index writes."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_search import tile_index as ti
from hybrid_search.config import Config


def engine_for(feats, paths=None):
    n = len(feats) if feats is not None else 3
    paths = paths or [f'image_{i}' for i in range(n)]
    coarse = SimpleNamespace(size=n, paths=paths, fp=np.zeros((n, 1), np.uint8),
                             n_bytes=1, has_boxes=lambda: True,
                             box_at=lambda i: (i, 0, i + 1, 1),
                             query_record=Mock(return_value=SimpleNamespace(fp=np.zeros(1, np.uint8))))
    return SimpleNamespace(cfg=Config(), coarse=coarse, _fine_feats=feats, meta={},
                           _query_fine=Mock(return_value=np.array([1., 0.], np.float32)),
                           _get_extractor=Mock(return_value=SimpleNamespace(device='cpu')))


class RecallTests(unittest.TestCase):
    def test_late_rows_and_opposite_fingerprint(self):
        feats = np.tile([.8, .6], (900, 1)).astype(np.float32)
        feats[-1] = [1, 0]
        eng = engine_for(feats)
        eng.coarse.fp[-1] = 255  # 旧 0.62 阈值会误删；LSH 桶 cap 也会丢尾部。
        for alias in ('exact', 'lsh', 'coarse'):
            with patch.object(ti, '_lsh_for', side_effect=AssertionError('must not prefilter')):
                out = ti.search_tiles(eng, 'query', top_k=5, coarse_k=1, method=alias)
            self.assertEqual(out.hits[0].path, 'image_899')
            self.assertEqual(out.hits[0].box, (899, 0, 900, 1))
            self.assertEqual(len(out.hits), 5)
        self.assertEqual(eng._query_fine.call_count, 3)
        eng.coarse.query_record.assert_not_called()

    def test_row_permutation(self):
        feats = np.array([[.7, .7], [.8, .2], [.2, .8], [1, 0]], np.float32)
        paths = ['a', 'b', 'c', 'target']
        expected = [h.path for h in ti.search_tiles(engine_for(feats, paths), 'q').hits]
        order = [3, 1, 0, 2]
        got = ti.search_tiles(engine_for(feats[order], [paths[i] for i in order]), 'q')
        self.assertEqual([h.path for h in got.hits], expected)

    def test_self_exclusion_and_dedup_fill_topk(self):
        eng = engine_for(np.array([[1, 0], [.9, .1], [.8, .2], [.7, .3]], np.float32),
                         ['query', 'a', 'a', 'b'])
        out = ti.search_tiles(eng, 'query', top_k=2, coarse_k=1)
        self.assertTrue(out.self_excluded)
        self.assertEqual([(h.rank, h.path) for h in out.hits], [(1, 'a'), (2, 'b')])
        self.assertEqual(out.hits[0].box, (1, 0, 2, 1))

    def test_empty_avoids_model(self):
        eng = engine_for(np.zeros((0, 2), np.float32))
        self.assertEqual(ti.search_tiles(eng, 'q').hits, [])
        eng._query_fine.assert_not_called()

    def test_query_failure_is_error(self):
        eng = engine_for(np.ones((2, 2), np.float32))
        eng._query_fine.return_value = None
        with self.assertRaisesRegex(RuntimeError, '提取失败'):
            ti.search_tiles(eng, 'q')

    def test_missing_fine_uses_fingerprint_without_model(self):
        eng = engine_for(None)
        eng.coarse.fp[:, 0] = [255, 0, 15]
        out = ti.search_tiles_tiled(eng, 'q')
        self.assertTrue(out.coarse_only)
        self.assertEqual(out.hits[0].path, 'image_1')
        eng._query_fine.assert_not_called()
        eng._get_extractor.assert_not_called()

    def test_no_usable_features_is_error(self):
        eng = engine_for(None)
        eng.coarse.fp = None
        with self.assertRaisesRegex(RuntimeError, '无指纹'):
            ti.search_tiles(eng, 'q')

    def test_blocked_equals_dense_and_is_readonly(self):
        rng = np.random.default_rng(3)
        feats = rng.normal(size=(8201, 17)).astype(np.float16)
        queries = rng.normal(size=(65, 17)).astype(np.float32)
        before = feats.copy()
        feats.flags.writeable = False
        r = feats.astype(np.float32)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        expected = (r @ q.T).max(axis=1)
        seen = []
        original = ti._matmul_scores
        def score(rows, query, gpu):
            seen.append((len(rows), query.shape[1]))
            return original(rows, query, gpu)
        with patch.object(ti, '_matmul_scores', side_effect=score):
            got = ti._exact_tile_scores(feats, queries, False)
        np.testing.assert_allclose(got, expected, atol=2e-6)
        np.testing.assert_array_equal(feats, before)
        self.assertEqual(len(seen), 9)
        self.assertTrue(all(r <= 4096 and c <= 32 for r, c in seen))

    def test_later_query_blocks_not_truncated(self):
        feats = np.array([[1, 0], [.8, .2], [0, 1]], np.float32)
        queries = np.array([[1, 0]] * 64 + [[0, 1]], np.float32)
        got = ti._exact_tile_scores(feats, queries, False)
        self.assertAlmostEqual(float(got[-1]), 1.)

    def test_gpu_failure_falls_back(self):
        import torch
        rows = np.array([[1, 0], [0, 1]], np.float32)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch, 'from_numpy', side_effect=RuntimeError('injected CUDA failure')):
            got = ti._exact_tile_scores(rows, np.array([1, 0], np.float32), True)
        np.testing.assert_array_equal(got, [1, 0])

    def test_invalid_features_fail_loudly(self):
        good = np.ones((2, 2), np.float32)
        for feats, q in ((good, [0, 0]), (good, [np.nan, 1]),
                         (good, [1, 2, 3]), (np.array([[np.inf, 1]]), [1, 0])):
            with self.assertRaises(RuntimeError):
                ti._exact_tile_scores(feats, np.asarray(q), False)
        eng = engine_for(good)
        eng.coarse.size = 3
        with self.assertRaisesRegex(RuntimeError, '行数不一致'):
            ti.search_tiles(eng, 'q')

    def test_invalid_arguments(self):
        eng = engine_for(np.ones((2, 2), np.float32))
        for kw in ({'top_k': 0}, {'coarse_k': -1}, {'method': 'typo'}):
            with self.assertRaises(ValueError):
                ti.search_tiles(eng, 'q', **kw)

    def test_auto_tile_false_is_single(self):
        eng = engine_for(np.ones((2, 2), np.float32))
        with patch.object(ti, 'read_bytes', side_effect=AssertionError('must use single')):
            self.assertTrue(ti.search_tiles_tiled(eng, 'q', auto_tile=False).hits)

    def test_scaled_query_matches_build_crops_and_meta(self):
        yy, xx = np.indices((1000, 1700))
        rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=2).astype(np.uint8)
        cfg = Config(dedup=False)
        with patch.object(ti, 'read_bytes', return_value=b'fake'), \
             patch.object(ti, 'decode_rgb', return_value=rgb):
            expected = ti._decode_to_crops('q', cfg, 256, .5, 500, 640)
        eng = engine_for(np.ones((2, 2), np.float32))
        eng.meta = {'tiles': {'tile': 256, 'overlap': .5, 'min_side': 500, 'pre_max': 640}}
        seen = []
        def transform(image):
            seen.append(np.array(image))
            return image
        ex = SimpleNamespace(device='cpu', transform=transform,
                             _forward=lambda batch: np.ones((1, 2), np.float32))
        eng._get_extractor.return_value = ex
        with patch.object(ti, 'read_bytes', return_value=b'fake'), \
             patch.object(ti, 'decode_rgb', return_value=rgb):
            out = ti.search_tiles_tiled(eng, 'q')
        self.assertTrue(out.hits)
        self.assertEqual(len(seen), len(expected))
        for got, (want, *_rest) in zip(seen, expected):
            np.testing.assert_array_equal(got, want)
        eng._query_fine.assert_not_called()

    def test_query_decode_and_partial_forward_failure(self):
        eng = engine_for(np.ones((2, 2), np.float32))
        with patch.object(ti, 'read_bytes', return_value=None):
            with self.assertRaisesRegex(RuntimeError, '读取或解码'):
                ti.search_tiles_tiled(eng, 'q')
        ex = SimpleNamespace(device='cpu', transform=lambda x: x,
                             _forward=Mock(side_effect=[np.ones((1, 2), np.float32), None]))
        eng._get_extractor.return_value = ex
        with patch.object(ti, 'read_bytes', return_value=b'fake'), \
             patch.object(ti, 'decode_rgb', return_value=np.zeros((1000, 1000, 3), np.uint8)):
            with self.assertRaisesRegex(RuntimeError, '不完整'):
                ti.search_tiles_tiled(eng, 'q')

    def test_service_error_then_recovery(self):
        from hybrid_search.service import SearchService
        eng = engine_for(np.ones((2, 2), np.float32))
        svc = SearchService(cfg=Config(), capture_log=False)
        events = []
        svc.subscribe(events.append)
        with patch.object(svc, 'engine_for', return_value=(eng, True)), \
             patch.object(ti, 'read_bytes', return_value=None):
            self.assertIsNone(svc.search('bad', 'q', 'tiles', prefix='unused'))
        self.assertTrue(any(e['event'] == 'task_error' for e in events))
        with patch.object(svc, 'engine_for', return_value=(eng, True)), \
             patch.object(ti, 'read_bytes', return_value=b'fake'), \
             patch.object(ti, 'decode_rgb', return_value=np.zeros((10, 10, 3), np.uint8)):
            result = svc.search('good', 'q', 'tiles', prefix='unused')
        self.assertTrue(result['hits'])
        self.assertTrue(any(e['event'] == 'task_done' and e['task_id'] == 'good' for e in events))

    def test_cli_default_and_legacy_aliases(self):
        from hybrid_search import cli
        from hybrid_search.engine import Outcome
        for extra in ([], ['--cand', 'lsh'], ['--cand', 'coarse'], ['--cand', 'exact']):
            args = cli.build_parser().parse_args(['search', 'query', '--mode', 'tiles'] + extra)
            with patch.object(ti, 'search_tiles_tiled', return_value=Outcome('query')) as search:
                cli._search_tiles_mode(object(), args, 'unused')
            self.assertIn(search.call_args.kwargs['method'], ('exact', 'lsh', 'coarse'))

    def test_cli_tile_prefix_follows_selected_gallery(self):
        from hybrid_search import cli
        from hybrid_search.engine import Outcome
        for extra in ([], ['--tiles-prefix', 'explicit_tiles']):
            args = cli.build_parser().parse_args(['search', 'query', '--mode', 'tiles',
                                                  '--prefix', 'chosen/gallery'] + extra)
            eng = Mock()
            expected = os.path.abspath('explicit_tiles' if extra else ti.tiles_prefix_of(args.prefix))
            with patch.object(cli.os.path, 'exists', return_value=True), \
                 patch.object(cli, 'HybridEngine', return_value=eng), \
                 patch.object(cli, '_search_tiles_mode', return_value=Outcome('query')), \
                 patch.object(cli, '_print_tiles_outcome'):
                self.assertEqual(cli.cmd_search(Config(), args), 0)
            eng.open.assert_called_once_with(expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
