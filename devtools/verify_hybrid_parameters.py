# -*- coding: utf-8 -*-
# ImageSearchTool · 混合参数与模型复用 / Hybrid parameters and model reuse
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_search import cli, tile_index as ti
from hybrid_search.config import Config
from hybrid_search.engine import Outcome, Hit
from hybrid_search.service import SearchService


def result(n, kind):
    return Outcome('query', hits=[Hit(i+1, f'{kind}_{i}', .9-i*.001, .5,
                                     float('nan'), float('nan'), match_kind=kind)
                                  for i in range(n)], times={'total': .1})


class ParameterTests(unittest.TestCase):
    def engines(self):
        full = SimpleNamespace(_extractor=None, open=lambda p: None)
        tiles = SimpleNamespace(_extractor=None, open=lambda p: None)
        def search(query, coarse_k, top_k):
            if full._extractor is None:
                full._extractor = object()  # model is initialized during the first query
            return result(top_k, 'full')
        full.search = search
        return full, tiles

    def test_service_requested_counts_and_config_default(self):
        for requested in (1, 7, 30, None):
            with self.subTest(top_k=requested):
                svc = SearchService(cfg=Config(top_k=13), capture_log=False)
                full, tiles = self.engines()
                calls = []
                def tile_search(engine, query, top_k, coarse_k):
                    self.assertIs(engine._extractor, full._extractor)
                    self.assertIsNotNone(engine._extractor)
                    calls.append(top_k)
                    return result(top_k, 'tile')
                with patch.object(svc, 'engine_for', side_effect=[(full, False), (tiles, False)]), \
                     patch.object(ti, 'search_tiles_tiled', side_effect=tile_search):
                    out = svc.search('test', 'query', 'hybrid', prefix='unused', top_k=requested)
                wanted = 13 if requested is None else requested
                self.assertIsNotNone(out)
                self.assertEqual(len(out['hits']), wanted)
                self.assertEqual(calls, [wanted])

    def test_cli_cold_model_shared(self):
        full, tiles = self.engines()
        args = cli.build_parser().parse_args(['search', 'query', '--mode', 'hybrid', '--top-k', '7'])
        def tile_search(engine, args, prefix):
            self.assertIs(engine._extractor, full._extractor)
            self.assertIsNotNone(engine._extractor)
            return result(args.top_k, 'tile')
        with patch.object(cli.os.path, 'exists', return_value=True), \
             patch.object(cli, 'HybridEngine', side_effect=[full, tiles]), \
             patch.object(cli, '_search_tiles_mode', side_effect=tile_search), \
             patch.object(cli, '_print_outcome') as output, patch('builtins.print'):
            self.assertEqual(cli.cmd_search(Config(), args), 0)
        self.assertEqual(len(output.call_args.args[0].hits), 7)

    def test_no_full_model_keeps_existing_tile_model(self):
        svc = SearchService(cfg=Config(), capture_log=False)
        full, tiles = self.engines()
        model = object()
        tiles._extractor = model
        full.search = lambda *args, **kwargs: Outcome('query')
        def tile_search(engine, *args, **kwargs):
            self.assertIs(engine._extractor, model)
            return result(1, 'tile')
        with patch.object(svc, 'engine_for', side_effect=[(full, True), (tiles, True)]), \
             patch.object(ti, 'search_tiles_tiled', side_effect=tile_search):
            self.assertTrue(svc.search('test', 'query', 'hybrid', prefix='unused')['hits'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
