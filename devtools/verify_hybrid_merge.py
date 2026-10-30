# -*- coding: utf-8 -*-
# ImageSearchTool · 混合结果回归 / Hybrid result regression
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""No model, gallery writes or UI dependencies required."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_search.engine import Hit, Outcome
from hybrid_search.tile_index import merge_hybrid


def hit(score, kind, path='same', box=None):
    return Hit(1, path, score, .5, float('nan'), float('nan'), box, kind)


class MergeTests(unittest.TestCase):
    def test_both_regardless_of_winner_or_tie(self):
        for a, b in ((.9, .7), (.7, .9), (.9, .9)):
            full = Outcome('q', hits=[hit(a, 'full')])
            tiles = Outcome('q', hits=[hit(b, 'tile', box=(1, 2, 3, 4))])
            out = merge_hybrid(full, tiles, 'q', 10)
            self.assertEqual(len(out.hits), 1)
            self.assertEqual(out.hits[0].match_kind, 'both')
            self.assertEqual(out.hits[0].fine_score, max(a, b))
            self.assertEqual(out.hits[0].box, (1, 2, 3, 4) if b > a else None)
            self.assertEqual(full.hits[0].match_kind, 'full')

    def test_valid_zero_negative_and_nan(self):
        full = Outcome('q', hits=[hit(0., 'full', 'zero'), hit(-.5, 'full', 'negative'),
                                  hit(float('nan'), 'full', 'unknown')])
        out = merge_hybrid(full, Outcome('q'), 'q', 10)
        self.assertEqual([h.path for h in out.hits], ['zero', 'negative', 'unknown'])
        self.assertEqual([h.fine_score for h in out.hits[:2]], [0., -.5])
        self.assertTrue(math.isnan(out.hits[-1].fine_score))
        self.assertFalse(out.coarse_only)

    def test_total_includes_both_without_double_counting_stages(self):
        full = Outcome('q', times={'total': .2, 'stage': .1})
        tiles = Outcome('q', times={'total': .8, 'stage': .7})
        with patch('hybrid_search.tile_index.time.perf_counter', side_effect=[10., 10.05]):
            out = merge_hybrid(full, tiles, 'q', 10)
        self.assertAlmostEqual(out.times['total'], 1.05)
        self.assertEqual(out.times['整图·total'], .2)
        self.assertEqual(out.times['瓦片·total'], .8)
        self.assertEqual(full.times, {'total': .2, 'stage': .1})

    def test_path_aliases_deduplicate(self):
        out = merge_hybrid(Outcome('q', hits=[hit(.8, 'full', './same')]),
                           Outcome('q', hits=[hit(.9, 'tile', 'same')]), 'q', 10)
        self.assertEqual(len(out.hits), 1)
        self.assertEqual(out.hits[0].match_kind, 'both')

    def test_duplicate_one_branch_is_not_both(self):
        out = merge_hybrid(Outcome('q', hits=[hit(.8, 'full'), hit(.9, 'full')]),
                           Outcome('q'), 'q', 10)
        self.assertEqual(out.hits[0].match_kind, 'full')

    def test_all_coarse_and_counts(self):
        out = merge_hybrid(Outcome('q', hits=[hit(float('nan'), 'full')], coarse_kept=3),
                           Outcome('q', coarse_kept=5, self_excluded=True), 'q', 10)
        self.assertTrue(out.coarse_only)
        self.assertTrue(out.self_excluded)
        self.assertEqual(out.coarse_kept, 8)

    def test_topk_ranks_and_empty(self):
        out = merge_hybrid(Outcome('q', hits=[hit(.7, 'full', 'a'), hit(.9, 'full', 'b')]),
                           Outcome('q', hits=[hit(.8, 'tile', 'c')]), 'q', 2)
        self.assertEqual([(h.rank, h.path) for h in out.hits], [(1, 'b'), (2, 'c')])
        self.assertEqual(merge_hybrid(Outcome('q'), Outcome('q'), 'q', 10).hits, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
