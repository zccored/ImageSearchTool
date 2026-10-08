# -*- coding: utf-8 -*-
# ImageSearchTool · L2 缓存替换、淘汰与并发预算回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""仅内存测试，无图库或磁盘缓存写入。python -E -B devtools/verify_cache_lru.py"""
import os
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search.prep_cache import PrepCache  # noqa: E402

SIZE = 602 * 1024


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.torch_presence = patch.dict(sys.modules, {"torch": object()})
        self.torch_presence.start()
        self.addCleanup(self.torch_presence.stop)

    def test_replacement_does_not_evict_other_key(self):
        cache = PrepCache("unused", mem_bytes=2 * SIZE)
        cache._put_mem("a", (1, 1))
        cache._put_mem("b", (2, 2))
        for i in range(100):
            cache._put_mem("b", (i, i))
        self.assertEqual(list(cache._mem), ["a", "b"])
        self.assertEqual(cache._mem_used, 2 * SIZE)
        self.assertEqual(cache._mem["b"], (99, 99))

    def test_replace_refreshes_lru(self):
        cache = PrepCache("unused", mem_bytes=2 * SIZE)
        for key in ("a", "b", "a", "c"):
            cache._put_mem(key, (key, key))
        self.assertEqual(list(cache._mem), ["a", "c"])
        self.assertEqual(cache._mem_used, 2 * SIZE)

    def test_zero_and_too_small_budget(self):
        for budget in (0, SIZE - 1):
            cache = PrepCache("unused", mem_bytes=budget)
            cache._put_mem("a", (1, 1))
            self.assertFalse(cache._mem)
            self.assertEqual(cache._mem_used, 0)

    def test_parallel_updates_keep_budget_and_entries(self):
        cache = PrepCache("unused", mem_bytes=2 * SIZE)
        cache._put_mem("a", (1, 1))
        errors = []

        def work():
            try:
                for i in range(500):
                    cache._put_mem("b", (i, i))
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=work) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(set(cache._mem), {"a", "b"})
        self.assertEqual(cache._mem_used, 2 * SIZE)

    def test_disk_get_replacement_remains_consistent(self):
        cache = PrepCache("unused", mem_bytes=2 * SIZE)
        cache._put_mem("a", (1, 1))
        cache._put_mem("b", (2, 2))
        with patch.object(cache, "key_for", return_value="a"):
            self.assertEqual(cache.get("not-a-real-file"), (1, 1))
        cache._put_mem("a", (3, 3))
        cache._put_mem("c", (4, 4))
        self.assertEqual(list(cache._mem), ["a", "c"])
        self.assertEqual(cache._mem_used, 2 * SIZE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
