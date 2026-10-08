# -*- coding: utf-8 -*-
# ImageSearchTool · PNG 线程资源生命周期回归验证
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""合成数据验证：线程退出/分配失败归还预算、原生句柄释放、缓存编解码。

python -E -B devtools/verify_png_resources.py
不读写用户图库，不生成索引，不依赖 GPU。
"""
import ctypes
import gc
import io
import os
import sys
import threading
import unittest
import zlib
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search import png_fast as pf  # noqa: E402


def in_threads(fn, count=1):
    errors = []

    def run():
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        if thread.is_alive():
            raise AssertionError("worker did not terminate")
    gc.collect()
    if errors:
        raise errors[0]


class ResourcesTest(unittest.TestCase):
    def setUp(self):
        self.before = pf.scratch_mb()
        self.cap, self.total = pf._SCRATCH_CAP, pf._SCRATCH_TOTAL

    def tearDown(self):
        pf.set_scratch_cap_mb(self.cap / 2**20)
        pf.set_scratch_budget_mb(self.total / 2**20)

    def test_repeated_pools_return_budget(self):
        pf.set_scratch_cap_mb(1)
        pf.set_scratch_budget_mb(self.before + 2)
        for _ in range(5):
            def work():
                a = pf._scratch_acquire(1 << 20)
                self.assertIs(a, pf._scratch_acquire(1 << 20))
            in_threads(work)
            self.assertEqual(pf.scratch_mb(), self.before)

    def test_allocation_failure_returns_reservation(self):
        pf.set_scratch_cap_mb(2)
        pf.set_scratch_budget_mb(self.before + 4)

        def work():
            buf = pf._scratch_acquire(1024)
            used = pf.scratch_mb()
            with patch.object(pf.np, "empty", side_effect=MemoryError):
                with self.assertRaises(MemoryError):
                    pf._scratch_acquire(2 << 20)
            self.assertEqual(pf.scratch_mb(), used)
            self.assertIs(buf, pf._scratch_acquire(1024))
        in_threads(work)
        self.assertEqual(pf.scratch_mb(), self.before)

    def test_multislot_growth_and_uncached_buffer(self):
        pf.set_scratch_cap_mb(1)
        pf.set_scratch_budget_mb(self.before + 1)

        def work():
            a = pf._scratch_acquire(128 << 10)
            b = pf._scratch_acquire(256 << 10)
            self.assertIsNot(a, b)
            c = pf._scratch_acquire(256 << 10, "cxbuf")
            self.assertIs(c, pf._scratch_acquire(256 << 10, "cxbuf"))
            pf._scratch_acquire(2 << 20)
            self.assertEqual(pf.scratch_mb(), self.before + 0.5)
            self.assertIs(b, pf._scratch_acquire(128 << 10))
        in_threads(work)
        self.assertEqual(pf.scratch_mb(), self.before)

    def test_parallel_budget_is_bounded_and_reusable(self):
        pf.set_scratch_cap_mb(1)
        pf.set_scratch_budget_mb(self.before + 2)
        barrier = threading.Barrier(8, timeout=10)

        def work():
            pf._scratch_acquire(1 << 20)
            barrier.wait()
            self.assertLessEqual(pf.scratch_mb(), self.before + 2)
            barrier.wait()
        in_threads(work, 8)
        self.assertEqual(pf.scratch_mb(), self.before)

    def test_decompressor_freed_on_thread_exit(self):
        lib = Mock()
        lib.libdeflate_alloc_decompressor.return_value = 1234
        with patch.object(pf, "_load_libdeflate", return_value=(lib, "fake")):
            def work():
                self.assertIs(pf._ensure_handle(), pf._ensure_handle())
            in_threads(work, 4)
        self.assertEqual(lib.libdeflate_alloc_decompressor.call_count, 4)
        self.assertEqual(lib.libdeflate_free_decompressor.call_count, 4)

    def test_compression_roundtrip_levels_and_cleanup(self):
        pair = pf._load_libdeflate()
        if not pair:
            self.skipTest("libdeflate DLL unavailable")
        lib = pair[0]
        # Check the pointer signature BEFORE invoking native allocation.
        self.assertIs(lib.libdeflate_alloc_compressor.restype, ctypes.c_void_p)
        self.assertIsNotNone(lib.libdeflate_zlib_compress.argtypes)
        payload = (bytes(range(256)) + b"abc" * 100) * 256
        free = lib.libdeflate_free_compressor
        with patch.object(lib, "libdeflate_free_compressor", wraps=free) as freed:
            def work():
                for level in (1, 6, 12, 1):
                    blob = pf.ldf_compress(payload, level)
                    self.assertIsNotNone(blob)
                    self.assertEqual(zlib.decompress(blob), payload)
                    out = np.empty(len(payload), dtype=np.uint8)
                    self.assertTrue(pf.ldf_decompress_into(blob, out))
                    self.assertEqual(out.tobytes(), payload)
                    # Compare with a fresh native compressor at the requested level.
                    handle = lib.libdeflate_alloc_compressor(level)
                    try:
                        n = lib.libdeflate_zlib_compress_bound(handle, len(payload))
                        target = ctypes.create_string_buffer(n)
                        got = lib.libdeflate_zlib_compress(
                            handle, payload, len(payload), target, n)
                        self.assertEqual(blob, target.raw[:got])
                    finally:
                        free(handle)
                self.assertIsNone(pf.ldf_compress(payload, 13))
            in_threads(work)
        self.assertEqual(freed.call_count, 3)

    def test_rgba_multithread_exact_pixels(self):
        if not pf.available():
            self.skipTest("native PNG dependencies unavailable")
        pixels = np.random.default_rng(7).integers(0, 256, (100, 129, 4), dtype=np.uint8)
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        data = output.getvalue()

        def work():
            for _ in range(8):
                np.testing.assert_array_equal(pf.decode_rgb(data), pixels[:, :, :3])
        in_threads(work, 18)
        self.assertEqual(pf.scratch_mb(), self.before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
