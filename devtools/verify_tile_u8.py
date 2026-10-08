# -*- coding: utf-8 -*-
# ImageSearchTool · uint8 瓦片输入协议与数值一致性回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""真实模型与合成像素，测试数值契约/CPU 回退，不读写用户图库。"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search.config import Config
from hybrid_search.fine import ResNetExtractor


class TileU8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("需要 CUDA 检查字节交接协议")
        cls.ex = ResNetExtractor(Config(device="cuda"))

    def test_all_byte_values_scale_exact(self):
        torch = self.torch
        values = torch.arange(256, dtype=torch.uint8)
        expected = values.float().div_(255)
        actual = self.ex._tile_scale_lut[values.cuda().long()].cpu()
        self.assertTrue(torch.equal(expected, actual))

    def test_preprocessing_keeps_pixels_and_dimensions(self):
        torch = self.torch
        rng = np.random.default_rng(31)
        for h, w in ((91, 128), (512, 512), (901, 1200)):
            im = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            reference = self.ex.transform(im)
            raw = self.ex.tile_transform(im)
            self.assertEqual(raw.dtype, torch.uint8)
            self.assertEqual(tuple(raw.shape), tuple(reference.shape))
            restored = self.ex._tile_scale_lut[raw.cuda().long()].cpu()
            self.assertTrue(torch.equal(reference, restored))
            self.assertEqual(raw.numel() * raw.element_size() * 4,
                             reference.numel() * reference.element_size())

    def test_model_inputs_and_features_match(self):
        torch = self.torch
        rng = np.random.default_rng(5)
        images = [Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)) for _ in range(4)]
        captured = []
        handle = self.ex.model.register_forward_pre_hook(
            lambda model, args: captured.append(args[0].detach().cpu().clone()))
        try:
            reference = self.ex._forward([self.ex.transform(im) for im in images])
            actual = self.ex._forward([self.ex.tile_transform(im) for im in images])
        finally:
            handle.remove()
        self.assertTrue(torch.equal(captured[0], captured[1]))
        np.testing.assert_allclose(reference, actual, rtol=0, atol=1e-5)

    def test_cpu_and_cpu_normalization_keep_float_protocol(self):
        torch = self.torch
        image = Image.fromarray(np.full((512, 512, 3), 127, np.uint8))
        for cfg in (Config(device="cpu"), Config(device="cuda", norm_on_gpu=False)):
            ex = ResNetExtractor(cfg)
            self.assertIs(ex.tile_transform, ex.transform)
            self.assertIsNone(ex._tile_scale_lut)
            tensor = ex.tile_transform(image)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(ex._forward([tensor]).shape, (1, ex.feature_dim))


if __name__ == "__main__":
    unittest.main()
