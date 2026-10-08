# -*- coding: utf-8 -*-
# ImageSearchTool · OpenCV 进程初始化回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_search import runtime
from hybrid_search.config import Config
from hybrid_search.engine import HybridEngine
from hybrid_search.service import SearchService, config_schema, cli_hint
from hybrid_search import cli


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        # 只在 mock OpenCV 的单元测试中重置状态；生产不提供重置接口。
        for name in ("_opencv_requested", "_opencv_effective"):
            p = patch.object(runtime, name, None)
            p.start()
            self.addCleanup(p.stop)
        self.setter = patch("cv2.setNumThreads").start()
        self.getter = patch("cv2.getNumThreads", return_value=1).start()
        self.addCleanup(patch.stopall)

    def test_once_across_concurrent_initializers(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(runtime.configure_opencv_threads, [1]*32)), [1]*32)
        self.setter.assert_called_once_with(1)

    def test_opt_out_never_mutates(self):
        self.getter.return_value = 7
        self.assertEqual(runtime.configure_opencv_threads(0), 7)
        self.assertEqual(runtime.opencv_thread_policy(), 0)
        self.assertEqual(runtime.configure_opencv_threads(0), 7)
        self.setter.assert_not_called()
        with self.assertRaisesRegex(ValueError, "重启"):
            runtime.configure_opencv_threads(1)

    def test_conflict_never_reconfigures(self):
        runtime.configure_opencv_threads(1)
        for value in (0, 2, 20):
            with self.assertRaisesRegex(ValueError, "重启"):
                runtime.configure_opencv_threads(value)
        self.setter.assert_called_once_with(1)

    def test_invalid_values_do_not_lock(self):
        for value in (-1, 129, 1.5, "1"):
            with self.assertRaises(ValueError):
                runtime.configure_opencv_threads(value)
        self.assertIsNone(runtime._opencv_requested)
        self.setter.assert_not_called()

    def test_failed_initialization_can_retry(self):
        self.setter.side_effect = [RuntimeError("fault"), None]
        with self.assertRaisesRegex(RuntimeError, "fault"):
            runtime.configure_opencv_threads(1)
        self.assertIsNone(runtime._opencv_requested)
        self.assertEqual(runtime.configure_opencv_threads(1), 1)

    def test_service_precheck_and_multiple_engines(self):
        svc = SearchService(capture_log=False)
        svc.set_config({"opencv_threads": 2})
        self.setter.assert_not_called()
        cfg = svc.cfg
        cfg.png_decoder = "cv2"
        cfg.silence_png_warnings = False
        self.getter.return_value = 2
        self.assertEqual(HybridEngine(cfg).opencv_threads, 2)
        self.assertEqual(runtime.opencv_thread_policy(), 2)
        self.assertEqual(HybridEngine(cfg).opencv_threads, 2)
        with self.assertRaisesRegex(ValueError, "重启"):
            svc.set_config({"opencv_threads": 1})
        self.assertEqual(svc.cfg.opencv_threads, 2)
        self.setter.assert_called_once_with(2)

    def test_cli_explicit_and_default(self):
        parser = cli.build_parser()
        for value in (None, 0, 1, 20):
            argv = ["build", "unused"]
            if value is not None:
                argv += ["--opencv-threads", str(value)]
            cfg = Config()
            cli._apply_feature_args(cfg, parser.parse_args(argv))
            self.assertEqual(cfg.opencv_threads, Config().opencv_threads if value is None else value)
        self.setter.assert_not_called()

    def test_old_index_compatibility_not_tied_to_threads(self):
        cfg = Config(png_decoder="cv2", silence_png_warnings=False)
        engine = HybridEngine(cfg)
        engine._check_cfg_compat({"coarse_size": cfg.coarse_size, "model": cfg.model})
        engine._check_cfg_compat({"opencv_threads": 20})

    def test_web_schema_and_cli_hint(self):
        fields = [f for p in config_schema()["pages"] for g in p["groups"] for f in g["fields"]]
        field = next(f for f in fields if f["key"] == "opencv_threads")
        self.assertEqual(field["default"], Config().opencv_threads)
        self.assertEqual(field["cli"], "--opencv-threads")
        self.assertEqual(cli_hint("opencv_threads", 0), "--opencv-threads 0")
        self.assertIn("重启", field["tip"])

    def test_task_initializes_before_body_even_without_engine(self):
        svc = SearchService(capture_log=False)
        def body():
            self.setter.assert_called_once_with(Config().opencv_threads)
            return "initialized"
        self.assertEqual(svc._run("scan-test", "scan", body), "initialized")
        with self.assertRaisesRegex(ValueError, "重启"):
            svc.set_config({"opencv_threads": 20})


if __name__ == "__main__":
    program = unittest.main(exit=False)
    print(f"Runtime tests: {program.result.testsRun}, success={program.result.wasSuccessful()}")
    sys.exit(0 if program.result.wasSuccessful() else 1)
