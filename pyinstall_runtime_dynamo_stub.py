# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — PyInstaller runtime hook：把 torch._dynamo 替换为宽松 no-op stub
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
PyInstaller runtime hook：把 torch._dynamo 替换为宽松 no-op stub。

问题（PyInstaller 冻结环境特有）：
  torch/_dynamo/utils.py 会在 import torch 时被加载，它 import torch._numpy，
  而 torch/_numpy/_ufuncs.py 的模块级 `for name in _binary: vars()[name] = …`
  在 PyInstaller importer 下抛 `NameError: name 'name' is not defined`
  （PyInstaller 字节码管线问题，社区已确认，上游未修）。

本程序只使用 torch 的 eager 推理（前向 + cudnn.benchmark），不依赖
torch.compile / torch._dynamo，因此直接注入 no-op stub 到 sys.modules：
任何对 torch._dynamo 及其子模块的属性访问都返回透传 callable / 假值对象。
本 hook 不 import torch（避免提前触发 _numpy 崩溃链）。
"""
import sys
import types


class _Noop(types.ModuleType):
    """宽松模块：非 dunder 属性一律返回透传对象。"""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _passthrough(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]          # 装饰器用法 @torch._dynamo.xxx
            return None

        return _passthrough


def _install():
    stub = _Noop("torch._dynamo")
    stub.__path__ = []                   # 视为包，子模块导入继续走下方 finder
    stub.config = _Noop("torch._dynamo.config")
    stub.config.__path__ = []
    sys.modules["torch._dynamo"] = stub
    sys.modules["torch._dynamo.config"] = stub.config

    class _Finder:
        """torch._dynamo.* 子模块 -> 同款 no-op stub。"""

        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("torch._dynamo."):
                return None
            from importlib.machinery import ModuleSpec

            m = _Noop(fullname)
            m.__path__ = []
            sys.modules[fullname] = m
            return ModuleSpec(fullname, _Loader(), is_package=True)

    class _Loader:
        def create_module(self, spec):
            return sys.modules.get(spec.name)

        def exec_module(self, module):
            pass

    try:
        sys.meta_path.insert(0, _Finder())
    except Exception:
        pass

    # 若 torch 已加载，同步包属性（正常不会发生：本 hook 在 import torch 前）
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_mod._dynamo = stub


try:
    _install()
except Exception:
    pass
