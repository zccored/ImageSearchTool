# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 前端无关的服务层：扫描 / 建库（整图·瓦片·compact）/ 检索（三模式）/ 去重 / 引擎缓存与释放
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
服务层：前端无关的**唯一编排层**（原先混在 `gui.py` 里的编排搬到这里）。

tkinter 界面（`gui.py`）、Web 界面（`gui_web.py` + `frontend/`）与无 GUI 回归脚本
共用这一份逻辑；界面只负责“画界面 + 调本层 + 渲染事件”。

设计约定（IPC 友好）
--------------------
* **命令**：每个方法首参 `task_id`（字符串，可为空）。方法内部自己 try/except，
  **不向调用方抛异常**：异常一律变成 `task_error` 事件（返回 None）。
* **事件**：通过 `emit(event: dict)` 回调抛出，`event["event"]` ∈
  `log` / `progress` / `phase_boundary` / `viz_frame` / `perf_report` /
  `task_done` / `task_error`；除 `log` 外都带 `task_id`；载荷见下文「事件载荷」表
  （本 docstring 即契约，界面不得自行约定字段）。
* **线程无关**：本层不创建线程，调用方决定在哪个线程跑（tkinter 用
  `Thread(daemon=True)`，pywebview 在 js_api 线程）；`release_engines()` 是
  mmap 释放时机的唯一持有者（Windows 下持 mmap 时无法替换 `.npy`）。
* **薄编排**：不含算法、不复制参数默认值（默认值只在 `config.Config`），
  索引位置只由 `auto_prefix()` / `tile_index.tiles_prefix_of()` 推导。
* **与 CLI 的关系**：`cli.py` 目前另有一份编排（本阶段不动它）；本层与它并行，
  两边都保持“薄”，最终目标是 CLI 也走本层 —— 在那之前**两边的命令与事件语义必须一致**
  （同一份 `engine`，只是入口不同）。
* **剪贴板不在本层**：`copy_path` 属前端原生能力（Tk clipboard /
  `navigator.clipboard`），由界面各自实现；本层只提供 `open_in_shell()`。

事件载荷
--------
| 事件 | 载荷 |
| :--- | :--- |
| `log` | `level`、`time`（HH:MM:SS）、`text`（`task_id=""`，全局日志） |
| `progress` | `done`、`total`、`phase` |
| `phase_boundary` | `phase` ∈ {`save`,`done`}、`done`、`total`（铁律：任务必须以它收尾） |
| `viz_frame` | `kind` ∈ {`coarse`,`fine`}、`data`（numpy 帧） |
| `perf_report` | `path`、`modal`（是否弹窗提示）、`stage` |
| `task_done` | `op`、`result` |
| `task_error` | `op`、`error`、`traceback`、`cancelled`、`title`（可空，供界面做弹窗标题） |
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from .config import Config
from .engine import HybridEngine
from .io_utils import LOGGER, collect_images, load_rgb
from .store import IndexFiles
from .store import compact as _store_compact
from .store import prune as _store_prune
from .thumbs import ThumbCache

__all__ = [
    "SearchService", "TaskCancelled",
    "EVENT_LOG", "EVENT_PROGRESS", "EVENT_PHASE_BOUNDARY", "EVENT_VIZ_FRAME",
    "EVENT_PERF_REPORT", "EVENT_TASK_DONE", "EVENT_TASK_ERROR",
    "PHASE_LABELS", "PHASE_SAVE", "PHASE_DONE", "SEARCH_MODES",
    "OP_SCAN", "OP_BUILD", "OP_ADD", "OP_TILES", "OP_COMPACT", "OP_SEARCH",
    "OP_DEDUP_SCAN", "OP_DEDUP_APPLY", "OP_HANDOFF", "OP_STATS",
    "HIT_FIELDS", "config_schema", "cli_hint", "cli_command",
]

# ---------------------------------------------------------------------------
# 事件类型 / 阶段 / 任务 op
# ---------------------------------------------------------------------------
EVENT_LOG = "log"
EVENT_PROGRESS = "progress"
EVENT_PHASE_BOUNDARY = "phase_boundary"
EVENT_VIZ_FRAME = "viz_frame"
EVENT_PERF_REPORT = "perf_report"
EVENT_TASK_DONE = "task_done"
EVENT_TASK_ERROR = "task_error"

# 阶段名（与 engine / tile_index 的回调约定一致）
PHASE_FUSED = "fused"
PHASE_COARSE = "coarse"
PHASE_FINE = "fine"
PHASE_VERIFY = "verify"
PHASE_TILES = "tiles"
PHASE_COMPACT = "compact"
PHASE_SAVE = "save"          # 阶段边界：特征提取完成、开始写盘
PHASE_DONE = "done"          # 阶段边界：全部落盘完成

PHASE_BOUNDARIES = (PHASE_SAVE, PHASE_DONE)

# 阶段名 -> 界面文案（原 `gui.py._PHASE_LABELS`，界面统一从这里取，避免两份漂移）
PHASE_LABELS = {
    PHASE_FUSED: "粗筛+ResNet 融合提取(单遍解码)",
    PHASE_COARSE: "① 二值法粗筛",
    PHASE_FINE: "② ResNet 全库特征",
    PHASE_VERIFY: "解码校验",
    PHASE_SAVE: "③ 特征提取完成 · 正在写盘",
    PHASE_DONE: "✅ 全部完成（粗筛与精排均已落盘）",
    PHASE_COMPACT: "索引存储格式转换",
    PHASE_TILES: "子图索引(瓦片)",
}

# 检索模式（与 `cli.py --mode`、界面单选框一致）
SEARCH_MODES = ("full", "tiles", "hybrid")
HYBRID_TOP_K = 20            # 混合检索固定返回条数（界面/CLI 同口径）

# 任务 op（task_done / task_error 里回传，界面据此分发）
OP_SCAN = "scan"
OP_BUILD = "build"
OP_ADD = "add"
OP_TILES = "tiles"
OP_COMPACT = "compact"
OP_SEARCH = "search"
OP_DEDUP_SCAN = "dedup_scan"
OP_DEDUP_APPLY = "dedup_apply"
OP_HANDOFF = "handoff"
OP_STATS = "stats"

# 命中元组字段顺序（`search()` 返回的 hits 是元组列表，与 gui.py 网格渲染保持同一口径）
HIT_FIELDS = ("rank", "path", "fine_score", "coarse_score", "d_fp",
              "box", "match_kind")

# 任务出错时的弹窗标题（其余任务由界面用“操作失败”）
TASK_TITLES = {OP_HANDOFF: "自动交接"}

# ---------------------------------------------------------------------------
# 参数 schema（参数页 / 等价 CLI 提示的唯一来源）
# ---------------------------------------------------------------------------
# 说明：
#   * **默认值不在这里**：一律在运行时从 `Config()` 读取（`config.py` 是唯一出处）；
#     本表只描述“怎么展示、怎么校验、等价哪个 CLI 开关”。
#   * `page`/`group` 决定参数页分组；`ui=True` 表示不是 `Config` 字段（属界面/命令参数）。
#   * `cli` = 等价 CLI 开关名；反向布尔（不传=开启）在 `cli_hint` 里按“值=False 才输出”处理。
#   * `adv=True` 表示“性能/高级开关”（T1 参数页要完整覆盖 `Config` 全部字段）。
#   * 范围（min/max）用于界面输入校验；`cli.py` 目前不做范围校验，两者不冲突。
PARAM_FIELDS: Tuple[Dict, ...] = (
    # ---- 建库：二值粗筛 -----------------------------------------------------
    {"key": "coarse_size", "label": "指纹边长(像素,平方)", "kind": "int",
     "page": "build", "group": "二值粗筛", "min": 8, "max": 256,
     "cli": "--coarse-size", "tip": "二值指纹尺寸：64 → 64×64=4096bit/张"},
    {"key": "coarse_blur", "label": "高斯模糊核(奇数)", "kind": "int",
     "page": "build", "group": "二值粗筛", "min": 1, "max": 31, "cli": "--blur",
     "tip": "二值化前的降噪强度（CLI 开关名是 --blur）"},
    {"key": "hu_weight", "label": "Hu矩 融合权重", "kind": "float",
     "page": "build", "group": "二值粗筛", "min": 0.0, "max": 1.0,
     "cli": "--hu-weight"},
    {"key": "fp_weight", "label": "指纹 融合权重", "kind": "float",
     "page": "build", "group": "二值粗筛", "min": 0.0, "max": 1.0,
     "cli": "--fp-weight"},
    {"key": "use_hu", "label": "使用轮廓 Hu 矩特征", "kind": "bool",
     "page": "build", "group": "二值粗筛", "cli": "--no-hu",
     "tip": "建库参数；改动后已建索引会拒绝打开"},
    {"key": "use_fp", "label": "使用二值图像指纹特征", "kind": "bool",
     "page": "build", "group": "二值粗筛", "cli": "--no-fp"},
    {"key": "invert_binary", "label": "白像素过半时取反(白底图库)", "kind": "bool",
     "page": "build", "group": "二值粗筛", "cli": "--invert-binary"},
    # ---- 建库：ResNet 精排 --------------------------------------------------
    {"key": "model", "label": "模型", "kind": "choice",
     "page": "build", "group": "ResNet 精排",
     "choices": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
     "cli": "--model"},
    {"key": "device", "label": "设备", "kind": "choice",
     "page": "build", "group": "ResNet 精排", "choices": ["auto", "cuda", "cpu"],
     "cli": "--device"},
    {"key": "batch", "label": "批大小(0=自动)", "kind": "int",
     "page": "build", "group": "ResNet 精排", "min": 0, "max": 512,
     "cli": "--batch", "tip": "实测 CUDA 自动值 256 最优；瓦片不跟随本项"},
    {"key": "fp16", "label": "GPU 半精度 FP16", "kind": "bool",
     "page": "build", "group": "ResNet 精排", "cli": "--no-fp16"},
    {"key": "store_fine", "label": "预构建 ResNet 全库索引(快/占内存)", "kind": "bool",
     "page": "build", "group": "ResNet 精排", "cli": "--no-store-fine"},
    {"key": "png_decoder", "label": "PNG 解码器(旁路)", "kind": "choice",
     "page": "build", "group": "ResNet 精排",
     "choices": ["libdeflate", "cv2", "imagecodecs", "pillow"],
     "cli": "--png-decoder",
     "tip": "在硬校验名单里：与建库时不一致会拒绝打开索引"},
    {"key": "prep_cache", "label": "启用预处理缓存(重复建库更快/占磁盘)", "kind": "bool",
     "page": "build", "group": "ResNet 精排", "cli": "--no-prep-cache"},
    {"key": "dedup", "label": "MD5 内容去重", "kind": "bool",
     "page": "build", "group": "ResNet 精排", "cli": "--no-dedup"},
    {"key": "workers", "label": "粗筛并行线程(0=自动≤8)", "kind": "int",
     "page": "build", "group": "ResNet 精排", "min": 0, "max": 64,
     "cli": "--workers",
     "tip": "0=自动(不超过8且不超过CPU核数)，1=串行最省内存；\n"
            "解码在C层释放GIL，多线程可线性提速"},
    {"key": "decode_workers", "label": "精排解码线程(0=自动≤8)", "kind": "int",
     "page": "build", "group": "ResNet 精排", "min": 0, "max": 64,
     "cli": "--decode-workers",
     "tip": "图像读盘+解码+预处理的多线程数；\n解码与GPU前向重叠，GPU场景建议4~8"},
    # ---- 建库：解码并发（吞吐调优）-----------------------------------------
    {"key": "big_decode_conc", "label": "大图解码并发上限(>12MP)", "kind": "int",
     "page": "build", "group": "解码并发(建库吞吐调优)", "min": 1, "max": 64,
     "cli": "--big-decode-conc",
     "tip": "同时解码 12MP+ 大图(大 PNG/高分辨率 JPEG)的线程数上限。\n"
            "每张峰值内存 36MB~100MB+，16GB 内存机建议 16~20；\n"
            "数值越高 CPU 越满，但内存余量不足会触发页回收反而变慢"},
    {"key": "tile_decode_slots", "label": "瓦片建库·同时解码图数", "kind": "int",
     "page": "build", "group": "解码并发(建库吞吐调优)", "min": 1, "max": 64,
     "cli": "--tile-decode-slots",
     "tip": "子图(瓦片)索引第一级同时解码的原图数上限，配合上面的并发门使用"},
    {"key": "tile_fwd_batch", "label": "瓦片建库·前向批大小", "kind": "int",
     "page": "build", "group": "解码并发(建库吞吐调优)", "min": 1, "max": 512,
     "cli": None, "adv": True,
     "tip": "瓦片路径专用批大小（默认 64）。实测 256 会拖垮吞吐(22.9s vs 11.8s)，"
            "不跟随 batch"},
    {"key": "tile_flush_ms", "label": "瓦片建库·GPU批等待(ms)", "kind": "int",
     "page": "build", "group": "解码并发(建库吞吐调优)", "min": 2, "max": 500,
     "cli": "--tile-flush-ms",
     "tip": "瓦片不足一批时的最长等待毫秒。\n小(如10)：批更碎(1-2行小批唤醒多)；"
            "大(如30)：批更整、唤醒更少"},
    {"key": "opencv_threads", "label": "OpenCV内部线程(0=保留外部设置)", "kind": "int",
     "page": "build", "group": "解码并发(建库吞吐调优)", "min": 0, "max": 128,
     "cli": "--opencv-threads",
     "tip": "只限制 OpenCV 内部并行，外层图片解码仍并行。进程级设置：首次任务(含扫描)前生效；"
            "使用后修改需重启应用，并在首次操作前设置。0=由宿主管理，不恢复原值。"},
    {"key": "torch_threads", "label": "torch推理线程(0=默认)", "kind": "int",
     "page": "build", "group": "解码并发(建库吞吐调优)", "min": 0, "max": 128,
     "cli": "--torch-threads",
     "tip": "CPU前向线程数。多数机型1线程最快\n(实测多线程反而慢)，GPU场景无需设置"},
    # ---- 建库：高级（性能开关，CLI 未逐一暴露）------------------------------
    {"key": "dedup_prefilter", "label": "解码前按内容 MD5 预筛重复副本", "kind": "bool",
     "page": "build", "group": "高级(性能开关)", "cli": None, "adv": True,
     "tip": "同内容文件只处理第一张（逐位等价；真实库实测 5.4% 属此类）"},
    {"key": "tile_md5_reuse", "label": "瓦片块 md5 复用整文件哈希", "kind": "bool",
     "page": "build", "group": "高级(性能开关)", "cli": None, "adv": True,
     "tip": "每图只哈希一次，块 md5 由 md5.copy()+update(框) 派生（实测 15.4×）"},
    {"key": "cv2_rgb_direct", "label": "cv2 解码直出 RGB", "kind": "bool",
     "page": "build", "group": "高级(性能开关)", "cli": None, "adv": True,
     "tip": "省一次全图 BGR→RGB 拷贝（实测 -10.2%，逐位一致）"},
    {"key": "norm_on_gpu", "label": "归一化搬到 GPU", "kind": "bool",
     "page": "build", "group": "高级(性能开关)", "cli": None, "adv": True,
     "tip": "仅在 CUDA 生效：transform 只做 Resize/CenterCrop/ToTensor"},
    {"key": "png_fast_scratch_mb", "label": "PNG 旁路每线程 scratch 上限(MB)",
     "kind": "float", "page": "build", "group": "高级(性能开关)",
     "min": 0.0, "max": 1024.0, "cli": "--png-fast-scratch-mb", "adv": True},
    {"key": "silence_png_warnings", "label": "屏蔽 libpng/iCCP stderr 噪音",
     "kind": "bool", "page": "build", "group": "高级(性能开关)",
     "cli": "--no-silence-png", "adv": True},
    {"key": "fast_load", "label": "新索引用侧车 .npy(可 mmap 快载)", "kind": "bool",
     "page": "build", "group": "高级(性能开关)", "cli": "--fast-load", "adv": True,
     "tip": "默认已是 True（界面不再单列）；CLI 的 --fast-load 只在显式传参时生效，"
            "不传即用 Config 默认（也是 True）"},
    # ---- 建库：图片格式 -----------------------------------------------------
    {"key": "extensions", "label": "图片格式(逗号分隔)", "kind": "extensions",
     "page": "build", "group": "图片格式", "cli": "--ext",
     "tip": "扫描与建索引支持的扩展名，改完重新扫描生效"},
    # ---- 检索参数 -----------------------------------------------------------
    {"key": "coarse_k", "label": "粗筛候选数 coarse_k", "kind": "int",
     "page": "search", "group": "检索流程", "min": 10, "max": 100000,
     "cli": "--coarse-k", "tip": "查询时可改；其余建库参数改了会被 open() 拒绝"},
    {"key": "top_k", "label": "最终返回数 top_k", "kind": "int",
     "page": "search", "group": "检索流程", "min": 1, "max": 100,
     "cli": "--top-k"},
    {"key": "exclude_self", "label": "剔除查询图自身", "kind": "bool",
     "page": "search", "group": "检索流程", "cli": "--no-exclude-self"},
    # ---- 界面/命令参数（不是 Config 字段）-----------------------------------
    {"key": "prefix", "label": "索引前缀(留空=自动放图库内)", "kind": "text",
     "page": "search", "group": "索引位置", "cli": "--prefix", "ui": True,
     "tip": "例如 D:/idx/my_gallery\n留空自动为 <图库目录>/.gallery_index/gallery"},
    {"key": "perf_build", "label": "索引阶段导出性能图(有性能损耗)", "kind": "bool",
     "page": "build", "group": "性能图导出", "cli": None, "ui": True,
     "tip": "本次建库/增量被采样(0.4s 一行)，任务结束写 HTML+JSON 到 perf_reports/；"
            "采样开销 <1%"},
    {"key": "perf_search", "label": "搜图阶段导出性能图(有性能损耗)", "kind": "bool",
     "page": "search", "group": "性能图导出", "cli": None, "ui": True,
     "tip": "每次检索都被采样并产出报告（含加载索引/粗筛/精排阶段耗时与内存曲线）"},
)

PARAM_PAGES = (
    {"key": "build", "title": "建库参数"},
    {"key": "search", "title": "检索参数"},
)

# 参数页提示（与索引几何相关的参数改了会被拒绝打开）
PARAM_HINT = ("提示：与索引相关的几何参数(指纹边长/模糊核/Hu·指纹开关/取反)修改后\n"
              "已建索引会拒绝打开，需重新“全部入库”重建（防错乱）；\n"
              "权重/候选数/top_k/剔除自身可随时微调。")

# ---------------------------------------------------------------------------
# schema / 等价 CLI 辅助（模块级函数：无状态，供界面与脚本直接调用）
# ---------------------------------------------------------------------------
def _fields_for_page(page: str) -> List[Dict]:
    return [f for f in PARAM_FIELDS if f.get("page") == page]


def field_of(key: str) -> Optional[Dict]:
    """按 key 取字段元数据（找不到返回 None）。"""
    for f in PARAM_FIELDS:
        if f["key"] == key:
            return f
    return None


def config_keys() -> List[str]:
    """`Config` 上可调字段名（按 schema 顺序；不含 prefix / perf_* 这类界面参数）。"""
    return [f["key"] for f in PARAM_FIELDS if not f.get("ui")]


def _json_value(v):
    """元组 -> 列表（schema 要 JSON 可序列化给 Web 前端）。"""
    return list(v) if isinstance(v, tuple) else v


def config_schema(current: Optional[Config] = None) -> Dict:
    """参数页 schema（JSON 可序列化）。

    `default` 一律现场读 `Config()`（默认值只在 `config.py` 写一次），
    `value` 取 `current`（缺省=一份新 `Config()`）。
    返回 {version, pages:[{key,title,groups:[{title,fields:[…]}]}], hint}。
    """
    cur = current if current is not None else Config()
    pages = []
    for page in PARAM_PAGES:
        groups: List[Dict] = []
        for f in _fields_for_page(page["key"]):
            g = f.get("group") or ""
            if not groups or groups[-1]["title"] != g:
                groups.append({"title": g, "fields": []})
            item = {
                "key": f["key"], "label": f["label"], "kind": f["kind"],
                "group": g, "clr_adv": bool(f.get("adv")), "ui": bool(f.get("ui")),
                "cli": f.get("cli"), "tip": f.get("tip", ""),
                "min": f.get("min"), "max": f.get("max"),
                "choices": list(f.get("choices") or []),
                "default": _json_value(
                    getattr(Config(), f["key"]) if not f.get("ui")
                    else (False if f["kind"] == "bool" else "")),
                "value": _json_value(
                    getattr(cur, f["key"]) if not f.get("ui") else
                    (False if f["kind"] == "bool" else "")),
            }
            groups[-1]["fields"].append(item)
        pages.append({"key": page["key"], "title": page["title"],
                      "groups": groups})
    return {"version": 1, "pages": pages, "hint": PARAM_HINT}


def cli_hint(key: str, value=None, cfg: Optional[Config] = None) -> str:
    """单个字段的“等价 CLI 片段”；无对应开关或与默认值一致时返回 ""。

    `value` 缺省取 `cfg`（再缺省取默认 `Config()`）。
    反向布尔（`--no-xxx`）在值为 False 时输出；
    例外：`fast_load` 的 CLI 是 `store_true`（不传会覆盖回 False），
    故值为 True 时才输出 `--fast-load`。
    """
    f = field_of(key)
    if f is None or f.get("ui") or not f.get("cli"):
        return ""
    if cfg is not None and value is None:
        value = getattr(cfg, key, None)
    if value is None:
        value = getattr(Config(), key)
    dflt = getattr(Config(), key)
    if f["kind"] == "bool":
        neg = str(f["cli"]).startswith("--no-")
        if key == "fast_load":
            return f["cli"] if value else ""
        if neg:
            return f["cli"] if not value else ""
        return f["cli"] if value else ""
    if f["kind"] == "extensions":
        extra = [e for e in value if e not in dflt]
        return " ".join(f"{f['cli']} {e.lstrip('.')}" for e in sorted(extra))
    if _json_value(value) == _json_value(dflt):
        return ""
    return f"{f['cli']} {value}"


CLI_TEMPLATES = {
    "build": "python main.py build <图库目录>",
    "add": "python main.py add <图库目录>",
    "build-tiles": "python main.py build-tiles <图库目录>",
    "add-tiles": "python main.py add-tiles <图库目录>",
    "search": "python main.py search <查询图>",
    "eval": "python main.py eval <查询目录>",
    "stats": "python main.py stats",
    "compact": "python main.py compact",
    "ingest": "python main.py ingest <交接文件>",
}


def cli_command(op: str, values: Optional[Dict] = None,
                cfg: Optional[Config] = None,
                positionals: Optional[List[str]] = None,
                diff_only: bool = True) -> str:
    """拼一条“等价 CLI 命令”（界面 T1 的过渡提示用）。

    * `values`：界面上被改动的参数（key -> 值）；缺省用 `cfg`。
    * `diff_only=True` 时只输出与 `Config()` 默认不同的开关（命令更短、可复现）。
    * ⚠️ `--ext` 只能追加扩展名（CLI 与默认取并集），因此删扩展名的改动
      无法用命令行表达；这种情况会附一句注释提示。
    """
    head = CLI_TEMPLATES.get(op, f"python main.py {op}")
    parts = [head]
    if positionals:
        parts.extend(positionals)
    src = dict(values or {})
    for k in config_keys():
        v = src.get(k, getattr(cfg, k, None) if cfg is not None else None)
        if v is None:
            v = getattr(Config(), k)
        frag = cli_hint(k, v)
        if frag:
            parts.append(frag)
    if op == "search":
        parts.append("--mode full")   # 界面模式单选；调用方可自行替换
    if op in ("build-tiles", "add-tiles"):
        parts.append("--prefix <瓦片前缀>")
    note = ""
    if diff_only and cfg is not None:
        removed = [e for e in Config().extensions if e not in tuple(cfg.extensions)]
        if removed:
            note = ("    ⚠️ 已移除扩展名 " + ",".join(removed)
                    + "：CLI 的 --ext 只能追加，需手工改 Config 或界面内建库")
    return "  ".join(p.strip() for p in parts if p and p.strip()) + note


# ---------------------------------------------------------------------------
# 日志 -> 事件（替代原 `gui.QueueLogHandler`：日志不再只属于某个界面）
# ---------------------------------------------------------------------------
class _ServiceLogHandler(logging.Handler):
    """把 `hybrid_search` 日志转成 `log` 事件（格式与原 GUI 日志框一致）。"""

    def __init__(self, emit_fn: Callable[[Dict], None]) -> None:
        super().__init__()
        self._emit_fn = emit_fn
        self.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:      # noqa: D102
        try:
            text = self.format(record)
        except Exception:                                    # noqa: BLE001
            return
        try:
            self._emit_fn({
                "event": EVENT_LOG, "task_id": "",
                "level": record.levelname,
                "time": time.strftime("%H:%M:%S",
                                      time.localtime(record.created)),
                "text": text,
            })
        except Exception:                                    # noqa: BLE001
            pass


class TaskCancelled(RuntimeError):
    """任务在安全检查点被取消（不会写出部分索引）。"""


# ---------------------------------------------------------------------------
# 服务层
# ---------------------------------------------------------------------------
class SearchService:
    """前端无关的编排层（扫描 / 建库 / 检索 / 去重 / 引擎缓存与释放 / 性能画像）。"""

    def __init__(self, cfg: Optional[Config] = None, *,
                 emit: Optional[Callable[[Dict], None]] = None,
                 capture_log: bool = True,
                 logger_name: str = "hybrid_search") -> None:
        self.cfg = cfg if cfg is not None else Config()
        self.root = ""            # 图库根目录（前端只传根目录，前缀由本层推导）
        self.prefix = ""          # 显式索引前缀（空 = 按根目录推导）
        self.last_perf_report = ""
        self._listeners: List[Callable[[Dict], None]] = []
        if emit is not None:
            self._listeners.append(emit)
        # 引擎缓存：键=前缀，值=(签名, 引擎)；最多两套（整图 + 瓦片）
        self._eng_lock = threading.Lock()
        self._eng_cache: Dict[str, Tuple[tuple, HybridEngine]] = {}
        # 任务取消标志 / 性能图当前阶段
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._prof_phase: Dict[str, str] = {}
        # 索引 md5 清单缓存（路径 -> 整图 md5；缩略图 key 用，索引改写/释放时失效）
        self._md5_lock = threading.Lock()
        self._md5_map: Dict[str, Dict[str, str]] = {}
        self._log_handler: Optional[_ServiceLogHandler] = None
        if capture_log:
            self.attach_log(logger_name)

    # ==================================================================
    # 事件 / 日志
    # ==================================================================
    def subscribe(self, fn: Callable[[Dict], None]) -> Callable[[], None]:
        """追加事件监听；返回“取消监听”函数。"""
        self._listeners.append(fn)

        def _off() -> None:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass
        return _off

    def emit(self, event: Dict) -> None:
        """把事件派发给所有监听者（监听者异常不影响任务）。"""
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:                                # noqa: BLE001
                pass

    def _emit(self, event_kind: str, task_id: str = "", **payload) -> Dict:
        """发事件；`event_kind` 不用 `kind` 命名，避免与载荷键 `kind`（可视化帧）冲突。"""
        ev = {"event": event_kind, "task_id": task_id}
        ev.update(payload)
        self.emit(ev)
        return ev

    def attach_log(self, logger_name: str = "hybrid_search") -> None:
        if self._log_handler is not None:
            return
        self._log_handler = _ServiceLogHandler(self.emit)
        lg = logging.getLogger(logger_name)
        lg.addHandler(self._log_handler)
        if lg.level == logging.NOTSET:
            lg.setLevel(logging.INFO)

    def detach_log(self) -> None:
        if self._log_handler is None:
            return
        logging.getLogger("hybrid_search").removeHandler(self._log_handler)
        self._log_handler = None

    def close(self) -> None:
        """解绑日志、释放引擎（界面退出时调用）。"""
        self.detach_log()
        self.release_engines("服务层关闭")

    # ==================================================================
    # 任务包装：取消 → 错误 → 完成（命令一律不向调用方抛异常）
    # ==================================================================
    def cancel(self, task_id: str) -> None:
        """请求取消任务（在安全检查点生效；引擎内部不中断，避免半成品索引）。"""
        ev = self._cancel_flags.get(task_id)
        if ev is None:
            ev = threading.Event()
            self._cancel_flags[task_id] = ev
        ev.set()

    def is_cancelled(self, task_id: str) -> bool:
        ev = self._cancel_flags.get(task_id)
        return bool(ev is not None and ev.is_set())

    def _check_cancel(self, task_id: str) -> None:
        if self.is_cancelled(task_id):
            raise TaskCancelled(f"任务已取消：{task_id}")

    def _run(self, task_id: str, op: str, fn: Callable,
             err_prefix: str = ""):
        """执行任务体：异常 -> `task_error`（返回 None）；成功 -> `task_done`。

        `err_prefix` 让错误文案与旧界面一致（如“扫描失败：…”）。
        """
        try:
            # 扫描校验也会解码；必须在任何任务体进入 OpenCV 前固定策略。
            from .runtime import configure_opencv_threads
            configure_opencv_threads(self.cfg.opencv_threads)
            result = fn()
        except TaskCancelled as e:                           # noqa: BLE001
            self._emit(EVENT_TASK_ERROR, task_id, op=op, error=str(e),
                       traceback="", cancelled=True,
                       title=TASK_TITLES.get(op))
            return None
        except Exception as e:                               # noqa: BLE001
            self._emit(EVENT_TASK_ERROR, task_id, op=op,
                       error=f"{err_prefix}{e}",
                       traceback=traceback.format_exc(), cancelled=False,
                       title=TASK_TITLES.get(op))
            return None
        finally:
            self._cancel_flags.pop(task_id, None)
            self._prof_phase.pop(task_id, None)
        self._emit(EVENT_TASK_DONE, task_id, op=op, result=result)
        return result

    # ==================================================================
    # 参数：schema / 读写 / 等价 CLI（P1 参数页的唯一来源）
    # ==================================================================
    def get_config_schema(self) -> Dict:
        return config_schema(self.cfg)

    def get_config(self) -> Dict:
        """当前参数（含未改动过的默认值；不含 prefix / perf_* 这类界面参数）。"""
        return {k: _json_value(getattr(self.cfg, k)) for k in config_keys()}

    def set_config(self, values: Dict) -> Dict:
        """按 schema 校验并写入当前参数；返回被采纳的项（非法值抛 ValueError）。"""
        applied: Dict = {}
        for k, v in dict(values or {}).items():
            f = field_of(k)
            if f is None or f.get("ui"):
                raise ValueError(f"未知参数: {k}")
            kind = f["kind"]
            if kind == "bool":
                v = bool(v)
            elif kind == "int":
                v = int(v)
                self._check_range(f, v)
            elif kind == "float":
                v = float(v)
                self._check_range(f, v)
            elif kind == "choice":
                v = str(v)
                if v not in list(f.get("choices") or []):
                    raise ValueError(f"{k}={v} 不在可选值 {f.get('choices')}")
            elif kind == "extensions":
                v = self.parse_extensions(v)
            else:
                v = str(v)
            if k == "opencv_threads":
                from .runtime import check_opencv_threads
                check_opencv_threads(v)
            setattr(self.cfg, k, v)
            applied[k] = _json_value(v)
        return applied

    @staticmethod
    def _check_range(f: Dict, v) -> None:
        lo, hi = f.get("min"), f.get("max")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(f"{f['key']}={v} 超出范围 [{lo},{hi}]")

    @staticmethod
    def parse_extensions(text) -> tuple:
        """把“jpg,png” / 列表 规整成 ('.jpg', '.png')（与 gui.py 同一写法）。"""
        items = list(text) if isinstance(text, (list, tuple)) else \
            str(text).replace("，", ",").split(",")
        out = set()
        for raw in items:
            e = str(raw).strip()
            if e:
                out.add(e if e.startswith(".") else "." + e)
        return tuple(sorted(out))

    @staticmethod
    def cli_hint(key: str, value=None, cfg: Optional[Config] = None) -> str:
        return cli_hint(key, value, cfg)

    def cli_command(self, op: str, values: Optional[Dict] = None,
                    positionals: Optional[List[str]] = None) -> str:
        """当前参数下的等价 CLI 命令（界面 T1 的过渡提示）。"""
        return cli_command(op, values, self.cfg, positionals)

    # ==================================================================
    # 位置：图库根 → 索引前缀（推导只在本层）
    # ==================================================================
    def set_location(self, root: Optional[str] = None,
                     prefix: Optional[str] = None) -> Dict:
        if root is not None:
            self.root = str(root).strip().rstrip("\\/")
        if prefix is not None:
            self.prefix = str(prefix).strip()
        with self._md5_lock:                 # 换图库/前缀：md5 清单缓存作废
            self._md5_map.clear()
        return self.location()

    def location(self) -> Dict:
        p = self.current_prefix()
        return {"root": self.root, "prefix": p,
                "tiles_prefix": self.tiles_prefix(p)}

    def current_prefix(self) -> str:
        return self.prefix or self.auto_prefix(self.root)

    @staticmethod
    def auto_prefix(root: str) -> str:
        """<图库根>/.gallery_index/gallery（唯一推导处，别处不要手拼）。"""
        d = str(root or "").strip().rstrip("\\/")
        return os.path.join(d, ".gallery_index", "gallery") if d else ""

    @staticmethod
    def tiles_prefix(prefix: str) -> str:
        """瓦片前缀：同目录 gallery_tiles（由 tile_index.tiles_prefix_of 给）。"""
        if not prefix:
            return ""
        from .tile_index import tiles_prefix_of
        return tiles_prefix_of(prefix)

    @staticmethod
    def meta_exists(prefix: str) -> bool:
        return bool(prefix) and os.path.exists(prefix + ".meta.json")

    def index_status(self, prefix: Optional[str] = None) -> Dict:
        """索引状态（只读 meta、不加载引擎）：界面显示“已索引 / 瓦片库状态”用。"""
        p = prefix or self.current_prefix()
        tp = self.tiles_prefix(p)
        out = {"prefix": p, "tiles_prefix": tp,
               "full_exists": self.meta_exists(p),
               "tiles_exists": bool(tp) and self.meta_exists(tp),
               "n": 0, "tiles_n": 0, "storage": "", "fine_exists": False}
        if out["full_exists"]:
            try:
                meta = IndexFiles(p).load_meta()
                out["n"] = int(meta.get("n") or 0)
                out["storage"] = str(meta.get("storage") or "")
                out["fine_exists"] = bool((meta.get("fine") or {}).get("exists"))
            except Exception as e:                           # noqa: BLE001
                LOGGER.warning("读取索引 meta 失败：%s", e)
        if out["tiles_exists"]:
            try:
                out["tiles_n"] = int(IndexFiles(tp).load_meta().get("n") or 0)
            except Exception as e:                           # noqa: BLE001
                LOGGER.warning("读取瓦片索引 meta 失败：%s", e)
        return out

    def indexed_paths(self, prefix: Optional[str] = None) -> List[str]:
        """已入库路径（normcase+abspath，供界面打 ✔/· 标记）；失败返回空表。"""
        p = prefix or self.current_prefix()
        if not self.meta_exists(p):
            return []
        try:
            data = IndexFiles(p).load_coarse()
            return [os.path.normcase(os.path.abspath(x)) for x in data["paths"]]
        except Exception as e:                               # noqa: BLE001
            LOGGER.warning("读取已索引清单失败：%s", e)
            return []

    # ---- 缩略图 key / 索引 md5 清单（Web 版 `/thumb/<key>` 路由用）----
    def index_md5_map(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """`{normcase(abspath(path)): 整图 md5}`（按前缀缓存；索引改写/释放时失效）。

        没有索引或读取失败时返回空表 —— 调用方自行退化为“路径 md5”。
        """
        p = prefix or self.current_prefix()
        if not self.meta_exists(p):
            return {}
        with self._md5_lock:
            cached = self._md5_map.get(p)
            if cached is not None:
                return cached
        out: Dict[str, str] = {}
        try:
            st = IndexFiles(p).load_coarse()
            md5s = st.get("md5s") or []
            for i, path in enumerate(st.get("paths") or []):
                md5 = md5s[i] if i < len(md5s) else ""
                if md5:
                    out[os.path.normcase(os.path.abspath(path))] = str(md5)
        except Exception as e:                               # noqa: BLE001
            LOGGER.warning("读取索引 md5 清单失败：%s", e)
        with self._md5_lock:
            self._md5_map[p] = out
        return out

    def thumb_keys(self, paths, prefix: Optional[str] = None) -> List[str]:
        """路径列表 -> 缩略图 key 列表（有序、一一对应）。

        key 优先取索引里的**整图 md5**（同内容副本共用一张缩略图），
        索引中没有该路径（或没有索引）时退化为 `ThumbCache.key_for(path)`。
        """
        md5s = self.index_md5_map(prefix)
        out: List[str] = []
        for p in (paths or []):
            path = str(p)
            md5 = md5s.get(os.path.normcase(os.path.abspath(path)), "") if md5s else ""
            out.append(md5 or ThumbCache.key_for(path))
        return out

    # ==================================================================
    # 事件回调：进度 / 阶段边界 / 可视化帧
    # ==================================================================
    def _progress(self, task_id: str, done, total, phase, prof=None) -> None:
        """进度回调入口：发 `progress`；阶段边界（save/done）另发 `phase_boundary`。"""
        try:
            done_i, total_i = int(done), int(total)
        except Exception:                                    # noqa: BLE001
            return
        phase = phase or ""
        self._emit(EVENT_PROGRESS, task_id, done=done_i,
                   total=total_i, phase=phase)
        if phase in PHASE_BOUNDARIES:
            # 铁律：任务必须以 save / done 收尾（devtools/verify_phase_events.py 守）
            self._emit(EVENT_PHASE_BOUNDARY, task_id, phase=phase,
                       done=done_i, total=total_i)
        if prof is None:
            return
        prof.bump(done_i, total_i)
        if phase and phase != self._prof_phase.get(task_id):
            self._prof_phase[task_id] = phase
            label = PHASE_LABELS.get(phase, phase)
            prof.mark(f"阶段：{label}", 计数=f"{done_i}/{total_i}")
            if phase == PHASE_DONE:
                prof.mark("特征提取与落盘全部完成")

    def _progress_cb(self, task_id: str, prof=None,
                     default_phase: str = "") -> Callable:
        """engine 的 (done,total,phase) 回调；`default_phase` 供两参回调（瓦片）用。"""
        def cb(done, total, phase=None):
            self._progress(task_id, done, total,
                           default_phase if phase is None else phase, prof)
        return cb

    def _frame_cb(self, task_id: str) -> Callable:
        """engine 的 frame_sink(path, data, phase) -> `viz_frame` 事件（节流在界面侧）。"""
        def cb(_path, data, phase):
            try:
                self._emit(EVENT_VIZ_FRAME, task_id, kind=phase, data=data)
            except Exception:                                # noqa: BLE001
                pass
        return cb

    # ==================================================================
    # 引擎缓存 / mmap 释放（时机的唯一持有者）
    # ==================================================================
    @staticmethod
    def engine_signature(cfg: Config, prefix: str) -> tuple:
        """影响已加载索引语义/特征的参数 —— 变则缓存失效。"""
        return (prefix, cfg.model, cfg.device, bool(cfg.fp16), cfg.coarse_size,
                cfg.coarse_blur, bool(cfg.use_hu), bool(cfg.use_fp),
                bool(cfg.invert_binary), getattr(cfg, "png_decoder", "cv2"),
                float(cfg.hu_weight), float(cfg.fp_weight))

    def engine_for(self, cfg: Optional[Config] = None,
                   prefix: Optional[str] = None):
        """返回 (引擎, 是否命中缓存)；未命中时加载索引（慢，见 open()）。"""
        cfg = cfg or self.cfg
        prefix = prefix or self.current_prefix()
        sig = self.engine_signature(cfg, prefix)
        with self._eng_lock:
            hit = self._eng_cache.get(prefix)
            if hit is not None and hit[0] == sig:
                return hit[1], True
        eng = HybridEngine(cfg)
        eng.open(prefix)
        with self._eng_lock:
            self._eng_cache[prefix] = (sig, eng)
            # 最多两套：整图 + 瓦片（混合检索同时用）；多出的先淘汰
            while len(self._eng_cache) > 2:
                for k in list(self._eng_cache):
                    if k != prefix:
                        self._eng_cache.pop(k, None)
                        break
                else:
                    break
        return eng, False

    def release_engines(self, reason: str = "", silent: bool = False) -> float:
        """释放缓存的索引引擎（mmap + 桶表）并回收内存；返回 RSS 回落 MB。

        Windows 下被 mmap 的 `.npy` 无法替换/删除 → 写索引前必须先调用本方法。
        """
        with self._eng_lock:
            n = len(self._eng_cache)
            self._eng_cache.clear()
        with self._md5_lock:                 # 索引可能已改写：md5 清单缓存一并作废
            self._md5_map.clear()
        freed = 0.0
        try:
            import gc
            import psutil
            p = psutil.Process()
            before = p.memory_info().rss
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:                                # noqa: BLE001
                pass
            freed = max(0.0, (before - p.memory_info().rss) / 2 ** 20)
        except Exception:                                    # noqa: BLE001
            pass
        if not silent:
            LOGGER.info("已释放索引内存：%d 套引擎，RSS 回落 %.0f MB%s", n, freed,
                        f"（{reason}）" if reason else "")
        return freed

    def _drop_engines(self, reason: str = "") -> None:
        """内部用法：静默释放（不刷日志，避免建库/检索日志被噪音淹没）。"""
        self.release_engines(reason, silent=True)

    # ==================================================================
    # 扫描图库
    # ==================================================================
    def scan(self, task_id: str, root: str, *, recursive: bool = True,
             verify: bool = False, cfg: Optional[Config] = None) -> Optional[Dict]:
        """扫描图库目录（可选“逐张解码校验”）。

        返回 {paths, count, broken, elapsed, formats}（与界面消息载荷一致）。
        """
        cfg = cfg or self.cfg

        def job() -> Dict:
            t0 = time.time()
            paths = collect_images(root, cfg.extensions, sort=True)
            if not recursive:
                base = os.path.normcase(os.path.abspath(root)) + os.sep
                paths = [p for p in paths
                         if os.path.sep not in
                         os.path.normcase(os.path.abspath(p))[len(base):]]
            cnt = Counter(os.path.splitext(p)[1].lower() for p in paths)
            broken = 0
            if verify and paths:
                ok = []
                for i, p in enumerate(paths):
                    self._check_cancel(task_id)
                    if i % 20 == 0:
                        self._progress(task_id, i, len(paths), PHASE_VERIFY)
                    if load_rgb(p) is None:
                        broken += 1
                    else:
                        ok.append(p)
                paths = ok
            return {"paths": paths, "count": len(paths), "broken": broken,
                    "elapsed": time.time() - t0,
                    "formats": dict(cnt.most_common())}
        return self._run(task_id, OP_SCAN, job, "扫描失败：")

    # ==================================================================
    # 性能画像（可选导出；未开时零开销）
    # ==================================================================
    @staticmethod
    def _perf_start(stage: str, title: str, cfg: Config, prefix: str,
                    meta: Optional[Dict] = None, enabled: bool = False):
        if not enabled:
            return None
        from perfwatch import StageProfiler
        prof = StageProfiler(stage, title, cfg=cfg, prefix=prefix, meta=meta)
        prof.start()
        return prof

    def _perf_finish(self, prof, *, task_id: str = "", modal: bool = False,
                     status: str = "ok", extra: Optional[list] = None,
                     note: str = "") -> str:
        """收尾并落盘报告；落盘后发 `perf_report` 事件（界面据此提示/点亮按钮）。"""
        if prof is None:
            return ""
        path = prof.stop(status=status, extra=extra, note=note)
        if path:
            self.last_perf_report = path
            self._emit(EVENT_PERF_REPORT, task_id, path=path,
                       modal=bool(modal), stage=prof.stage)
        return path

    def latest_perf_report(self) -> str:
        """最近一次生成的性能图（本进程内优先；否则取 perf_reports/ 最新）。"""
        if self.last_perf_report and os.path.exists(self.last_perf_report):
            return self.last_perf_report
        try:
            from perfwatch import latest_report
            return latest_report()
        except Exception:                                    # noqa: BLE001
            return ""

    # ==================================================================
    # 建库：整图索引（全量 build / 增量 add）
    # ==================================================================
    def build_index(self, task_id: str, prefix: Optional[str] = None, *,
                    paths: Optional[List[str]] = None,
                    img_dir: Optional[str] = None, force: bool = True,
                    cfg: Optional[Config] = None, perf: bool = False,
                    title: str = "",
                    perf_meta: Optional[Dict] = None) -> Optional[Dict]:
        """整图索引全量构建。返回 {n, prefix, title, perf}。"""
        return self._index_run(task_id, OP_BUILD, "build", prefix, paths=paths,
                               img_dir=img_dir, force=force, cfg=cfg, perf=perf,
                               title=title, perf_meta=perf_meta)

    def add_index(self, task_id: str, prefix: Optional[str] = None, *,
                  paths: Optional[List[str]] = None,
                  img_dir: Optional[str] = None,
                  cfg: Optional[Config] = None, perf: bool = False,
                  title: str = "",
                  perf_meta: Optional[Dict] = None) -> Optional[Dict]:
        """整图索引增量（路径 + MD5 去重）。返回 {n, prefix, title, perf}。"""
        return self._index_run(task_id, OP_ADD, "add", prefix, paths=paths,
                               img_dir=img_dir, force=False, cfg=cfg, perf=perf,
                               title=title, perf_meta=perf_meta)

    def _index_run(self, task_id: str, op: str, action: str,
                   prefix: Optional[str], *, paths, img_dir, force,
                   cfg, perf, title, perf_meta) -> Optional[Dict]:
        cfg = cfg or self.cfg
        prefix = prefix or self.current_prefix()
        title = title or ("新建索引" if action == "build" else "增量入库")

        def job() -> Dict:
            meta: Dict = {"图片数": len(paths)} if paths is not None else {}
            if perf_meta:
                meta.update(perf_meta)
            prof = self._perf_start("index", title, cfg, prefix, meta,
                                    enabled=perf)
            try:
                # 建库前先释放缓存引擎：把内存让给解码流水线（瓦片库可占 1GB+）
                self._drop_engines("建库前腾内存")
                self._check_cancel(task_id)
                t0 = time.time()
                eng = HybridEngine(cfg)
                if prof:
                    prof.mark("初始化引擎", 耗时=round(time.time() - t0, 3))
                    prof.mark("解码+特征提取(粗筛与ResNet同一条流水线)")
                kw = {"img_dir": img_dir, "paths": paths,
                      "progress": self._progress_cb(task_id, prof),
                      "frame_sink": self._frame_cb(task_id)}
                if action == "build":
                    kw["force"] = force
                n = (eng.build if action == "build" else eng.add)(prefix, **kw)
                if prof:
                    # op 返回即代表“特征提取 + 落盘”全部结束（engine 内部有
                    # save/done 阶段边界事件，这里再补一条终结打点）
                    prof.mark("op 返回：提取+落盘完成", 张数=n)
                perf_path = self._perf_finish(
                    prof, task_id=task_id, modal=True, note=f"{n} 张",
                    extra=[("结果", [("入库张数", n)])])
                return {"n": n, "prefix": prefix, "title": title,
                        "perf": perf_path}
            except Exception as e:                           # noqa: BLE001
                self._perf_finish(prof, task_id=task_id, status="error",
                                  note=str(e)[:120])
                raise
        return self._run(task_id, op, job, f"{title}失败：")

    # ==================================================================
    # 建库：子图（瓦片）索引
    # ==================================================================
    def tiles_index(self, task_id: str, tiles_prefix: Optional[str] = None, *,
                    paths: Optional[List[str]] = None,
                    img_dir: Optional[str] = None,
                    exists: Optional[bool] = None,
                    cfg: Optional[Config] = None, perf: bool = False,
                    title: str = "",
                    perf_meta: Optional[Dict] = None) -> Optional[Dict]:
        """子图(瓦片)索引：无索引 → 全量构建；已有 → 增量（切块参数从 meta 恢复）。

        `exists` 缺省按 meta 是否存在判断。返回 {n, title, prefix, exists, perf}。
        """
        cfg = cfg or self.cfg
        tp = tiles_prefix or self.tiles_prefix(self.current_prefix())
        exists = self.meta_exists(tp) if exists is None else bool(exists)
        title = title or ("子图索引增量" if exists else "子图索引构建")

        def job() -> Dict:
            from . import tile_index as TI
            meta: Dict = {"图片数": len(paths)} if paths is not None else {}
            meta["模式"] = "增量" if exists else "全量"
            if perf_meta:
                meta.update(perf_meta)
            prof = self._perf_start("index", title, cfg, tp, meta, enabled=perf)
            try:
                self._drop_engines("瓦片建库前腾内存")
                self._check_cancel(task_id)
                eng = HybridEngine(cfg)
                # 瓦片进度回调是 (done,total) 两参；带第三参时是阶段边界
                # （save/done），必须原样透传，否则性能图看不到“结束状态”
                cb = self._progress_cb(task_id, prof, default_phase=PHASE_TILES)
                viz = self._frame_cb(task_id)
                if exists:
                    t0 = time.time()
                    eng.open(tp)
                    if prof:
                        prof.mark("加载已有瓦片索引", 复用="否",
                                  行数=eng.coarse.size,
                                  耗时=round(time.time() - t0, 2))
                        prof.mark("增量解码+特征提取")
                    n = TI.add_tiles(eng, tp, img_dir=img_dir, paths=paths,
                                     progress=cb, frame_sink=viz)
                else:
                    if prof:
                        prof.mark("全量解码+特征提取")
                    n = TI.build_tiles(eng, tp, img_dir=img_dir, paths=paths,
                                       progress=cb, frame_sink=viz)
                if prof:
                    prof.mark("写盘完成")
                perf_path = self._perf_finish(
                    prof, task_id=task_id, modal=True, note=f"{n} 块",
                    extra=[("结果", [("入库瓦片", n)])])
                return {"n": n, "title": title, "prefix": tp,
                        "exists": exists, "perf": perf_path}
            except Exception as e:                           # noqa: BLE001
                self._perf_finish(prof, task_id=task_id, status="error",
                                  note=str(e)[:120])
                raise
        return self._run(task_id, OP_TILES, job, f"{title}失败：")

    # ==================================================================
    # 建库：索引存储格式转换（旧 npz → 侧车 .npy，幂等）
    # ==================================================================
    def compact(self, task_id: str, prefixes: Optional[List[str]] = None, *,
                prefix: Optional[str] = None, cfg: Optional[Config] = None,
                perf: bool = False, delete_legacy: bool = False,
                title: str = "索引存储格式转换") -> Optional[Dict]:
        """转换索引存储格式；`prefixes` 缺省 = 整图 + 瓦片前缀中存在的那些。

        返回 {prefixes: [每个索引的转换结果], perf}。
        """
        cfg = cfg or self.cfg
        if prefixes is None:
            base = prefix or self.current_prefix()
            prefixes = [p for p in (base, self.tiles_prefix(base))
                        if self.meta_exists(p)]
        targets = [p for p in list(prefixes or []) if self.meta_exists(p)]

        def job() -> Dict:
            if not targets:
                raise FileNotFoundError("没有找到可转换的索引")
            prof = self._perf_start("index", title, cfg, ", ".join(targets),
                                    {}, enabled=perf)
            try:
                self._drop_engines("转换前释放索引内存")
                self._check_cancel(task_id)
                total, out = len(targets), []
                for i, p in enumerate(targets):
                    r = _store_compact(
                        p, delete_legacy=delete_legacy,
                        progress=self._compact_prog(task_id, i, total, prof))
                    LOGGER.info("%s", (
                        f"{os.path.basename(p)}.* 已是侧车格式"
                        if r.get("already") else
                        f"{os.path.basename(p)}.* 转换完成：{r['n']} 行，"
                        f"{r['sec']:.1f}s"))
                    out.append(r)
                perf_path = self._perf_finish(prof, task_id=task_id, modal=True,
                                              note="完成")
                return {"prefixes": out, "perf": perf_path}
            except Exception as e:                           # noqa: BLE001
                self._perf_finish(prof, task_id=task_id, status="error",
                                  note=str(e)[:120])
                raise
        return self._run(task_id, OP_COMPACT, job, "索引存储转换失败：")

    def _compact_prog(self, task_id: str, i: int, total: int,
                      prof=None) -> Callable:
        """compact 的 (done,total,phase) 回调：折算成“全部索引”的统一进度。"""
        def cb(done, total_i, phase=""):
            self._progress(task_id, i * int(total_i) + int(done),
                           total * int(total_i), PHASE_COMPACT, prof)
        return cb

    # ==================================================================
    # 检索（三种模式 + 混合合并）
    # ==================================================================
    def search(self, task_id: str, query: str, mode: str = "full", *,
               prefix: Optional[str] = None, cfg: Optional[Config] = None,
               coarse_k: Optional[int] = None, top_k: Optional[int] = None,
               perf: bool = False, title: str = "") -> Optional[Dict]:
        """以图搜图：full（整图）/ tiles（瓦片）/ hybrid（两路合并，固定 Top-20）。

        返回 {hits, db, kept, coarse_only, self_excluded, times, method, perf}；
        `hits` 是元组列表，字段顺序见 `HIT_FIELDS`（界面网格按同一下标渲染）。
        """
        cfg = cfg or self.cfg
        prefix = prefix or self.current_prefix()
        mode = str(mode or "full").lower()
        coarse_k = cfg.coarse_k if coarse_k is None else int(coarse_k)
        top_k = cfg.top_k if top_k is None else int(top_k)
        title = title or f"以图搜图({mode})"

        def job() -> Dict:
            from . import tile_index as TI
            prof = self._perf_start("search", title, cfg, prefix,
                                    {"查询图": os.path.basename(query),
                                     "模式": mode}, enabled=perf)
            try:
                if mode == "full":
                    t0 = time.time()
                    eng, cached = self.engine_for(cfg, prefix)
                    if prof:
                        prof.mark("加载索引", 复用="是" if cached else "否",
                                  耗时=round(time.time() - t0, 3),
                                  库内=eng.coarse.size)
                        prof.mark("粗筛+ResNet检索")
                    out = eng.search(query, coarse_k=coarse_k, top_k=top_k)
                elif mode == "tiles":
                    tp = TI.tiles_prefix_of(prefix)
                    t0 = time.time()
                    eng, cached = self.engine_for(cfg, tp)
                    if prof:
                        prof.mark("加载瓦片索引", 复用="是" if cached else "否",
                                  耗时=round(time.time() - t0, 3),
                                  瓦片数=eng.coarse.size)
                        prof.mark("瓦片检索(切块聚合+精排)")
                    out = TI.search_tiles_tiled(eng, query, top_k=top_k,
                                                coarse_k=coarse_k)
                elif mode == "hybrid":
                    # 两路各取 HYBRID_TOP_K 再合并（同一原图去重取高分）
                    tp = TI.tiles_prefix_of(prefix)
                    t0 = time.time()
                    eng_full, c1 = self.engine_for(cfg, prefix)
                    eng_t, c2 = self.engine_for(cfg, tp)
                    # 共享特征提取器（同 cfg 模型），避免重复加载模型
                    ex = getattr(eng_full, "_extractor", None)
                    if ex is not None:
                        eng_t._extractor = ex      # noqa: SLF001 同项目协作
                    if prof:
                        prof.mark("加载两套索引", 复用=f"{c1}/{c2}",
                                  耗时=round(time.time() - t0, 3))
                        prof.mark("整图检索")
                    o_full = eng_full.search(query, coarse_k=coarse_k,
                                             top_k=HYBRID_TOP_K)
                    if prof:
                        prof.mark("瓦片检索")
                    o_t = TI.search_tiles_tiled(eng_t, query, top_k=HYBRID_TOP_K,
                                                coarse_k=coarse_k)
                    out = TI.merge_hybrid(o_full, o_t, query, top_k=HYBRID_TOP_K)
                else:
                    raise ValueError(f"未知检索模式：{mode}"
                                     f"（可选 {'/'.join(SEARCH_MODES)}）")
                hits = [(h.rank, h.path, h.fine_score, h.coarse_score, h.d_fp,
                         h.box, h.match_kind) for h in out.hits]
                perf_path = self._perf_finish(
                    prof, task_id=task_id, note=f"命中 {len(hits)} 条",
                    extra=[("检索耗时明细(s)", list(out.times.items())),
                           ("结果", [("库内条目", out.db_size),
                                     ("候选保留", out.coarse_kept),
                                     ("仅粗筛", int(bool(out.coarse_only))),
                                     ("剔除自身", int(bool(out.self_excluded)))])])
                return {"hits": hits, "db": out.db_size, "kept": out.coarse_kept,
                        "coarse_only": out.coarse_only,
                        "self_excluded": out.self_excluded,
                        "times": dict(out.times), "method": mode,
                        "perf": perf_path}
            except Exception as e:                           # noqa: BLE001
                self._perf_finish(prof, task_id=task_id, status="error",
                                  note=str(e)[:120])
                raise
        return self._run(task_id, OP_SEARCH, job, "检索失败：")

    # ==================================================================
    # 去重：扫描 / 应用（删除·移动）/ 索引同步（prune）
    # ==================================================================
    def dedup_scan(self, task_id: str, paths: List[str], *,
                   threshold: float = 0.02,
                   prefix: Optional[str] = None,
                   cfg: Optional[Config] = None):
        """扫描重复图（完全重复 MD5 + 近似重复指纹汉明）；有索引时复用其 md5/指纹。

        `threshold` 是汉明比例上限（0.02 = 4096 位里差异 ≤82 位）。
        返回 `dedup.DupReport`。
        """
        def job():
            from . import dedup as DD
            p = prefix if prefix is not None else self.current_prefix()
            use = p if self.meta_exists(p) else None      # 索引缺失就纯解码
            return DD.scan_duplicates(
                list(paths), prefix=use, threshold=float(threshold),
                cancel=(lambda: self.is_cancelled(task_id)),
                progress=self._dedup_prog(task_id))
        return self._run(task_id, OP_DEDUP_SCAN, job, "查验去重失败：")

    def _dedup_prog(self, task_id: str) -> Callable:
        def cb(done, total, phase=""):
            self._progress(task_id, done, max(int(total), 1),
                           f"查验去重·{phase}")
        return cb

    def dedup_delete(self, task_id: str, paths: List[str], *,
                     prefix: Optional[str] = None,
                     sync: bool = True) -> Optional[Dict]:
        """把选中的重复图移入 Windows 回收站（可还原），随后按需同步索引。

        返回 {removed, failed, prune}：removed/failed 是 (成功路径) /
        [(失败路径, 原因)]，prune 是各前缀的 `store.prune()` 结果。
        """
        def job() -> Dict:
            from . import dedup as DD
            ok, bad = DD.recycle_paths(list(paths))
            pruned = self.prune_index(task_id, prefix, ok) if (sync and ok) else []
            return {"removed": ok, "failed": bad, "prune": pruned}
        return self._run(task_id, OP_DEDUP_APPLY, job, "删除失败：")

    def dedup_move(self, task_id: str, paths: List[str], dest: str, *,
                   base_root: Optional[str] = None,
                   prefix: Optional[str] = None,
                   sync: bool = True) -> Optional[Dict]:
        """把选中的重复图移动到目标图库（保留相对目录结构），随后按需同步索引。"""
        def job() -> Dict:
            from . import dedup as DD
            ok, bad = DD.move_paths(list(paths), dest, base_root=base_root)
            pruned = self.prune_index(task_id, prefix, ok) if (sync and ok) else []
            return {"removed": ok, "failed": bad, "prune": pruned, "dest": dest}
        return self._run(task_id, OP_DEDUP_APPLY, job, "移动失败：")

    def prune_index(self, task_id: str, prefix: Optional[str] = None,
                    removed: Optional[List[str]] = None,
                    *, sync_tiles: bool = True) -> List[Dict]:
        """删除/移动后同步索引（整图 + 瓦片）：剔除路径对应的行/瓦片。

        返回各前缀的 prune 结果（失败只记日志，不打断其余前缀）。
        """
        base = prefix or self.current_prefix()
        drop = list(removed or [])
        if not base or not drop:
            return []
        self._drop_engines("索引剔除前释放内存")
        cands = (base, self.tiles_prefix(base)) if sync_tiles else (base,)
        out: List[Dict] = []
        for p in [x for x in cands if self.meta_exists(x)]:
            try:
                r = _store_prune(p, drop)
                if r.get("removed"):
                    LOGGER.info("索引同步：%s.* 剔除 %d 条（剩 %d 条）",
                                os.path.basename(p), r["removed"], r["kept"])
                out.append(r)
            except Exception as e:                           # noqa: BLE001
                LOGGER.warning("索引同步失败（可稍后重建/增量刷新）：%s", e)
        return out

    # ==================================================================
    # 跨进程交接（img_server → 增量建库）
    # ==================================================================
    def handoff(self, task_id: str, request_path: str) -> Optional[Dict]:
        """处理交接文件：校验 request → 定位图库根 → 增量入库 → 回写 result。"""
        def job() -> Dict:
            from . import handoff as H
            p = str(request_path or "")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"交接文件不存在：{p}")
            req = H.load_request(p)
            roots = [r["path"] for r in req.get("roots", [])]
            if not roots:
                raise ValueError("请求中缺少增量根目录(roots)")
            explicit = req.get("prefix") or ""
            loc0 = None if explicit else H.locate_gallery_root(roots[0])
            prefix = explicit or (loc0["prefix"] if loc0
                                  else self.auto_prefix(roots[0]))
            LOGGER.info("【自动交接】收到 img_server 请求：%s",
                        req.get("request_id", ""))
            for r in roots:
                LOGGER.info("    增量来源: %s", r)
            if loc0 and loc0["root"] != roots[0]:
                LOGGER.info("    自动定位图库根: %s -> %s"
                            "（子目录自身无索引，增量并入上级图库索引）",
                            roots[0], loc0["root"])
            LOGGER.info("    索引前缀 : %s", prefix)
            LOGGER.info("    交接方案 : %s（顺序执行）",
                        H.mode_labels(req.get("modes")))
            LOGGER.info("开始校验并增量建库（路径+MD5 去重，重复内容自动跳过）…")
            self._progress(task_id, 0, 1, PHASE_FUSED)

            def cb(done, total, phase=None):
                # 交接可能是「整图 + 子图」两段：阶段名随 handoff 层透传
                # （它已把 save/done 边界归一成阶段名），缺省按整图段显示。
                self._progress(task_id, done, total,
                               phase if phase in (PHASE_FUSED, PHASE_TILES)
                               else PHASE_FUSED)

            result = dict(H.process_request_file(p, progress=cb))
            result["roots"] = roots          # 界面据此更新“图库目录”输入框
            return result
        return self._run(task_id, OP_HANDOFF, job, "自动交接失败：")

    # ==================================================================
    # 统计 / 结果导出 / 打开原图 / 对端程序（切换启动）
    # ==================================================================
    def stats(self, task_id: str, prefix: Optional[str] = None,
              cfg: Optional[Config] = None) -> Optional[Dict]:
        """索引统计（复用引擎缓存，避免重复加载）。"""
        cfg = cfg or self.cfg
        target = prefix or self.current_prefix()

        def job() -> Dict:
            eng, _cached = self.engine_for(cfg, target)
            return eng.stats()
        return self._run(task_id, OP_STATS, job, "读取索引统计失败：")

    @staticmethod
    def export_sheet(paths: List[str], out_path: str) -> bool:
        """把 Top-K 结果拼成总览 PNG（失败返回 False，不抛错）。"""
        from .visuals import save_contact_sheet
        return bool(save_contact_sheet(list(paths), out_path))

    @staticmethod
    def open_in_shell(path: str) -> Tuple[bool, str]:
        """用系统默认程序打开原图；返回 (成功, 错误信息)。"""
        p = str(path or "")
        if not p or not os.path.exists(p):
            return False, "路径不存在"
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)      # type: ignore[attr-defined]  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
            return True, ""
        except Exception as e:                               # noqa: BLE001
            return False, str(e)

    @staticmethod
    def peer_state() -> Dict:
        """“切换启动”目标状态（peer_launcher.check()：目标优先级 + main.py 哈希白名单）。

        目标优先级（有 main.py 就只用 main.py，否则用端口画板）由 peer_launcher 决定，
        这里原样透传给界面，界面据此改按钮文字。
        """
        import peer_launcher as PL
        st = PL.check()
        return {"state": st, "main": PL.PEER_MAIN, "name": PL.PEER_NAME,
                "target": st.get("target"),
                "target_path": st.get("target_path"),
                "target_how": st.get("target_how"),
                "target_label": PL.target_label(st.get("target")),
                "codes": {"ok": PL.ST_OK, "missing": PL.ST_MISSING,
                          "unregistered": PL.ST_UNREGISTERED,
                          "mismatch": PL.ST_MISMATCH}}

    @staticmethod
    def peer_register() -> Dict:
        """把当前目标哈希写入白名单（人工确认可信后调用）。"""
        import peer_launcher as PL
        return PL.register()

    @staticmethod
    def peer_launch():
        """启动对端进程（返回 (proc, err)）；窗口生命周期由界面负责。"""
        import peer_launcher as PL
        return PL.launch()
