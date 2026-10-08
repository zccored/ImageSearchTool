# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 跨进程交接协议接收端：request 校验 → 图库根自动定位 → 增量入库 → result 回写
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
handoff —— 跨进程“下载完成 → 增量建库”交接协议（接收端实现）。

配合 img_server（下载方）按钮：下载+哈希校验完成后写一份 request JSON，
然后退出；本模块负责：
  1) 读取/校验 request（schema v1/v2；v2 新增 modes：整图/子图选做或都做）
  2) 对每个 target 根执行增量建库（整图复用 HybridEngine.add：路径+MD5 去重，
     幂等安全——同一批重复触发也只会把真正的新内容入库；子图走 tile_index：
     按路径去重，瓦片索引不存在时首次切块构建）
  3) 把结果写回 result JSON（供 img_server / 用户审计）
  4) 启动器（handoff_launcher.py）等待 img_server 退出后再调用本模块

不变量：
  * 不改动任何图库文件（只读扫描 + 索引写索引目录）；
  * 不常驻、不影响 image-search 正常打开/处理流程（只在显式调用时生效）；
  * 交接文件流转：request_*.json →(处理中)→ working_*.json → 完成后
    写 result_*.json 并删除 working_（防重入）。
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional

# 默认交接目录（与 img_server 项目同级共享）：<上级>/handoff/
DEFAULT_HANDOFF_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "handoff"))

SCHEMA = 2                      # 当前协议版本（写入方用）
SUPPORTED_SCHEMAS = (1, 2)      # 兼容读取：v1 无 modes 字段，按“两个都做”处理

# 交接方案（顺序即执行顺序：先整图，后子图）
MODES = ("full", "tiles")
DEFAULT_MODES = ("full", "tiles")
MODE_LABELS = {"full": "整图增量", "tiles": "子图(瓦片)增量"}

# 进度阶段名（与 service.PHASE_* 的字符串协议一致；此处不 import service，避免循环依赖）
PHASE_FUSED = "fused"
PHASE_TILES = "tiles"
PHASE_SAVE = "save"             # 阶段边界：特征提取完成、开始写盘
PHASE_DONE = "done"             # 阶段边界：该阶段全部落盘完成
_BOUNDARY_PHASES = (PHASE_SAVE, PHASE_DONE)

# ---------------------------------------------------------------------------
# request 结构（由 img_server 侧写入，字段见 docs/HANDOFF_PROTOCOL.md）
# {
#   "schema": 2, "kind": "download_batch_complete",
#   "request_id": "…", "ts": "ISO",
#   "source": "img_server",
#   "roots": [{"path": "下载根", "note": ""}],          # 至少一个
#   "modes": ["full", "tiles"],                         # 可选：至少一个，顺序固定
#   "prefix": "可选；缺省按图库根自动定位（见 locate_gallery_root）",
#   "open_mode": "gui" | "cli",
#   "expect_exit": {"pids": [], "names": []},           # 由 launcher 使用
#   "note": ""
# }
# ---------------------------------------------------------------------------


def _default_handoff_dir() -> str:
    env = os.environ.get("IMG_HANDOFF_DIR")
    return env if env else DEFAULT_HANDOFF_DIR


def list_requests(handoff_dir: Optional[str] = None) -> List[str]:
    d = handoff_dir or _default_handoff_dir()
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "request_*.json")))


def load_request(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        req = json.load(f)
    return validate(req)


def normalize_modes(modes) -> List[str]:
    """
    归一化交接方案（request.modes）：
      * 缺省 / None / 空数组 —— 按默认“两个都做一遍”处理（兼容 schema v1 请求）；
      * 支持中英文别名（full/whole/image/整图、tiles/tile/子图/瓦片）；
      * 去重，并强制固定执行顺序：先 full（整图），后 tiles（子图）。
    非法取值直接报错（宁可让用户看到失败，也不要静默少做一步）。
    """
    if modes is None:
        return list(DEFAULT_MODES)
    if isinstance(modes, str):
        modes = [modes]
    if not isinstance(modes, (list, tuple)):
        raise ValueError(f"modes 必须是数组（如 [\"full\", \"tiles\"]），"
                         f"收到 {type(modes).__name__}")
    alias = {"full": "full", "whole": "full", "image": "full", "整图": "full",
             "tiles": "tiles", "tile": "tiles", "子图": "tiles", "瓦片": "tiles"}
    picked = []
    for m in modes:
        key = alias.get(str(m).strip().lower())
        if key is None:
            raise ValueError(f"modes 含未知取值 {m!r}（只能是 full=整图 / tiles=子图）")
        if key not in picked:
            picked.append(key)
    if not picked:                      # 空数组：视同未指定，两个都做
        return list(DEFAULT_MODES)
    return [m for m in MODES if m in picked]


def mode_labels(modes) -> str:
    """把 modes 渲染成中文串，用于日志/审计（如「整图增量 + 子图(瓦片)增量」）。"""
    return " + ".join(MODE_LABELS.get(m, m) for m in normalize_modes(modes))


def validate(req: Dict) -> Dict:
    if req.get("schema") not in SUPPORTED_SCHEMAS:
        raise ValueError(f"不支持的交接协议 schema={req.get('schema')}"
                         f"（期望 {SCHEMA}，兼容 {SUPPORTED_SCHEMAS[0]}）")
    if req.get("kind") != "download_batch_complete":
        raise ValueError("kind 必须是 download_batch_complete")
    roots = req.get("roots") or []
    if not roots:
        raise ValueError("roots 至少需要一个下载根目录")
    out = dict(req)
    out["roots"] = [{"path": r.get("path"), "note": r.get("note", "")}
                    for r in roots if r.get("path")]
    out["modes"] = normalize_modes(req.get("modes"))
    out.setdefault("open_mode", "gui")
    out.setdefault("expect_exit", {"pids": [], "names": []})
    return out


def locate_gallery_root(path: str) -> Optional[Dict]:
    """
    自动校验/定位“真正的图库根目录”：
      图库索引的规范位置是 <图库根>/.gallery_index/<前缀名>.*；
      img_server 提交的下载根可能是图库根自身，也可能是图库根之下的
      某个子目录/子图集（此时该子目录下没有索引）。
    规则：沿 path 自身 → 各级父目录向上逐级检查，返回最近一个
    含本程序索引（<dir>/.gallery_index/*.meta.json）的目录，即图库根宿主。
    命中返回 {"root": <宿主目录>, "prefix": <宿主实际索引前缀>}，
    前缀名取宿主 .gallery_index 下真实存在的索引名（兼容自定义前缀）；
    整条祖先链都找不到时返回 None（调用方应把 path 自身当作新图库根）。
    """
    cand = os.path.abspath(path)
    while True:
        idx_dir = os.path.join(cand, ".gallery_index")
        if os.path.isdir(idx_dir):
            metas = sorted(glob.glob(os.path.join(idx_dir, "*.meta.json")))
            if metas:
                base = os.path.basename(metas[0])
                name = (base[:-len(".meta.json")]
                        if base.endswith(".meta.json") else base)
                return {"root": cand, "prefix": os.path.join(idx_dir, name)}
        parent = os.path.dirname(cand)
        if parent == cand:                 # 到达文件系统根（如 <盘符>:\），没有宿主
            return None
        cand = parent


def mark_working(req_path: str, handoff_dir: Optional[str] = None) -> str:
    """request_*.json -> working_*.json（防止双实例重复处理）。
    若传入的已是 working_*.json（GUI 路径由 launcher 转好）则原样返回。"""
    d = handoff_dir or _default_handoff_dir()
    name = os.path.basename(req_path)
    if name.startswith("working_"):
        return req_path
    dst = os.path.join(d, name.replace("request_", "working_", 1))
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(req_path, dst)
    return dst


def working_files(handoff_dir: Optional[str] = None) -> List[str]:
    d = handoff_dir or _default_handoff_dir()
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "working_*.json")))


def result_path_for(req_id: str, handoff_dir: Optional[str] = None) -> str:
    d = handoff_dir or _default_handoff_dir()
    return os.path.join(d, f"result_{req_id}.json")


def write_result(req_id: str, result: Dict,
                 handoff_dir: Optional[str] = None) -> str:
    d = handoff_dir or _default_handoff_dir()
    os.makedirs(d, exist_ok=True)
    fp = result_path_for(req_id, d)
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fp)
    return fp


# ---------------------------------------------------------------------------
# 增量建库执行体（CLI 与 GUI 自动模式共用）
# ---------------------------------------------------------------------------
def _stage_progress(progress, default_phase: str):
    """进度回调适配器：内部各层既有两参 progress(done,total) 也有三参
    progress(done,total,phase)。统一包一层：
      * 缺省阶段（None）用 default_phase 补上；
      * 阶段边界 save/done 也归到 default_phase——交接是「整图→子图」两个
        阶段连做，若把边界的 “done” 透传出去，界面会在整图做完时就显示
        “全部完成”，与后面还有子图阶段矛盾；
      * 其它阶段（如瓦片自己的 tiles）原样透传。"""
    if progress is None:
        return None

    def cb(done, total, phase=None):
        progress(done, total,
                 default_phase if (not phase or phase in _BOUNDARY_PHASES)
                 else phase)

    return cb


def run_ingest(req: Dict, progress=None, modes=None) -> Dict:
    """
    执行交接入库（自动校验图库位置）。按 request.modes 顺序把选定方案各做一遍，
    缺省（或 schema v1 老请求）为“整图 + 子图”两个都做：

      ① full —— 整图（粗筛 + ResNet）增量：复用 HybridEngine.add，内部按
         路径+MD5 去重，重复触发安全；索引尚不存在时自动从零构建；
      ② tiles —— 子图（512px 瓦片）增量：瓦片索引写在与整图索引同目录的
         gallery_tiles.*；已有瓦片索引走 add_tiles（按路径去重），没有则
         build_tiles 首次切块构建（耗时较长，属正常）。

    图库定位（每个下载根各判一次）：
      * 未显式指定 prefix 时，先 locate_gallery_root 沿祖先目录向上定位真正的
        图库根——下载根是图库根之下的子目录时，增量并入上级既有索引，
        而不是在子目录里另建一套；
      * 整条链都没有图库索引时，把该下载根自身当作新图库根
        （prefix = <根>/.gallery_index/gallery，与 GUI 默认一致）；
      * 显式指定 prefix 时尊重请求，不做定位。
    单根 / 单方案失败只记录到 errors，不中断其余（两个方案互不牵连）。
    返回 result dict（随后由调用方 write_result；steps 含定位与各阶段明细供审计）。
    """
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    from hybrid_search.store import IndexFiles
    from hybrid_search import tile_index as TI

    roots = [r["path"] for r in req["roots"]]
    if not roots:
        raise ValueError("roots 为空")
    explicit_prefix = req.get("prefix") or ""
    modes = normalize_modes(modes if modes is not None else req.get("modes"))

    # 计划表：(root, prefix, gallery_root, located)
    #   located=True  —— 由祖先链定位到的图库根（请求根是它的子目录）
    #   located=False —— 该根自身即图库根（显式 prefix 或首次建库）
    plans = []
    for r in roots:
        if explicit_prefix:
            plans.append((r, explicit_prefix, r, False))
        else:
            loc = locate_gallery_root(r)
            if loc:
                plans.append((r, loc["prefix"], loc["root"], True))
            else:
                plans.append((r, os.path.join(r, ".gallery_index", "gallery"),
                              r, False))

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    steps = []
    total_added = 0
    total_tiles_added = 0
    total_secs = 0.0
    errors = []
    notices = []                     # 非致命提示（如：请求根自身有索引、上级另有图库宿主）
    built = set()                    # 本批次内已从零建过库的 prefix
    for root, prefix, gallery_root, located in plans:
        # 请求根自身就是宿主、但其上级链还存在其它图库宿主时提示——
        # 通常是历史错位建库（下载根被当成了图库根），提醒用户归并/清理。
        if (not explicit_prefix and located
                and os.path.normcase(gallery_root) == os.path.normcase(root)):
            parent_loc = locate_gallery_root(os.path.dirname(root))
            if parent_loc:
                notices.append(
                    f"{root} 自身已有图库索引，且上级 {parent_loc['root']} "
                    f"也存在图库索引；若前者是历史错位产物，删除 "
                    f"{root}/.gallery_index 后重试将自动并入上级图库索引")
        t_root0 = time.time()
        # step 保留整图字段（added/mode/total_in_index/secs）以兼容旧调用方；
        # 新增 stages（逐方案明细）与 tiles_* 汇总。
        step = {"root": root, "gallery_root": gallery_root,
                "located": located, "prefix": prefix,
                "modes": list(modes), "stages": [], "added": 0, "tiles_added": 0}

        # ---- ① 整图增量（粗筛 + ResNet 融合，路径+MD5 去重） ----
        if "full" in modes:
            t0 = time.time()
            st = {"mode": "full", "label": MODE_LABELS["full"],
                  "prefix": prefix, "added": 0}
            try:
                cfg = Config()                   # 全默认：PNG 走 libdeflate 旁路、GPU 自动
                from .runtime import opencv_thread_policy
                policy = opencv_thread_policy()
                if policy is not None:
                    cfg.opencv_threads = policy
                cfg.device = "auto"
                eng = HybridEngine(cfg)
                cb = _stage_progress(progress, PHASE_FUSED)
                if prefix not in built and not IndexFiles(prefix).meta_exists():
                    # 图库根索引尚不存在 -> 从零构建
                    n = eng.build(prefix, img_dir=root, progress=cb)
                    built.add(prefix)
                    mode = "首次构建"
                else:
                    n = eng.add(prefix, img_dir=root, progress=cb)
                    mode = "增量"
                dt = time.time() - t0
                total_added += n
                total_secs += dt
                st.update(added=n, build_mode=mode, secs=round(dt, 2),
                          total_in_index=eng.coarse.size)
                step.update(added=n, mode=mode, secs=round(dt, 2),
                            total_in_index=eng.coarse.size)
            except Exception as e:               # noqa: BLE001 —— 单方案失败不中断其余
                dt = time.time() - t0
                errors.append({"root": root, "mode": "full", "error": repr(e)})
                st.update(error=repr(e), secs=round(dt, 2))
            step["stages"].append(st)

        # ---- ② 子图（瓦片）增量：写 <图库根>/.gallery_index/gallery_tiles.* ----
        if "tiles" in modes:
            t0 = time.time()
            tp = TI.tiles_prefix_of(prefix)
            st = {"mode": "tiles", "label": MODE_LABELS["tiles"],
                  "prefix": tp, "added": 0}
            try:
                cfg = Config()
                from .runtime import opencv_thread_policy
                policy = opencv_thread_policy()
                if policy is not None:
                    cfg.opencv_threads = policy
                cfg.device = "auto"
                eng = HybridEngine(cfg)
                cb = _stage_progress(progress, PHASE_TILES)
                if IndexFiles(tp).meta_exists():
                    n = TI.add_tiles(eng, tp, img_dir=root, progress=cb)
                    tmode = "增量"
                else:
                    notices.append(
                        f"{root} 尚无子图索引，本次首次切块构建（512px）：{tp}")
                    n = TI.build_tiles(eng, tp, img_dir=root, progress=cb)
                    tmode = "首次构建"
                dt = time.time() - t0
                total_tiles_added += n
                total_secs += dt
                st.update(added=n, build_mode=tmode, secs=round(dt, 2),
                          total_tiles=eng.coarse.size)
                step.update(tiles_added=n, tiles_mode=tmode,
                            tiles_prefix=tp, tiles_secs=round(dt, 2))
            except Exception as e:               # noqa: BLE001
                dt = time.time() - t0
                errors.append({"root": root, "mode": "tiles", "error": repr(e)})
                st.update(error=repr(e), secs=round(dt, 2))
            step["stages"].append(st)

        step["secs"] = round(time.time() - t_root0, 2)
        steps.append(step)
    return {
        "request_id": req.get("request_id", ""),
        "ok": not errors,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prefix": plans[0][1],
        "modes": list(modes),
        "steps": steps,
        "notices": notices,
        "total_added": total_added,
        "total_tiles_added": total_tiles_added,
        "total_secs": round(total_secs, 2),
        "errors": errors,
    }


def process_request_file(path: str, progress=None,
                         handoff_dir: Optional[str] = None,
                         modes=None) -> Dict:
    """request 文件 -> working -> 执行 -> result 文件。返回 result。
    handoff_dir 缺省 = request 文件所在目录（便于把交接目录放在任意位置）。
    modes 只在需要命令行覆盖请求里的方案时传（None = 用 request.modes）。"""
    d = handoff_dir or os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    req = load_request(path)
    req_id = req.get("request_id") or os.path.basename(path)
    working = mark_working(path, d)
    result = {"request_id": req_id, "ok": False, "steps": []}
    try:
        result = run_ingest(req, progress=progress, modes=modes)
        result["ok"] = result["ok"] and not result["errors"]
    except Exception as e:                      # noqa: BLE001
        result["ok"] = False
        result["fatal_error"] = repr(e)
    write_result(req_id, result, d)
    try:
        os.remove(working)
    except OSError:
        pass
    return result
