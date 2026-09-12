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
  1) 读取/校验 request（schema v1）
  2) 对每个 target 根执行增量建库（复用 HybridEngine.add：路径+MD5 去重，
     幂等安全——同一批重复触发也只会把真正的新内容入库）
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

SCHEMA = 1

# ---------------------------------------------------------------------------
# request 结构（由 img_server 侧写入，字段见 docs/HANDOFF_PROTOCOL.md）
# {
#   "schema": 1, "kind": "download_batch_complete",
#   "request_id": "…", "ts": "ISO",
#   "source": "img_server",
#   "roots": [{"path": "下载根", "note": ""}],          # 至少一个
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


def validate(req: Dict) -> Dict:
    if req.get("schema") != SCHEMA:
        raise ValueError(f"不支持的交接协议 schema={req.get('schema')}（期望 {SCHEMA}）")
    if req.get("kind") != "download_batch_complete":
        raise ValueError("kind 必须是 download_batch_complete")
    roots = req.get("roots") or []
    if not roots:
        raise ValueError("roots 至少需要一个下载根目录")
    out = dict(req)
    out["roots"] = [{"path": r.get("path"), "note": r.get("note", "")}
                    for r in roots if r.get("path")]
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
        if parent == cand:                 # 到达文件系统根（如 F:\），没有宿主
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
def run_ingest(req: Dict, progress=None) -> Dict:
    """
    执行增量入库（自动校验图库位置）：
      * 未显式指定 prefix 时，对每个下载根先调 locate_gallery_root 沿
        祖先目录向上定位真正的图库根——下载根是图库根之下的子目录时，
        增量会并入上级图库根的既有索引，而不是在子目录里另建一套；
      * 整条链都没有图库索引时，才把该下载根自身当作新图库根
        （prefix = 该根/.gallery_index/gallery，与 GUI 默认一致），首次自动建库；
      * 显式指定 prefix 时尊重请求，不做定位。
      对每个（root, prefix）调 HybridEngine.add(prefix, img_dir=root)：
        * add 内部按 路径+MD5 去重——重复触发安全；
        * 需要 ResNet 全库索引时走“融合单遍解码”自动补算新图特征；
        * 索引尚不存在时（首次）自动从零构建。
    返回 result dict（随后由调用方 write_result；steps 含定位结果供审计）。
    """
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    from hybrid_search.store import IndexFiles

    roots = [r["path"] for r in req["roots"]]
    if not roots:
        raise ValueError("roots 为空")
    explicit_prefix = req.get("prefix") or ""

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
        t0 = time.time()
        try:
            cfg = Config()                       # 全默认：cv2 PNG、GPU 自动
            cfg.device = "auto"
            eng = HybridEngine(cfg)
            if prefix not in built and not IndexFiles(prefix).meta_exists():
                # 图库根索引尚不存在 -> 从零构建
                n = eng.build(prefix, img_dir=root, progress=progress)
                built.add(prefix)
                mode = "首次构建"
            else:
                n = eng.add(prefix, img_dir=root, progress=progress)
                mode = "增量"
            dt = time.time() - t0
            total_added += n
            total_secs += dt
            steps.append({"root": root, "gallery_root": gallery_root,
                          "located": located, "prefix": prefix,
                          "added": n, "mode": mode,
                          "total_in_index": eng.coarse.size,
                          "secs": round(dt, 2)})
        except Exception as e:                   # noqa: BLE001 —— 单根失败不中断其余
            dt = time.time() - t0
            errors.append({"root": root, "error": repr(e)})
            steps.append({"root": root, "gallery_root": gallery_root,
                          "located": located, "prefix": prefix,
                          "added": 0, "secs": round(dt, 2),
                          "error": repr(e)})
    return {
        "request_id": req.get("request_id", ""),
        "ok": not errors,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prefix": plans[0][1],
        "steps": steps,
        "notices": notices,
        "total_added": total_added,
        "total_secs": round(total_secs, 2),
        "errors": errors,
    }


def process_request_file(path: str, progress=None,
                         handoff_dir: Optional[str] = None) -> Dict:
    """request 文件 -> working -> 执行 -> result 文件。返回 result。
    handoff_dir 缺省 = request 文件所在目录（便于把交接目录放在任意位置）。"""
    d = handoff_dir or os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    req = load_request(path)
    req_id = req.get("request_id") or os.path.basename(path)
    working = mark_working(path, d)
    result = {"request_id": req_id, "ok": False, "steps": []}
    try:
        result = run_ingest(req, progress=progress)
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
