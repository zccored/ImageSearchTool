# -*- coding: utf-8 -*-
"""图库联邦/链式索引插件（默认关闭，不属于主进程核心）。

解决的问题：
  * 临时图库在 C 建索引后，图片被分发到多个盘；
  * 每个分发盘根目录可放一个 node marker，记录自己的索引文件、区域、
    断点/分发信息，并链到其它节点；
  * 从任意分发目录开始，可以递归发现同一条链上的全部图库并合并扫描；
  * 从总索引目录开始，可以一次校验全部节点；缺失/不可读的节点自动断开其区域；
  * 索引文件在 marker 中有大小、SHA256、行数和断点信息，可追溯。

文件：
  <分发根>/.image-search-federation.json      node marker
  <总索引目录>/federation.json                 total manifest
  <总索引目录>/federation_trace.json           索引文件追溯表

设计约束：
  * 默认 federation=False，关闭时所有函数不参与主流程；
  * 只做文件/manifest 层，不改特征格式，不重新计算特征；
  * 节点不可用只过滤该节点区域，不抛异常中断整个查询。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .io_utils import LOGGER

MARKER_NAME = ".image-search-federation.json"
TOTAL_NAME = "federation.json"
TRACE_NAME = "federation_trace.json"
VERSION = 1
INDEX_SUFFIXES = (
    ".paths.npy", ".fine_paths.npy", ".md5s.npy", ".hu.npy", ".fp.npy",
    ".coarse.npz", ".fine.npz", ".meta.json",
    "_tiles.paths.npy", "_tiles.fine_paths.npy", "_tiles.meta.json",
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _rows(path: str) -> int:
    try:
        if path.endswith(".npy"):
            return int(np.load(path, allow_pickle=True, mmap_mode="r").shape[0])
    except Exception:
        pass
    return 0


def _resolve(p: str, base: str) -> str:
    if not p:
        return ""
    p = os.path.expanduser(str(p))
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(base, p))


@dataclasses.dataclass
class Node:
    node_id: str
    root: str
    marker: str
    index_prefix: str = ""
    root_id: str = ""
    available: bool = False
    index_files: List[dict] = dataclasses.field(default_factory=list)
    resume: dict = dataclasses.field(default_factory=dict)
    linked: List[str] = dataclasses.field(default_factory=list)
    region_bytes: int = 0
    raw: dict = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id, "root": self.root, "marker": self.marker,
            "index_prefix": self.index_prefix, "root_id": self.root_id,
            "available": self.available, "index_files": self.index_files,
            "resume": self.resume, "linked": self.linked,
            "region_bytes": self.region_bytes,
        }


def read_marker(path_or_root: str) -> Optional[dict]:
    p = path_or_root
    if os.path.isdir(p):
        p = os.path.join(p, MARKER_NAME)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:                                  # noqa: BLE001
        LOGGER.warning("联邦 marker 读取失败 %s: %r", p, e)
        return None


def _node_from_marker(marker_path: str, raw: dict) -> Node:
    base = os.path.dirname(marker_path)
    root = _resolve(raw.get("root", base), base)
    idx = _resolve(raw.get("index_prefix", ""), base)
    return Node(
        node_id=str(raw.get("node_id") or os.path.basename(root) or "node"),
        root=root, marker=os.path.abspath(marker_path),
        index_prefix=idx, root_id=str(raw.get("root_id") or ""),
        available=os.path.isdir(root),
        index_files=list(raw.get("index_files") or []),
        resume=dict(raw.get("resume") or {}),
        linked=list(raw.get("linked_nodes") or []),
        region_bytes=int(raw.get("region_bytes") or 0), raw=raw,
    )


def discover(start: str, max_nodes: int = 256) -> List[Node]:
    """从目录/marker 开始，沿 linked_nodes / chain 递归发现节点。"""
    queue: List[str] = []
    if os.path.isfile(start):
        queue.append(os.path.abspath(start))
    else:
        m = os.path.join(os.path.abspath(start), MARKER_NAME)
        if os.path.isfile(m):
            queue.append(m)
        else:
            total = os.path.join(os.path.abspath(start), TOTAL_NAME)
            if os.path.isfile(total):
                try:
                    with open(total, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    for n in data.get("nodes", []):
                        ref = n.get("marker") if isinstance(n, dict) else n
                        if ref:
                            queue.append(_resolve(ref, os.path.dirname(total)))
                except Exception:
                    pass
    seen: set = set()
    out: List[Node] = []
    while queue and len(out) < max_nodes:
        mp = os.path.abspath(queue.pop(0))
        if os.path.normcase(mp) in seen:
            continue
        seen.add(os.path.normcase(mp))
        raw = read_marker(mp)
        if not raw:
            continue
        node = _node_from_marker(mp, raw)
        out.append(node)
        refs = list(raw.get("linked_nodes") or [])
        if raw.get("chain"):
            refs.extend(raw["chain"] if isinstance(raw["chain"], list) else [raw["chain"]])
        for ref in refs:
            if isinstance(ref, dict):
                ref = ref.get("marker") or ref.get("root")
            if ref:
                queue.append(_resolve(str(ref), os.path.dirname(mp)))
    return out


def scan_roots(start: str) -> List[str]:
    """扫描用：有 marker 时返回链上所有存在节点；否则原样返回。"""
    nodes = discover(start)
    if not nodes:
        return [os.path.abspath(start)]
    roots = []
    for n in nodes:
        if n.available and n.root not in roots:
            roots.append(n.root)
    return roots or [os.path.abspath(start)]


def _index_files(index_prefix: str) -> List[dict]:
    rows = []
    for suffix in INDEX_SUFFIXES:
        p = os.path.abspath(index_prefix + suffix)
        if not os.path.isfile(p):
            continue
        try:
            st = os.stat(p)
            rows.append({
                "path": p, "suffix": suffix, "bytes": int(st.st_size),
                "rows": _rows(p), "sha256": _sha256(p),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(st.st_mtime)),
            })
        except Exception as e:                              # noqa: BLE001
            rows.append({"path": p, "suffix": suffix, "error": repr(e)})
    return rows


def write_node_marker(node_root: str, index_prefix: str, *,
                      root_id: Optional[str] = None,
                      node_id: Optional[str] = None,
                      linked_nodes: Optional[Sequence[str]] = None,
                      resume: Optional[dict] = None,
                      region_bytes: int = 0) -> dict:
    root = os.path.abspath(node_root)
    os.makedirs(root, exist_ok=True)
    root_id = root_id or ("fed-" + uuid.uuid4().hex[:12])
    node_id = node_id or ("node-" + uuid.uuid4().hex[:8])
    marker = {
        "version": VERSION,
        "root_id": root_id,
        "node_id": node_id,
        "root": root,
        "index_prefix": os.path.abspath(index_prefix),
        "index_files": _index_files(index_prefix),
        "linked_nodes": [os.path.abspath(x) for x in (linked_nodes or [])],
        "region_bytes": int(region_bytes),
        "resume": dict(resume or {}),
        "updated_at": _now(),
    }
    mp = os.path.join(root, MARKER_NAME)
    tmp = mp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(marker, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, mp)
    return marker


def write_total(total_dir: str, index_prefix: str, nodes: Sequence[Node],
                root_id: Optional[str] = None) -> dict:
    os.makedirs(total_dir, exist_ok=True)
    rid = root_id or next((n.root_id for n in nodes if n.root_id), "") \
        or ("fed-" + uuid.uuid4().hex[:12])
    total = {
        "version": VERSION,
        "root_id": rid,
        "total_dir": os.path.abspath(total_dir),
        "total_index_prefix": os.path.abspath(index_prefix),
        "nodes": [{"node_id": n.node_id, "root": n.root,
                   "marker": n.marker, "available": n.available,
                   "region_bytes": n.region_bytes}
                  for n in nodes],
        "generated_at": _now(),
    }
    p = os.path.join(total_dir, TOTAL_NAME)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(total, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    # 索引文件追溯表：每个索引文件来自哪个节点 / 断点信息
    trace = {"root_id": rid, "total_index_prefix": os.path.abspath(index_prefix),
             "generated_at": _now(),
             "nodes": [{"node_id": n.node_id, "root": n.root,
                        "available": n.available,
                        "index_files": n.index_files,
                        "resume": n.resume} for n in nodes]}
    tp = os.path.join(total_dir, TRACE_NAME)
    with open(tp + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(trace, fh, ensure_ascii=False, indent=1)
    os.replace(tp + ".tmp", tp)
    return total


def load_total(total_dir: str) -> List[Node]:
    p = os.path.join(os.path.abspath(total_dir), TOTAL_NAME)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    out = []
    for n in data.get("nodes", []):
        mp = _resolve(n.get("marker", ""), os.path.dirname(p))
        raw = read_marker(mp)
        if raw:
            out.append(_node_from_marker(mp, raw))
        else:
            out.append(Node(node_id=str(n.get("node_id") or "missing"),
                            root=_resolve(n.get("root", ""), os.path.dirname(p)),
                            marker=mp, root_id=str(data.get("root_id") or ""),
                            available=False))
    return out


def publish_after_distribution(index_prefix: str, total_dir: str,
                               dist_report: dict, *,
                               node_id: str = "",
                               linked_extra: Optional[Sequence[str]] = None) -> dict:
    """分发完成后写每个目标盘的 node marker + 总 manifest。"""
    total_dir = os.path.abspath(total_dir or os.path.dirname(index_prefix))
    targets = [x for x in (dist_report.get("targets") or []) if x.get("path")]
    if not targets:
        return {"ok": False, "reason": "no targets"}
    root_id = "fed-" + uuid.uuid4().hex[:12]
    markers = []
    # 先算出 marker 路径，便于互相 linked（链式）
    for i, t in enumerate(targets):
        root = os.path.abspath(t["path"])
        markers.append(os.path.join(root, MARKER_NAME))
    nodes: List[Node] = []
    for i, t in enumerate(targets):
        root = os.path.abspath(t["path"])
        nid = (node_id + "-" + str(i)) if node_id else f"node-{i}-{os.path.basename(root)}"
        linked = [m for j, m in enumerate(markers) if j != i] + list(linked_extra or [])
        resume = {
            "kind": "distribution",
            "source_root": dist_report.get("source_root", ""),
            "target": root,
            "files": dist_report.get("files", 0),
            "moved": dist_report.get("moved", 0),
            "manifest": dist_report.get("manifest", ""),
            "breakpoint": {
                "done": int(dist_report.get("moved", 0)),
                "total": int(dist_report.get("files", 0)),
                "updated_at": _now(),
            },
        }
        raw = write_node_marker(root, index_prefix, root_id=root_id,
                                node_id=nid, linked_nodes=linked,
                                resume=resume,
                                region_bytes=int(t.get("assigned") or 0))
        nodes.append(_node_from_marker(os.path.join(root, MARKER_NAME), raw))
    total = write_total(total_dir, index_prefix, nodes, root_id=root_id)
    return {"ok": True, "total": total, "nodes": [n.as_dict() for n in nodes],
            "total_dir": total_dir}


def filter_hits(total_or_start: str, prefix: str, hits: Iterable[Tuple]) -> Tuple[List[Tuple], int]:
    """过滤掉路径已不存在的命中；hits 元组第 1 项为 path。"""
    out, dropped = [], 0
    for h in hits:
        try:
            p = str(h[1])
        except Exception:
            out.append(h)
            continue
        if p and os.path.exists(p):
            out.append(h)
        else:
            dropped += 1
    return out, dropped


def verify(total_dir: str) -> dict:
    nodes = load_total(total_dir)
    if not nodes:
        nodes = discover(total_dir)
    report = {"total_dir": os.path.abspath(total_dir), "nodes": [],
              "ok": True}
    for n in nodes:
        item = {"node_id": n.node_id, "root": n.root, "marker": n.marker,
                "available": n.available,
                "index_files": len(n.index_files),
                "missing_index_files": 0, "resume": n.resume}
        for f in n.index_files:
            p = f.get("path")
            if p and not os.path.isfile(p):
                item["missing_index_files"] += 1
        if not n.available or item["missing_index_files"]:
            report["ok"] = False
        report["nodes"].append(item)
    return report
