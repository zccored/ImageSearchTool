# -*- coding: utf-8 -*-
"""图库分发插件：建库完成后把临时图库文件迁移到多个目标盘，并重写索引路径。

设计约束：
  * 默认关闭；只有显式传入 targets 才生效；
  * 不是核心依赖，删除本文件/关闭开关不影响建库与查询；
  * 先移动文件，全部成功后才重写索引路径；
  * 每个文件写 manifest；失败按 manifest 逆操作回滚；
  * 重写前备份旧路径数组，rollback 可恢复；
  * 目标按 `path=weight` 描述，按已分配字节/权重做简单天平。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import stat
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .io_utils import LOGGER

Progress = Optional[Callable[[int, int], None]]


@dataclasses.dataclass
class Target:
    path: str
    weight: float = 1.0
    free: int = 0
    assigned: int = 0

    def root(self) -> str:
        return os.path.abspath(os.path.expanduser(self.path))


def parse_targets(value) -> List[Target]:
    """支持 'path=weight;path2=weight2'、list、目标对象。"""
    if value is None:
        return []
    items: List[str] = []
    if isinstance(value, str):
        items = [x.strip() for x in value.replace("\n", ";").split(";") if x.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, Target):
        items = [value.path]
    else:
        items = [str(value).strip()]
    out: List[Target] = []
    for item in items:
        weight = 1.0
        path = item
        # path=weight（权重必须是数字；Windows 路径里通常没有 '='）
        if "=" in item:
            a, b = item.rsplit("=", 1)
            try:
                w = float(b)
                if w > 0:
                    path, weight = a.strip(), w
            except ValueError:
                pass
        if not path:
            continue
        path = path.strip().strip('"').strip("'")
        out.append(Target(path=path, weight=weight))
    return out


def _sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".dist.tmp.npy"
    np.save(tmp, arr, allow_pickle=True)
    os.replace(tmp, path)


def _path_arrays(prefix: str) -> List[str]:
    """所有需要重写的路径数组（full + tiles）。"""
    cands = [prefix + ".paths.npy", prefix + ".fine_paths.npy",
             prefix + "_tiles.paths.npy", prefix + "_tiles.fine_paths.npy"]
    return [p for p in cands if os.path.exists(p)]


def _load_paths(path: str) -> List[str]:
    return [str(x) for x in np.load(path, allow_pickle=True)]


def _coerce_targets(value) -> List[Target]:
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, Target) for x in value):
        return list(value)
    return parse_targets(value)


def _choose(targets: Sequence[Target], size: int) -> Target:
    # 已分配字节 / 权重越小越优先；权重高的大盘会多分
    return min(targets, key=lambda t: (t.assigned + size) / max(t.weight, 1e-9))


def _free_of(root: str) -> int:
    try:
        return int(shutil.disk_usage(root).free)
    except OSError:
        return 0


def distribute(prefix: str, source_root: str, targets,
               progress: Progress = None, verify_hash: bool = False,
               dry_run: bool = False) -> dict:
    """把索引引用的文件从 source_root 迁移到 targets，并重写路径数组。

    返回 report；失败抛出异常，已移动文件会尽量回滚。
    """
    src_root = os.path.abspath(source_root)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.dirname(os.path.abspath(prefix))
    manifest_path = os.path.join(out_dir, f"distribution_manifest_{ts}.jsonl")
    report_path = os.path.join(out_dir, f"distribution_report_{ts}.json")
    backup_dir = os.path.join(out_dir, f"distribution_backup_{ts}")
    os.makedirs(backup_dir, exist_ok=True)

    tg = _coerce_targets(targets)
    if not tg:
        raise ValueError("分发已启用但没有目标（格式：路径=权重;路径=权重）")
    for t in tg:
        root = t.root()
        os.makedirs(root, exist_ok=True)
        t.free = _free_of(root)
    # 目标不能与源重叠
    for t in tg:
        a, b = os.path.normcase(t.root()), os.path.normcase(src_root)
        if a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep):
            raise ValueError(f"分发目标与源目录重叠：{t.path}")

    old_arrays: Dict[str, List[str]] = {p: _load_paths(p) for p in _path_arrays(prefix)}
    if not old_arrays:
        raise FileNotFoundError("找不到索引路径数组（paths.npy）")
    old_paths = old_arrays[prefix + ".paths.npy"] if (prefix + ".paths.npy") in old_arrays else next(iter(old_arrays.values()))
    # 唯一文件清单（tiles 数组可能重复引用同一张图）
    unique: List[str] = []
    seen = set()
    for p in old_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    missing = [p for p in unique if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"源文件缺失 {len(missing)} 个，例如 {missing[0]}")

    mapping: Dict[str, str] = {}
    move_rows: List[dict] = []
    moved: List[Tuple[str, str]] = []
    try:
        # 备份路径数组
        for p in old_arrays:
            shutil.copy2(p, os.path.join(backup_dir, os.path.basename(p)))
        backup_meta = {"prefix": prefix, "source_root": src_root,
                       "targets": [{"path": t.path, "weight": t.weight} for t in tg],
                       "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(backup_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(backup_meta, fh, ensure_ascii=False, indent=1)

        total = len(unique)
        if progress:
            progress(0, total)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            for i, src in enumerate(unique, 1):
                rel = os.path.relpath(src, src_root)
                if rel.startswith(".."):
                    raise ValueError(f"文件不在源根下：{src}")
                size = os.path.getsize(src)
                t = _choose(tg, size)
                dst = os.path.join(t.root(), rel)
                if os.path.exists(dst):
                    raise FileExistsError(f"目标已存在，拒绝覆盖：{dst}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "source": src, "dest": dst, "size": size,
                       "target": t.path, "status": "moving"}
                if dry_run:
                    row["status"] = "planned"
                    mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    mapping[src] = dst
                    t.assigned += size
                    if progress:
                        progress(i, total)
                    continue
                h1 = _sha256(src) if verify_hash else ""
                try:
                    shutil.copy2(src, dst)
                    if os.path.getsize(dst) != size:
                        raise IOError(f"复制后大小不一致：{dst}")
                    h2 = _sha256(dst) if verify_hash else ""
                    if verify_hash and h1 != h2:
                        raise IOError(f"SHA256 不一致：{dst}")
                    # 去掉只读，再删源
                    try:
                        os.chmod(src, stat.S_IWRITE | stat.S_IREAD)
                    except OSError:
                        pass
                    os.remove(src)
                except Exception:
                    # 本文件复制失败：清理目标，整体失败
                    try:
                        if os.path.exists(dst):
                            os.remove(dst)
                    except OSError:
                        pass
                    raise
                row.update({"status": "moved", "sha256": h1 if verify_hash else ""})
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                mf.flush()
                mapping[src] = dst
                moved.append((src, dst))
                t.assigned += size
                if progress:
                    progress(i, total)
                if i % 128 == 0:
                    LOGGER.info("分发进度 %d/%d（%.1f MB）", i, total, t.assigned / 1e6)

        if dry_run:
            report = {"dry_run": True, "files": len(unique),
                      "targets": [{"path": t.path, "weight": t.weight,
                                   "assigned": t.assigned} for t in tg],
                      "manifest": manifest_path}
            return report

        # 全部移动成功后才重写索引
        new_arrays = {}
        for p, arr in old_arrays.items():
            new_arrays[p] = [mapping.get(x, x) for x in arr]
        for p, new in new_arrays.items():
            _atomic_npy(p, np.array(new, dtype=object))
        # 校验新路径都存在
        for p, new in new_arrays.items():
            miss = [x for x in new if x and not os.path.exists(x)]
            if miss:
                raise FileNotFoundError(f"重写后仍缺失 {len(miss)} 个：{miss[0]}")

        report = {
            "ok": True,
            "files": len(unique),
            "moved": len(moved),
            "source_root": src_root,
            "targets": [{"path": t.path, "weight": t.weight,
                         "assigned": t.assigned, "free_before": t.free} for t in tg],
            "manifest": manifest_path,
            "backup_dir": backup_dir,
            "arrays": {os.path.basename(k): len(v) for k, v in new_arrays.items()},
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        return report
    except Exception as e:
        LOGGER.error("分发失败，开始回滚：%r", e)
        # 逆序把已移动文件搬回
        for src, dst in reversed(moved):
            try:
                if os.path.exists(dst) and not os.path.exists(src):
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dst, src)
            except Exception as re:
                LOGGER.error("回滚移动失败 %s -> %s: %r", dst, src, re)
        # 路径数组只在最后成功后才重写；若已写则从备份恢复
        try:
            for p in old_arrays:
                bak = os.path.join(backup_dir, os.path.basename(p))
                if os.path.exists(bak):
                    shutil.copy2(bak, p)
        except Exception as re:                            # noqa: BLE001
            LOGGER.error("回滚路径数组失败：%r", re)
        raise


def rollback(manifest_path: str, restore_backup: bool = True) -> dict:
    """按 manifest 逆操作：目标搬回源；若备份存在则恢复路径数组。"""
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    moved = 0
    for r in reversed(rows):
        if r.get("status") != "moved":
            continue
        src, dst = r["source"], r["dest"]
        if os.path.exists(dst) and not os.path.exists(src):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            moved += 1
    # backup 目录从 manifest 同目录的 distribution_backup_* 找最近
    if restore_backup:
        d = os.path.dirname(os.path.abspath(manifest_path))
        baks = sorted([x for x in os.listdir(d)
                       if x.startswith("distribution_backup_")], reverse=True)
        if baks:
            bdir = os.path.join(d, baks[0])
            for f in os.listdir(bdir):
                if f.endswith(".npy"):
                    # 文件名即原数组名，目标在 prefix 同目录；manifest 没有 prefix，
                    # 由调用方给出（这里只恢复同目录同名文件）
                    dst = os.path.join(d, f)
                    if os.path.exists(dst):
                        shutil.copy2(os.path.join(bdir, f), dst)
    return {"moved_back": moved, "manifest": manifest_path}
