# -*- coding: utf-8 -*-
# ImageSearchTool · 不覆盖历史数据的可核验测试/备份归档
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""归档名的 SHA256 等于 manifest.json 实际字节的 SHA256。

python -E -B devtools/archive_batch.py --out <备份父目录> --backup docs/perf-plan.md
python -E -B devtools/archive_batch.py --out <测试父目录> --params '{"command":"..."}'
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_archive(parent, *, backup=(), parameters=None):
    from paths import GALLERY_ROOT
    parent = Path(parent).resolve()
    gallery = Path(GALLERY_ROOT).resolve()
    if parent == gallery or gallery in parent.parents:
        raise ValueError("禁止写图库归档")
    timestamp = datetime.now().astimezone()
    sources = []
    for name in backup:
        source = (ROOT / name).resolve(strict=True)
        relative = source.relative_to(ROOT)
        if not source.is_file():
            raise ValueError(f"备份目标不是文件: {source}")
        sources.append({"path": relative.as_posix(), "sha256": digest(source)})
    sources.sort(key=lambda item: item["path"])
    manifest = {"project": "image-search", "timestamp": timestamp.isoformat(),
                "kind": "backup" if sources else "test"}
    if sources:
        manifest["files"] = sources
    else:
        if not parameters:
            raise ValueError("测试归档必须包含运行参数")
        manifest["parameters"] = parameters
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
               + "\n").encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    stamp = timestamp.strftime("%Y%m%d-%H%M%S%f")[:-3]
    destination = parent / f"image-search_{stamp}_{sha}"
    # mkdir 原子拒绝同名目录；错误时保留半成品用于诊断，绝不删历史归档。
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "manifest.json").open("xb") as stream:
        stream.write(payload)
    for entry in sources:
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / entry["path"], target)
        if digest(target) != entry["sha256"] or digest(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"备份期间源文件变化或复制失败: {entry['path']}")
    if digest(destination / "manifest.json") != sha:
        raise RuntimeError("归档清单校验失败")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backup", nargs="+")
    group.add_argument("--params", type=json.loads)
    args = parser.parse_args()
    print(create_archive(args.out, backup=args.backup or (), parameters=args.params))


if __name__ == "__main__":
    main()
