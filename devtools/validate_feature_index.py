# -*- coding: utf-8 -*-
r"""完整图库特征校验（现有 full 索引 / 未来断点检查点共用）。

用法：
    python devtools/validate_feature_index.py --prefix F:\视频\.gallery_index\gallery \
        [--json out.json]

校验项：
  * fine / paths / fine_paths / hu / fp / md5s 存在且形状匹配；
  * paths 与 fine_paths 逐项一致，且无重复路径；
  * 特征维度一致、float32、无 NaN/Inf、无零行；
  * 每行 L2 范数 ≈ 1（建库时已归一化）；
  * hu 可有限值，fp 为定长字节串，md5 无重复；
  * 输出 JSON 报告，失败时返回非零退出码。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load(prefix: str, suffix: str, mmap: bool = False):
    p = prefix + suffix
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p, mmap_mode="r" if mmap else None, allow_pickle=True)


def validate(prefix: str, norm_tol: float = 2e-4) -> dict:
    report = {"prefix": prefix, "ok": False, "errors": [], "warnings": [], "counts": {}}
    try:
        fine = _load(prefix, ".fine.npy", mmap=True)
        paths = _load(prefix, ".paths.npy")
        fine_paths = _load(prefix, ".fine_paths.npy")
        hu = _load(prefix, ".hu.npy", mmap=True)
        fp = _load(prefix, ".fp.npy", mmap=True)
        md5s = _load(prefix, ".md5s.npy")
    except Exception as e:                                # noqa: BLE001
        report["errors"].append(f"load: {e!r}")
        return report

    paths = [str(x) for x in paths]
    fine_paths = [str(x) for x in fine_paths]
    md5s = [str(x) for x in md5s]
    n = len(paths)
    report["counts"] = {"n": n, "feature_dim": int(fine.shape[1]) if fine.ndim == 2 else None,
                        "hu_dim": int(hu.shape[1]) if hu.ndim == 2 else None,
                        "fp_bytes": int(fp.shape[1]) if fp.ndim == 2 else None}

    if fine.ndim != 2:
        report["errors"].append(f"fine 维度不是 2D: {fine.shape}")
    if len(fine_paths) != n:
        report["errors"].append(f"fine_paths={len(fine_paths)} != paths={n}")
    if paths != fine_paths:
        report["errors"].append("paths 与 fine_paths 顺序/内容不一致")
    if len(set(paths)) != n:
        report["errors"].append(f"重复路径 {n - len(set(paths))} 条")
    if len(md5s) == n and len(set(md5s)) != len(md5s):
        report["warnings"].append(f"重复 md5 {len(md5s) - len(set(md5s))} 条（可能是同内容不同路径）")
    if hu.ndim != 2 or hu.shape[0] != n:
        report["errors"].append(f"hu 形状不匹配: {hu.shape}")
    else:
        try:
            if not np.isfinite(np.asarray(hu)).all():
                report["errors"].append("hu 含 NaN/Inf")
        except Exception as e:                            # noqa: BLE001
            report["errors"].append(f"hu 检查失败: {e!r}")
    if fp.ndim != 2 or fp.shape[0] != n:
        report["errors"].append(f"fp 形状不匹配: {fp.shape}")

    # 分块检查 fine，避免一次性占用内存
    bad_chunks = 0
    zero_rows = 0
    norm_min, norm_max = float("inf"), float("-inf")
    step = 2048
    for i in range(0, n, step):
        x = np.asarray(fine[i:i + step], dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != fine.shape[1]:
            report["errors"].append(f"fine 第 {i} 行块形状异常 {x.shape}")
            bad_chunks += 1
            continue
        if not np.isfinite(x).all():
            bad_chunks += 1
        norms = np.linalg.norm(x, axis=1)
        zero_rows += int((norms < 1e-6).sum())
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        if abs(float(norms.mean()) - 1.0) > norm_tol:
            report["warnings"].append(f"第 {i} 行块平均范数 {norms.mean():.9f} 偏离 1")

    report["counts"].update({"bad_chunks": bad_chunks, "zero_rows": zero_rows,
                             "norm_min": norm_min, "norm_max": norm_max})
    if bad_chunks:
        report["errors"].append(f"{bad_chunks} 个行块含 NaN/Inf")
    if zero_rows:
        report["errors"].append(f"{zero_rows} 行零特征")
    if not (1.0 - norm_tol <= norm_min <= norm_max <= 1.0 + norm_tol):
        report["errors"].append(f"范数越界 [{norm_min:.9f}, {norm_max:.9f}]")
    report["ok"] = not report["errors"]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="完整图库特征校验")
    ap.add_argument("--prefix", required=True, help="索引前缀，如 .../gallery")
    ap.add_argument("--json", default="", help="报告输出路径")
    ap.add_argument("--norm-tol", type=float, default=2e-4)
    a = ap.parse_args()
    rep = validate(a.prefix, a.norm_tol)
    txt = json.dumps(rep, ensure_ascii=False, indent=1)
    print(txt)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(txt)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
