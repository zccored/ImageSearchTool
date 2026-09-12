# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器（二值法粗筛 + ResNet 精排）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""命令行查验去重（只读，不删除/移动任何文件）。

用法:
  python devtools/dedup_scan_real.py F:\\视频            # 默认阈值 4%
  python devtools/dedup_scan_real.py F:\\视频 3          # 阈值 3%
  python devtools/dedup_scan_real.py F:\\视频 --top 20   # 只列前 20 组
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import psutil  # noqa: E402

from hybrid_search import dedup as DD  # noqa: E402
from hybrid_search.config import Config  # noqa: E402
from hybrid_search.io_utils import collect_images  # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else r"F:\视频"
    threshold = 0.04
    top = 15
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--top" and i + 1 < len(args):
            top = int(args[i + 1])
        elif not a.startswith("--") and a.replace(".", "", 1).isdigit():
            threshold = float(a) / 100.0
    cfg = Config()
    prefix = os.path.join(root, ".gallery_index", "gallery")
    t0 = time.time()
    paths = collect_images(root, cfg.extensions)
    print(f"扫描 {root}：{len(paths)} 张图片（{time.time() - t0:.1f}s）")
    last = [0.0]

    def prog(done, total, phase):
        now = time.time()
        if now - last[0] < 1.5 and done < total:
            return
        last[0] = now
        print(f"  [{phase}] {done}/{total}", flush=True)

    rep = DD.scan_duplicates(paths, prefix=prefix, threshold=threshold,
                             progress=prog)
    print(f"\n用时 {rep.elapsed:.1f}s | 复用索引 {rep.indexed_used} 张 | "
          f"解码 {rep.decoded} 张 | 读取失败 {len(rep.errors)} 张")
    print(f"重复组 {len(rep.groups)}（完全 {rep.n_exact} / 近似 {rep.n_near}）"
          f" | 涉及图片 {rep.n_images} 张 | 可释放 {human(rep.wasted_bytes)}")
    for g in rep.groups[:top]:
        kind = "完全" if g.all_exact else ("近似" if g.kind == "near" else "含完全")
        print(f"\n组{g.gid} [{kind}] {len(g.members)} 张 "
              f"可释放 {human(g.wasted_bytes)} "
              f"最大汉明={g.max_hamming * 100:.2f}% "
              + (f"最小余弦={g.min_cos:.4f}" if g.min_cos is not None else ""))
        for m in g.members:
            print(f"   {'保留' if m is g.keep else '重复'} {m.name[:44]:<44} "
                  f"{m.w}x{m.h} {human(m.size):>9} "
                  f"汉明={m.hamming * 100:5.2f}% "
                  f"{'字节相同' if m.exact_copy else '        '} "
                  f"{'已入库' if m.indexed else '未入库'}  {m.path}")
    if len(rep.groups) > top:
        print(f"\n…还有 {len(rep.groups) - top} 组未列出（--top N 调整）")
    print(f"\n常驻内存 {psutil.Process().memory_info().rss / 2**20:.0f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
