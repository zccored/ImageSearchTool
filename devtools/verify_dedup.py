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

"""重复图查验回归：完全重复 / 近似重复 / 无关图不误报 / 回收站 / 移动。

用法: python devtools/verify_dedup.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from hybrid_search import dedup as D  # noqa: E402

work = tempfile.mkdtemp(prefix="deduptest_")
ok = True


def base_image(seed: int, w=900, h=700):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (xx / w * 200 + seed * 17) % 256
    g = (yy / h * 180 + 60) % 256
    b = ((xx + yy) / (w + h) * 220 + seed * 40) % 256
    arr = np.stack([r, g, b], axis=2).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for k in range(5):
        cx = rng.randint(80, w - 80)
        cy = rng.randint(80, h - 80)
        rad = rng.randint(40, 140)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                  fill=(rng.randint(0, 255), rng.randint(0, 255),
                        rng.randint(0, 255)))
    d.text((40, 40), f"IMG-{seed}", fill=(255, 255, 255))
    return im


try:
    # ---- 构造：A 系列(4 份近似/完全重复) + 3 张无关图 -----------------
    a = base_image(1)
    p_a = os.path.join(work, "a.jpg")
    a.save(p_a, quality=92)
    p_a_copy = os.path.join(work, "a_复制.jpg")
    shutil.copy2(p_a, p_a_copy)                          # 完全相同字节
    p_a_re = os.path.join(work, "a_重编码.jpg")
    a.save(p_a_re, quality=60)                           # 重新编码
    p_a_small = os.path.join(work, "a_缩放.jpg")
    a.resize((int(a.width * 0.9), int(a.height * 0.9))).save(p_a_small,
                                                            quality=92)
    p_a_png = os.path.join(work, "a.png")
    a.save(p_a_png)                                      # 同图不同格式

    others = []
    for s in (2, 3, 4):
        p = os.path.join(work, f"other{s}.jpg")
        base_image(s).save(p, quality=92)
        others.append(p)

    paths = [p_a, p_a_copy, p_a_re, p_a_small, p_a_png] + others
    print(f"构造 {len(paths)} 张：A 系列 5 张(完全1 + 近似3 + 同图异格式1) + 无关 3 张")

    rep = D.scan_duplicates(paths, threshold=0.04, workers=4)
    print(f"扫描 {rep.scanned} 张，用时 {rep.elapsed:.2f}s，"
          f"解码 {rep.decoded} 张，复用索引 {rep.indexed_used} 张")
    print(f"组数 {len(rep.groups)}（完全 {rep.n_exact} / 近似 {rep.n_near}）"
          f"，涉及图片 {rep.n_images} 张，可释放 {rep.wasted_bytes / 2**20:.2f} MB")
    for g in rep.groups:
        print(f"  组{g.gid} [{g.kind}] {len(g.members)} 张 "
              f"max汉明={g.max_hamming * 100:.2f}% "
              f"min余弦={'n/a' if g.min_cos is None else f'{g.min_cos:.4f}'}")
        for m in g.members:
            print(f"     {'保留' if m is g.keep else '重复'} "
                  f"{m.name:<16} {m.w}x{m.h} {m.size / 1024:8.1f}KB "
                  f"汉明={m.hamming * 100:5.2f}% "
                  f"{'字节相同' if m.exact_copy else '       '} "
                  f"md5={m.md5[:8]}")

    # ---- 断言 --------------------------------------------------------
    names = [{m.name for m in g.members} for g in rep.groups]
    a_group = next((g for g in rep.groups if p_a in [m.path for m in g.members]),
                   None)
    if a_group is None:
        print("✗ 未找到 A 系列重复组")
        ok = False
    else:
        got = {m.name for m in a_group.members}
        want = {"a.jpg", "a_复制.jpg", "a_重编码.jpg", "a_缩放.jpg", "a.png"}
        if got == want:
            print(f"✓ A 系列 5 张全部归入一组（kind={a_group.kind}）")
        else:
            print(f"✗ A 系列分组不全：缺 {want - got} 多 {got - want}")
            ok = False
        if a_group.keep.name == "a.jpg" or a_group.keep.name == "a.png":
            print(f"✓ 默认保留 {a_group.keep.name}（像素/体积最大优先）")
        else:
            print(f"? 默认保留 {a_group.keep.name}")
    # 无关图不应被分组
    for g in rep.groups:
        bad = [m.name for m in g.members if m.name.startswith("other")]
        if bad:
            print(f"✗ 无关图被误分组：{bad}")
            ok = False
    if not any(m.name.startswith("other") for g in rep.groups for m in g.members):
        print("✓ 3 张无关图未误报")

    # ---- 文件操作：回收站 + 移动 --------------------------------------
    junk = os.path.join(work, "to_recycle.txt")
    open(junk, "w").write("x")
    done, failed = D.recycle_paths([junk])
    if done and not os.path.exists(junk):
        print("✓ 回收站删除可用（文件已移出目录）")
    else:
        print(f"✗ 回收站删除失败：{failed}")
        ok = False

    dest = os.path.join(work, "moved")
    sub = os.path.join(work, "sub", "deep.jpg")
    os.makedirs(os.path.dirname(sub), exist_ok=True)
    base_image(9).save(sub, quality=90)
    moved, failed = D.move_paths([sub], dest, base_root=work)
    if moved and os.path.exists(os.path.join(dest, "sub", "deep.jpg")):
        print("✓ 移动保留目录结构")
    else:
        print(f"✗ 移动失败：{failed}")
        ok = False
    print("结果:", "全部通过" if ok else "存在失败项")
    sys.exit(0 if ok else 1)
finally:
    shutil.rmtree(work, ignore_errors=True)
