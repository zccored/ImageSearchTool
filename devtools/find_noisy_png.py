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

"""找出真正含 iCCP/cHRM 块的 PNG，并测量噪音输出的真实代价。

用法: python devtools/find_noisy_png.py [最多扫描张数]
"""
import os
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search.io_utils import (decode_rgb, silence_png_noise,  # noqa: E402
                                    stderr_noise_stats)

ROOT = r"F:\视频"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
NOISY_CHUNKS = (b"iCCP", b"cHRM", b"sRGB")


def png_chunks(path, max_bytes=8192):
    """读 PNG 头部的 chunk 名列表（iCCP/cHRM 规范上位于 IDAT 之前）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(max_bytes)
    except OSError:
        return set()
    if not head.startswith(b"\x89PNG\r\n\x1a\n"):
        return set()
    names, pos = set(), 8
    while pos + 8 <= len(head):
        (length,) = struct.unpack(">I", head[pos:pos + 4])
        name = head[pos + 4:pos + 8]
        names.add(name)
        if name == b"IDAT":
            break
        pos += 12 + length
        if length > len(head):
            break
    return names


def collect(limit):
    out = []
    for dp, _dn, fn in os.walk(ROOT):
        for f in fn:
            if f.lower().endswith(".png"):
                out.append(os.path.join(dp, f))
                if len(out) >= limit:
                    return out
    return out


files = collect(LIMIT)
print(f"扫描 {len(files)} 个 PNG 的 chunk…")
noisy = []
for p in files:
    names = png_chunks(p)
    if names & set(NOISY_CHUNKS):
        noisy.append((p, sorted(n.decode() for n in names & set(NOISY_CHUNKS))))
print(f"含 iCCP/cHRM/sRGB 的 PNG：{len(noisy)} 个")
for p, ch in noisy[:8]:
    print(f"   {ch}  {os.path.basename(p)[:50]}  {os.path.getsize(p) / 1024:.0f}KB")

sample = [p for p, _c in noisy[:24]]
if not sample:
    print("\n本图库未发现 iCCP/cHRM PNG（噪音来自其它来源或已修复的旧文件）")
    sys.exit(0)


def decode_all(paths):
    ok, t0 = 0, time.time()
    for p in paths:
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        if decode_rgb(data) is not None:
            ok += 1
    return ok, time.time() - t0


def fd2_to(path):
    saved = os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 2)
    os.close(fd)
    return lambda: (os.dup2(saved, 2), os.close(saved))


print(f"\n对 {len(sample)} 个含 iCCP 的 PNG 解码两轮：")
tmp = os.path.join(tempfile.gettempdir(), "noisy.log")
sys.stderr.flush()
restore = fd2_to(tmp)
try:
    ok1, dt1 = decode_all(sample)
finally:
    sys.stderr.flush()
    restore()
lines = [ln for ln in open(tmp, "rb").read().decode("utf-8", "replace").splitlines()
         if "libpng" in ln or "iCCP" in ln]
print(f"  未过滤：{len(lines)} 条噪音，用时 {dt1 * 1000:.0f} ms")

before = stderr_noise_stats()
silence_png_noise(True)
sys.stderr.flush()
tmp2 = os.path.join(tempfile.gettempdir(), "quiet.log")
restore2 = fd2_to(tmp2)
try:
    ok2, dt2 = decode_all(sample)
finally:
    sys.stderr.flush()
    restore2()
after = stderr_noise_stats()
left = [ln for ln in open(tmp2, "rb").read().decode("utf-8", "replace").splitlines()
        if "libpng" in ln or "iCCP" in ln]
print(f"  已过滤：残留 {len(left)} 条，suppressed={after['suppressed'] - before['suppressed']}，"
      f"用时 {dt2 * 1000:.0f} ms")
print(f"\n结论：噪音 {len(lines)} 条 → {len(left)} 条；"
      f"吞吐 {len(sample) / dt1:.1f} → {len(sample) / dt2:.1f} 张/秒"
      f"（{(dt1 / dt2 - 1) * 100:+.1f}%）；解码一致={ok1 == ok2}")
