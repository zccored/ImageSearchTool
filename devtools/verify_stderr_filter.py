# -*- coding: utf-8 -*-
"""用合成写入验证 libpng 噪音过滤器机制（不依赖具体文件是否告警）。

关键顺序：先把 fd 2 指向日志文件（模拟“终端”），再安装过滤器（此时过滤器
的 saved_fd 就是该日志文件）——噪音被吞、真实错误行应落到日志文件里。
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search.io_utils import silence_png_noise, stderr_noise_stats  # noqa: E402

NOISE = b"libpng warning: iCCP: known incorrect sRGB profile\n"
NOISE2 = b"libpng error: PNG input buffer is incomplete\n"
REAL = "CUDA ERROR: 这行必须原样透出\n".encode("utf-8")

orig_fd2 = os.dup(2)          # 先备份真实 stderr（供最后还原）
log_path = os.path.join(tempfile.gettempdir(), "filter_check.log")
log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
sys.stderr.flush()
os.dup2(log_fd, 2)            # fd 2 -> 日志文件（模拟终端）
os.close(log_fd)

silence_png_noise(True)       # 安装过滤器：内部 saved_fd = 日志文件
before = stderr_noise_stats()
for _ in range(50):
    os.write(2, NOISE)
os.write(2, REAL)
os.write(2, NOISE2)

time.sleep(0.4)               # 等过滤线程把转发内容写完
os.fsync
os.dup2(orig_fd2, 2)          # 还原真实 stderr
os.close(orig_fd2)

after = stderr_noise_stats()
txt = open(log_path, "rb").read().decode("utf-8", "replace")
sup = after["suppressed"] - before["suppressed"]
fwd = after["forwarded"] - before["forwarded"]
print("=== 过滤器机制 ===")
print(f"  suppressed 增量 = {sup}（期望 51 = 50×iCCP + 1×incomplete）")
print(f"  forwarded  增量 = {fwd}（期望 1 = 真实错误行）")
print(f"  日志文件内容   = {txt.strip()[:80]!r}")
ok = (sup == 51 and fwd == 1 and "CUDA ERROR" in txt and "libpng" not in txt)
print("  判定:", "✓ 噪音被吞、真实错误透出" if ok else "✗ 异常")

print("\n=== 参考：每行 stderr 写入成本（本环境为管道，非真实控制台）===")
devnull = os.open(os.devnull, os.O_WRONLY)
saved2 = os.dup(2)
os.dup2(devnull, 2)
os.close(devnull)
try:
    t0 = time.time()
    for _ in range(20000):
        os.write(2, NOISE)
    dt = time.time() - t0
finally:
    os.dup2(saved2, 2)
    os.close(saved2)
print(f"  20000 行 -> {dt * 1000:.0f} ms，{dt / 20000 * 1e6:.2f} µs/行")
print("  真实交互式控制台（Windows Terminal/cmd）每行成本远高于此，"
      "成千上万行会明显拖慢并刷屏。")
