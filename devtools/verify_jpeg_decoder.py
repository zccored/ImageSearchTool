# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — JPEG 解码器一致性验证（cv2 vs TurboJPEG）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------
"""逐位比对 cv2（现状）与 TurboJPEG 的 JPEG 解码结果，并压测线程安全与回退。

**为什么必须逐位一致**：粗筛指纹/Hu/ResNet 输入都由解码像素直接决定，
一旦有位差就要重建索引（37,683 张整图 + 444,252 块瓦片）并重跑召回评估。
所以"换 JPEG 解码器"的准入线是 **0 位差**，不是"看起来一样"。

四段验证：
  (a) 真实图库分层抽样（按 基线/渐进 × 分量数 × DCT 缩放档）
  (b) 合成格式矩阵（PIL 编码：基线/渐进 × 灰度/彩色 × 4:4:4/4:2:2/4:2:0 × 四个缩放档）
  (c) EXIF 方向 1..8 —— 现状 cv2 会自动转正；候选实现对 orient!=1 **一律回退 cv2**
      （构造上保证 0 位差；本图库实测 100% orient=1，故无覆盖率损失）
  (d) 18 路线程池压测 + 坏文件/非 JPEG/极小图回退

只读图片，不写索引、不删文件。

用法: python -E devtools/verify_jpeg_decoder.py [真实样本扫描数=3000] [每档上限=40]
"""
import io
import os
import random
import sys
import threading
import time
import warnings
from collections import Counter, defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from paths import GALLERY_ROOT  # noqa: E402

GALLERY_INDEX = os.path.join(GALLERY_ROOT, ".gallery_index")
DLL = os.environ.get("TURBOJPEG_DLL") or os.path.join(
    os.environ.get("TEMP", ""), "c2_jpeg", "turbojpeg.dll")
_TARGET = 2048


def scale_of(w, h):
    ms = max(w, h)
    if ms <= _TARGET * 1.25:
        return (1, 1)
    if ms <= _TARGET * 2.5:
        return (1, 2)
    if ms <= _TARGET * 5:
        return (1, 4)
    return (1, 8)


_TL = threading.local()


def tj():
    """每线程一个 TurboJPEG 句柄（生产实现同款设计）。"""
    if getattr(_TL, "obj", None) is None:
        from turbojpeg import TJPF_RGB, TurboJPEG
        _TL.obj = TurboJPEG(DLL)
        _TL.rgb = TJPF_RGB
    return _TL.obj, _TL.rgb


def cand_rgb(iu, data: bytes, probe):
    """候选实现（= 计划写进 io_utils 的逻辑，务必保持同步）：

    只接管 **JPEG + EXIF 方向为 1 + 尾部带 EOI(FFD9)**，按与现状完全相同的 DCT 缩放档
    解码直出 RGB；其余一律返回 None 交回 cv2，构造上保证 0 位差。

    为什么必须查尾部 EOI：实测**截断的 JPEG** 两者行为不同 —— cv2 imdecode 返回 None
    （文件被跳过），TurboJPEG 却会补边解出半张图。若不拦，这类文件会**从"不入索引"
    变成"入索引"**，索引内容就变了。缺 EOI 一律回退即可完全避开这一档
    （实测本图库 23,300 张里只有 69 张缺 EOI，且它们在 cv2 下都能正常解出，
     即这条守卫在当前语料上不改变任何一张的结果）。
    注：TurboJPEG 的 TJFLAG_STOPONWARNING 对 libjpeg 的 "Premature end of JPEG file"
    这类警告无效（实测截断文件照样解出），所以不能用它替代这条守卫。
    """
    if probe is None or probe[0] != "JPEG" or probe[2] != 1:
        return None
    if data[-2:] != b"\xff\xd9":                          # 缺 EOI：疑似截断，交回 cv2
        return None
    obj, TJPF_RGB = tj()
    try:
        return obj.decode(data, pixel_format=TJPF_RGB,
                          scaling_factor=scale_of(*probe[1]))
    except Exception:                                     # noqa: BLE001
        return None


# ------------------------------------------------------------------ 头部扫描
def jpeg_head_info(head: bytes):
    """(w, h, 类别, 分量数, 采样比) —— 只看标记段。"""
    if head[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(head)
    w = h = comp = 0
    kind, subs = "?", ""
    while i + 4 <= n:
        if head[i] != 0xFF:
            i += 1
            continue
        m = head[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0x01 or 0xD0 <= m <= 0xD8:
            i += 2
            continue
        if m == 0xDA:
            break
        ln = int.from_bytes(head[i + 2:i + 4], "big")
        seg = head[i + 4:i + 2 + ln]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            kind = {0xC0: "基线", 0xC1: "扩展序列", 0xC2: "渐进", 0xC3: "无损"}.get(m, "SOF%02x" % m)
            if len(seg) >= 6:
                h, w = int.from_bytes(seg[1:3], "big"), int.from_bytes(seg[3:5], "big")
                comp = seg[5]
                if len(seg) >= 6 + 3 * comp:
                    hv = seg[7]
                    subs = {0x11: "4:4:4", 0x22: "4:2:0", 0x21: "4:2:2",
                            0x12: "4:4:0"}.get(hv, "%02x" % hv)
                    if comp == 1:
                        subs = "灰度"
        i += 2 + ln
    return (w, h, kind, comp, subs) if w and h else None


def main() -> int:
    scan = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    per_fmt = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    from hybrid_search import io_utils as iu

    if not os.path.exists(DLL):
        print("**缺 turbojpeg.dll**：%s\n请先跑 python -E devtools/fetch_turbojpeg.py" % DLL)
        return 2
    import turbojpeg
    print("PyTurboJPEG %s · DLL %s（%d B）" % (turbojpeg.__version__, DLL, os.path.getsize(DLL)))

    fails = []

    # ---------------------------------------------------------- (a) 真实图库
    paths = [str(x) for x in np.load(os.path.join(GALLERY_INDEX, "gallery.paths.npy"),
                                     allow_pickle=True)]
    jpgs = [p for p in paths if p.lower().endswith((".jpg", ".jpeg"))]
    random.seed(7)
    random.shuffle(jpgs)
    buckets = defaultdict(list)
    for p in jpgs[:scan]:
        try:
            with open(p, "rb") as f:
                info = jpeg_head_info(f.read(32 << 10))
        except OSError:
            continue
        if not info:
            continue
        key = "%s %d分量 %s %s" % (info[2], info[3], info[4],
                                   "1/%d" % scale_of(info[0], info[1])[1])
        if len(buckets[key]) < per_fmt:
            buckets[key].append(p)
    print("\n(a) 真实图库：扫描 %d 张，得到 %d 个分层" % (min(scan, len(jpgs)), len(buckets)))
    a_ok = a_bad = a_err = 0
    t_cv = t_tj = 0.0
    for key in sorted(buckets):
        files = buckets[key]
        ok = bad = err = 0
        for p in files:
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            probe = iu._probe(data)
            t0 = time.perf_counter()
            ref = iu.decode_rgb(data)
            t_cv += time.perf_counter() - t0
            t0 = time.perf_counter()
            c = cand_rgb(iu, data, probe)
            t_tj += time.perf_counter() - t0
            if c is None:                                 # 回退路径：等价于现状
                if ref is None:
                    err += 1
                else:
                    ok += 1
                continue
            if ref is None:
                err += 1
            elif c.shape == ref.shape and np.array_equal(c, ref):
                ok += 1
            else:
                bad += 1
                fails.append((key, p, "位差",
                              0 if ref is None or c.shape != ref.shape
                              else int(np.abs(ref.astype(int) - c.astype(int)).max())))
        a_ok += ok
        a_bad += bad
        a_err += err
        print("   %-34s 一致 %3d/%-3d 不一致 %d 失败 %d" % (key, ok, len(files), bad, err))
    print("   —— 真实样本合计：一致 %d，不一致 %d，失败 %d" % (a_ok, a_bad, a_err))

    # ---------------------------------------------------------- (b) 合成矩阵
    print("\n(b) 合成格式矩阵（PIL 编码，覆盖真实图库稀缺的组合）")
    from PIL import Image
    rng = np.random.default_rng(5)
    cases = []
    skipped = []
    for side in (300, 512, 2600, 3000, 5400, 6000, 11000):
        base = (rng.random((side, side, 3)) * 255).astype(np.uint8)
        # 注：Pillow 10.1.0 **不能编码"渐进 + subsampling≠2"**的 JPEG
        # （libjpeg 报 Suspension not allowed here / encoder error -2），
        # 与本轮换库无关，是编码侧的 Pillow 限制；渐进 4:4:4 的覆盖由真实图库样本提供。
        for kw, tag in (({"quality": 90, "subsampling": 0}, "基线 4:4:4"),
                        ({"quality": 90, "subsampling": 1}, "基线 4:2:2"),
                        ({"quality": 90, "subsampling": 2}, "基线 4:2:0"),
                        ({"quality": 88, "progressive": True, "subsampling": 2}, "渐进 4:2:0"),
                        ({"quality": 88, "progressive": True}, "渐进 默认采样"),
                        ({"quality": 75}, "低质量 默认采样"),
                        ({"quality": 100, "subsampling": 0}, "质量100 4:4:4")):
            buf = io.BytesIO()
            try:
                Image.fromarray(base).save(buf, "JPEG", **kw)
            except Exception as e:                        # noqa: BLE001
                skipped.append(("%dpx %s" % (side, tag), repr(e)[:60]))
                continue
            cases.append(("%dpx %s" % (side, tag), buf.getvalue()))
        g = (rng.random((side, side)) * 255).astype(np.uint8)
        for kw, tag in (({"quality": 90}, "灰度"), ({"quality": 88, "progressive": True}, "灰度渐进")):
            buf = io.BytesIO()
            try:
                Image.fromarray(g).save(buf, "JPEG", **kw)
            except Exception as e:                        # noqa: BLE001
                skipped.append(("%dpx %s" % (side, tag), repr(e)[:60]))
                continue
            cases.append(("%dpx %s" % (side, tag), buf.getvalue()))
    if skipped:
        print("   跳过 %d 个 Pillow 编不出来的组合，例：%s" % (len(skipped), skipped[:2]))
    # 极小图
    for side in (1, 2, 8, 17):
        buf = io.BytesIO()
        Image.fromarray((rng.random((side, side, 3)) * 255).astype(np.uint8)).save(buf, "JPEG")
        cases.append(("%dpx 极小" % side, buf.getvalue()))
    # CMYK（Adobe APP14）
    try:
        buf = io.BytesIO()
        Image.fromarray((rng.random((600, 600, 4)) * 255).astype(np.uint8), "CMYK").save(
            buf, "JPEG", quality=90)
        cases.append(("600px CMYK", buf.getvalue()))
    except Exception as e:                                # noqa: BLE001
        print("   （CMYK 样例生成失败：%r）" % e)

    b_ok = b_bad = b_fb = 0
    for tag, data in cases:
        probe = iu._probe(data)
        ref = iu.decode_rgb(data)
        c = cand_rgb(iu, data, probe)
        if c is None:
            b_fb += 1
            print("   %-24s 回退 cv2（probe=%s）" % (tag, probe and (probe[0], probe[2])))
            continue
        if ref is None:
            b_bad += 1
            fails.append(("合成", tag, "cv2 失败而候选成功", 0))
            print("   %-24s **cv2 解不了、候选能解** —— 需人工确认" % tag)
        elif c.shape == ref.shape and np.array_equal(c, ref):
            b_ok += 1
            print("   %-24s 逐位一致  %s  缩放 1/1" % (tag, ref.shape))
        else:
            b_bad += 1
            d = 0 if c.shape != ref.shape else int(np.abs(c.astype(int) - ref.astype(int)).max())
            fails.append(("合成", tag, "位差", d))
            print("   %-24s **不一致**  cv2 %s / 候选 %s  最大差 %d" % (tag, ref.shape, c.shape, d))
    print("   —— 合成合计：一致 %d，不一致 %d，回退 %d" % (b_ok, b_bad, b_fb))

    # ---------------------------------------------------------- (c) EXIF 方向
    print("\n(c) EXIF 方向 1..8：候选必须对 orient!=1 回退 cv2（构造上 0 位差）")
    c_bad = 0
    for orient in range(1, 9):
        base = (rng.random((900, 1400, 3)) * 255).astype(np.uint8)
        ex = Image.Exif()
        ex[0x0112] = orient
        buf = io.BytesIO()
        Image.fromarray(base).save(buf, "JPEG", quality=90, exif=ex.tobytes())
        data = buf.getvalue()
        probe = iu._probe(data)
        ref = iu.decode_rgb(data)
        c = cand_rgb(iu, data, probe)
        got = iu._probe(data)[2]
        if orient != 1:
            expect_fb = c is None
            c_bad += (not expect_fb)
            print("   orient=%d  probe 读到 %s  候选%s（期望回退）" % (orient, got, "回退" if c is None else "**未回退**"))
        else:
            same = (c is not None and ref is not None and c.shape == ref.shape
                    and np.array_equal(c, ref))
            c_bad += (not same)
            print("   orient=1  probe 读到 %s  候选%s" % (got, "逐位一致" if same else "**不一致**"))

    # ---------------------------------------------------------- (d) 线程压测
    print("\n(d) 18 路线程池压测（每线程一个 TurboJPEG 句柄）")
    pool = []
    for key in sorted(buckets):
        for p in buckets[key][:6]:
            try:
                pool.append(open(p, "rb").read())
            except OSError:
                pass
    pool = pool * 4
    refs = []
    for data in pool:
        probe = iu._probe(data)
        refs.append((probe, iu.decode_rgb(data)))
    idx = [0]
    lock = threading.Lock()
    errs = []

    def work():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(pool):
                return
            probe, ref = refs[i]
            c = cand_rgb(None, pool[i], probe)
            if ref is None:
                continue
            if c is None:
                continue                                  # 回退，属正常
            if c.shape != ref.shape or not np.array_equal(c, ref):
                with lock:
                    errs.append(i)

    t0 = time.perf_counter()
    ths = [threading.Thread(target=work, daemon=True) for _ in range(18)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print("   %d 张 × 18 路，耗时 %.2f s，位差/异常 %d 张" % (len(pool), time.perf_counter() - t0, len(errs)))
    fails += [("压测", "idx=%d" % i, "位差", 0) for i in errs[:10]]

    # ---------------------------------------------------------- (e) 损坏/截断
    # 关键：截断的 JPEG 两个库都"能解出东西"（libjpeg 补灰/补边），但**补法可能不同**。
    # 只要有一张真实文件落在这种状态，换库就会改索引 → 必须逐位比对，不能只看"没抛异常"。
    print("\n(e) 损坏/截断 JPEG：候选与 cv2 必须逐位一致，否则必须回退")
    good = None
    for tag, data in cases:
        if tag.startswith("300px 基线 4:2:0"):
            good = data
            break
    good = good or cases[0][1]
    weird = [("空字节", b""), ("非图像", b"this is not an image at all"),
             ("截断(前 200B)", good[:200]), ("只有 SOI", b"\xff\xd8"),
             ("JPEG 头 + PNG 体", good[:2] + b"\x89PNG\r\n\x1a\n" + good[10:]),
             ("缺尾部 1B", good[:-1]), ("缺尾部 2B", good[:-2]),
             ("截断(95%)", good[:int(len(good) * .95)]),
             ("截断(80%)", good[:int(len(good) * .80)]),
             ("截断(50%)", good[:int(len(good) * .50)]),
             ("中间翻转字节", good[:len(good) // 2] + bytes([good[len(good) // 2] ^ 0xFF])
              + good[len(good) // 2 + 1:]),
             ("熵段乱码", good[:len(good) // 3] + bytes(len(good) // 3))]
    e_bad = e_fb = e_mismatch = 0
    for tag, data in weird:
        try:
            probe = iu._probe(data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")            # PyTurboJPEG 截断时会 warnings.warn
                ref = iu.decode_rgb(data)
                c = cand_rgb(iu, data, probe)
            if c is None:
                e_fb += 1
                print("   %-16s 回退 cv2（与现状等价）" % tag)
            elif ref is None:
                e_bad += 1
                fails.append(("回退", tag, "cv2 解不了而候选能解", 0))
                print("   %-16s **cv2 失败 / 候选成功** —— 会改变索引" % tag)
            elif c.shape == ref.shape and np.array_equal(c, ref):
                print("   %-16s 直出 %s 且逐位一致" % (tag, c.shape))
            else:
                e_mismatch += 1
                d = 0 if c.shape != ref.shape else int(np.abs(c.astype(int) - ref.astype(int)).max())
                fails.append(("回退", tag, "与 cv2 位差", d))
                print("   %-16s **与 cv2 不一致** 形状 %s/%s 最大差 %d"
                      % (tag, c.shape, ref.shape, d))
        except Exception as ex:                           # noqa: BLE001
            e_bad += 1
            fails.append(("回退", tag, "抛异常", 0))
            print("   %-16s **抛异常** %r" % (tag, ex))
    print("   —— 回退 %d 个 / 位差 %d 个 / 异常或单向成功 %d 个" % (e_fb, e_mismatch, e_bad))

    # ---------------------------------------------------------- (f) 全库截断普查
    # 真实图库若存在"缺 EOI"的 JPEG，就落进 (e) 的高风险区 → 先数清楚有多少张。
    print("\n(f) 全库 JPEG 尾部普查（缺 EOI 标记 = 截断风险文件）")
    n_all = n_bad_tail = 0
    bad_tail = []
    for p in jpgs:
        try:
            with open(p, "rb") as f:
                f.seek(-2, os.SEEK_END)
                tail = f.read(2)
                sz = f.tell() + 2
        except OSError:
            continue
        n_all += 1
        if len(tail) < 2 or tail != b"\xff\xd9":
            n_bad_tail += 1
            if len(bad_tail) < 10:
                bad_tail.append((p, sz, tail.hex()))
    print("   检查 %d 张：缺 EOI 的 %d 张（%.3f%%）%s"
          % (n_all, n_bad_tail, 100.0 * n_bad_tail / max(n_all, 1),
             ("例：" + "; ".join("%s(%s)" % (os.path.basename(x[0]), x[2]) for x in bad_tail[:3]))
             if bad_tail else ""))
    if n_bad_tail:
        # 对缺 EOI 的文件逐一做候选 vs cv2 逐位比对
        mism = 0
        for p, _sz, _h in bad_tail:
            try:
                data = open(p, "rb").read()
                probe = iu._probe(data)
                ref = iu.decode_rgb(data)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c = cand_rgb(iu, data, probe)
                if c is not None and (ref is None or c.shape != ref.shape
                                      or not np.array_equal(c, ref)):
                    mism += 1
                    fails.append(("缺EOI", p, "位差", 0))
            except Exception:                             # noqa: BLE001
                mism += 1
        print("   其中候选与 cv2 不一致 %d 张 → %s"
              % (mism, "需在实现里加截断检测并回退" if mism else "无需特殊处理"))
        e_bad += mism

    # ---------------------------------------------------------- 结论
    tot_bad = a_bad + b_bad + c_bad + len(errs) + e_bad
    print("\n===== 结论 =====")
    print("真实 %d/%d 一致，合成 %d/%d 一致，EXIF/线程/回退问题 %d"
          % (a_ok, a_ok + a_bad, b_ok, b_ok + b_bad, c_bad + len(errs) + e_bad))
    print("候选相对现状的解码耗时比（同档，仅真实样本段）：%.3fx"
          % ((t_cv / t_tj) if t_tj else 0))
    if fails:
        print("问题清单（前 10）：")
        for f in fails[:10]:
            print("   %s" % (f,))
    print("判定：%s" % ("**可安全切换**（0 位差，回退路径与现状等价）" if tot_bad == 0
                      else "**存在 %d 处问题 —— 不可切换**" % tot_bad))
    return 0 if tot_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
