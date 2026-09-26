# -*- coding: utf-8 -*-
"""GDeflate 探针：搞清 nvCOMP 5.3 的 GDeflate 能否吃"外来 DEFLATE 流"、输出格式是否兼容、
硬件解码后端是否可用。只读图库样本，不写任何工程数据。"""
import os
import struct
import sys
import time
import zlib

import numpy as np
import torch
# CUDA 守卫：本脚本基准依赖 NVIDIA CUDA；AMD / 无卡环境下友好退出而不是抛栈。
if not torch.cuda.is_available():
    print("需要 NVIDIA CUDA 设备（AMD 显卡或无卡环境无法运行本基准）；已跳过。")
    raise SystemExit(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from nvidia import nvcomp                                            # noqa: E402

print("nvcomp", nvcomp.__version__, "cuda", nvcomp.__cuda_version__)
print("gpu", torch.cuda.get_device_name(0))


def show(tag, fn):
    try:
        v = fn()
        print("[OK ] %-34s -> %s" % (tag, v))
        return v
    except Exception as e:                                           # noqa: BLE001
        msg = str(e).replace("\n", " ")[:220]
        print("[ERR] %-34s -> %s: %s" % (tag, type(e).__name__, msg))
        return None


def idat_of(data: bytes) -> bytes:
    i, out = 8, []
    while i + 8 <= len(data):
        (ln,) = struct.unpack(">I", data[i:i + 4])
        typ = data[i + 4:i + 8]
        if typ == b"IDAT":
            out.append(data[i + 8:i + 8 + ln])
        elif typ == b"IEND":
            break
        i += 12 + ln
    return b"".join(out)


# ---------------------------------------------------------------- 1) 构造 codec
codec = show("Codec(algorithm=GDeflate)", lambda: nvcomp.Codec(algorithm="GDeflate", device_id=0))
if codec is None:
    codec = show("Codec(algorithm=gdeflate)", lambda: nvcomp.Codec(algorithm="gdeflate", device_id=0))
print("codec:", codec)

# ---------------------------------------------------------------- 2) 往返正确性
payload = (np.random.RandomState(0).randint(0, 256, size=4 << 20).astype(np.uint8))
p_t = torch.from_numpy(payload).cuda()
enc = show("encode(4MB uint8)", lambda: codec.encode(nvcomp.as_array(p_t)))
if enc is not None:
    cb = show("encoded -> bytes", lambda: bytes(np.from_dlpack(enc).cpu().numpy().tobytes()) if False else enc)
    try:
        enc_t = torch.from_dlpack(enc) if hasattr(enc, "__dlpack__") else None
        print("       enc type:", type(enc), "size?", getattr(enc, "size", None),
              "buffer_size?", getattr(enc, "buffer_size", None))
        raw = enc_t.cpu().numpy().tobytes() if enc_t is not None else None
    except Exception as e:                                           # noqa: BLE001
        print("       dlpack fail", e)
        raw = None
    if raw:
        print("       enc head:", raw[:8].hex(), "len", len(raw))
        print("[?] ", "zlib(wbits=15) 解 GDeflate 输出:",
              show("zlib.decompress(enc)", lambda: len(zlib.decompress(raw))))
        print("[?] ", "raw deflate(wbits=-15):",
              show("zlib raw inflate(enc)", lambda: len(zlib.decompressobj(-15).decompress(raw))))
        # 反向：GDeflate 解 zlib 流 / raw deflate 流
        out = torch.empty_like(p_t)
        z = zlib.compress(payload)
        z_t = torch.frombuffer(bytearray(z), dtype=torch.uint8).cuda()
        show("decode(zlib 流 by GDeflate)",
             lambda: codec.decode(nvcomp.as_array(z_t), out=nvcomp.as_array(out)))
        rawd = z[2:-4]
        r_t = torch.frombuffer(bytearray(rawd), dtype=torch.uint8).cuda()
        show("decode(raw deflate by GDeflate)",
             lambda: codec.decode(nvcomp.as_array(r_t), out=nvcomp.as_array(out)))
        # 自产数据用 GDeflate 解
        r = show("decode(GDeflate 流)", lambda: codec.decode(enc, out=nvcomp.as_array(out)))
        if r is not None:
            got = out.cpu().numpy().tobytes()
            print("       roundtrip identical:", got == payload.tobytes())

# ---------------------------------------------------------------- 3) 真实 PNG IDAT
ROOT = r"F:\视频"
png = None
for dp, _dn, fn in os.walk(ROOT):
    for f in fn:
        if f.lower().endswith(".png") and os.path.getsize(os.path.join(dp, f)) > 3 << 20:
            png = os.path.join(dp, f)
            break
    if png:
        break
if png:
    data = open(png, "rb").read()
    w, h = struct.unpack(">II", data[16:24])
    bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(data[25], 4)
    rawsz = (w * bpp * data[24] // 8 + 1) * h
    idat = idat_of(data)
    print("real png:", os.path.basename(png), "%dx%d raw=%.1fMB idat=%.1fMB"
          % (w, h, rawsz / 2 ** 20, len(idat) / 2 ** 20))
    out = torch.empty(rawsz, dtype=torch.uint8, device="cuda")
    zl = torch.frombuffer(bytearray(idat), dtype=torch.uint8).cuda()
    show("real PNG IDAT -> GDeflate decode",
         lambda: codec.decode(nvcomp.as_array(zl), out=nvcomp.as_array(out)))
    rd = torch.frombuffer(bytearray(idat[2:-4]), dtype=torch.uint8).cuda()
    show("real PNG raw deflate -> GDeflate decode",
         lambda: codec.decode(nvcomp.as_array(rd), out=nvcomp.as_array(out)))
    # GDeflate 压缩"解出后的像素缓冲"再解回：自建格式场景
    try:
        px = torch.from_numpy(np.frombuffer(zlib.decompress(idat), dtype=np.uint8).copy()).cuda()
        t0 = time.perf_counter()
        e2 = codec.encode(nvcomp.as_array(px))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        o2 = torch.empty_like(px)
        codec.decode(e2, out=nvcomp.as_array(o2))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ok = o2.cpu().numpy().tobytes() == px.cpu().numpy().tobytes()
        print("[OK ] self-format: compress %.1f ms, decompress %.1f ms, %.0f MB/s, identical=%s"
              % ((t1 - t0) * 1e3, (t2 - t1) * 1e3, rawsz / (t2 - t1) / 2 ** 20, ok))
    except Exception as e:                                           # noqa: BLE001
        print("[ERR] self-format:", type(e).__name__, str(e)[:200])

print("bitstream kinds:", [m for m in dir(nvcomp.BitstreamKind) if not m.startswith("_")])
