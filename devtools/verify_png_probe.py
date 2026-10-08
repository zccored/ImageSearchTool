# -*- coding: utf-8 -*-
# ImageSearchTool · PNG 元数据探测回归（不应为普通 PNG 解码像素）
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见仓库根 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""合成 PNG 验证文件头探测、前后置 EXIF/XMP、原始 EXIF profile 与回退。

python -E -B devtools/verify_png_probe.py
"""
import io
import os
import struct
import sys
import unittest
import zlib
from unittest.mock import patch

from PIL import Image, PngImagePlugin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_search import io_utils as iu  # noqa: E402


def png(mode="RGBA", **kwargs):
    buf = io.BytesIO()
    Image.new(mode, (33, 17)).save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def append_chunk(data, kind, payload):
    return data[:-12] + chunk(kind, payload) + data[-12:]


def old_probe(data):
    try:
        with Image.open(io.BytesIO(data)) as im:
            try:
                orient = int(im.getexif().get(0x0112, 1))
            except Exception:
                orient = 1
            return im.format, im.size, orient
    except Exception:
        return None


class ProbeTest(unittest.TestCase):
    def test_plain_formats_do_not_decode(self):
        for mode in ("L", "LA", "RGB", "RGBA", "P", "I;16"):
            with self.subTest(mode=mode):
                data = png(mode)
                with patch.object(PngImagePlugin.PngImageFile, "load") as load:
                    self.assertEqual(iu._probe(data), ("PNG", (33, 17), 1))
                    load.assert_not_called()

    def test_ordinary_text_does_not_decode(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "sampling parameters" * 20)
        info.add_text("comment", "compressed comment", zip=True)
        info.add_itxt("description", "ordinary metadata", zip=True)
        data = png(pnginfo=info)
        data = append_chunk(data, b"tEXt", b"Comment\x00after IDAT")
        with patch.object(PngImagePlugin.PngImageFile, "load") as load:
            self.assertEqual(iu._probe(data), ("PNG", (33, 17), 1))
            load.assert_not_called()

    def test_exif_before_and_after_idat_all_orientations(self):
        for orientation in range(1, 9):
            exif = Image.Exif()
            exif[0x0112] = orientation
            for data in (png(exif=exif),
                         append_chunk(png(), b"eXIf", exif.tobytes()[6:])):
                with self.subTest(orientation=orientation):
                    self.assertEqual(iu._probe(data), old_probe(data))
                    self.assertEqual(iu._probe(data)[2], orientation)

    def test_xmp_text_variants_and_raw_exif(self):
        xmp = b'<rdf tiff:Orientation="6"/>'
        key = b"XML:com.adobe.xmp\x00"
        variants = [
            (b"tEXt", key + xmp),
            (b"zTXt", key + b"\x00" + zlib.compress(xmp)),
            (b"iTXt", key + b"\x00\x00\x00\x00" + xmp),
            (b"iTXt", key + b"\x01\x00\x00\x00" + zlib.compress(xmp)),
        ]
        exif = Image.Exif()
        exif[0x0112] = 8
        profile = b"\nexif\n100\n" + exif.tobytes().hex().encode("ascii")
        variants.append((b"tEXt", b"Raw profile type exif\x00" + profile))
        for kind, payload in variants:
            for data in (append_chunk(png(), kind, payload),
                         png()[:33] + chunk(kind, payload) + png()[33:]):
                with self.subTest(kind=kind, payload=payload[:30]):
                    self.assertEqual(iu._probe(data), old_probe(data))
                    self.assertIn(iu._probe(data)[2], (6, 8))

    def test_jpeg_and_malformed_inputs_keep_behavior(self):
        buf = io.BytesIO()
        exif = Image.Exif()
        exif[0x0112] = 6
        Image.new("RGB", (33, 17)).save(buf, format="JPEG", exif=exif)
        for data in (buf.getvalue(), b"", b"not PNG", png()[:40],
                     png()[:-15], append_chunk(png(), b"eXIf", b"bad")):
            self.assertEqual(iu._probe(data), old_probe(data))


if __name__ == "__main__":
    unittest.main(verbosity=2)
