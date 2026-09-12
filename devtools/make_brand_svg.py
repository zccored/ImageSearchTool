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
"""生成品牌图 SVG（banner / social / logo / icon），视觉元素全部取自项目自身特征：

  * 九宫格 = 瓦片索引，按 **25% 重叠**排布（对应 tile_index 的 512px+25% 切块协议）
  * 两条连线 + 光锥 = 把选中瓦片"放大"到前景（局部命中那条链路）
  * 前景彩色方块 = GUI 可视化面板的 **16×16 采样象限马赛克**
  * 红框 = 命中框（GUI 原色 #ff4040）；配色沿用性能图报告主题

用法: python devtools/make_brand_svg.py [输出目录]      # 默认 docs/brand
确定性输出：同版本重复运行逐字节一致（马赛克用固定种子，两个尺度图案完全对应）。
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

# 项目自身配色（取自 perfscope 报告主题、可视化面板、命中框）
BG_0, BG_1, BG_2 = "#0b1017", "#10141a", "#0a0e14"
INK, MUTED, DIM = "#e6edf3", "#9aa7b0", "#6b7885"
BLUE, BLUE_L, CYAN = "#3b82f6", "#60a5fa", "#22d3ee"
REPORT = ["#9fd0ff", "#ffd28f", "#7fdb9a", "#ff7b4a", "#d17d3d",
          "#8ab4ff", "#74e2a0", "#7bd1ff"]
HIT = "#ff4040"
FONT = ("'Segoe UI','Microsoft YaHei UI','PingFang SC',"
        "'Noto Sans SC',sans-serif")
MOSAIC_SEED = 20260912
TILE_SPAN = 400.0            # 马赛克内部绘图坐标系（前景瓦片一律按此画再缩放）


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def shade(color, f):
    r, g, b = hex2rgb(color)
    return "#%02x%02x%02x" % (max(0, min(255, int(r * f))),
                              max(0, min(255, int(g * f))),
                              max(0, min(255, int(b * f))))


def mosaic(x, y, size, n, seed=MOSAIC_SEED):
    """n×n "采样象限"彩色马赛克：中心亮、边缘暗（像一张缩略图）。"""
    rng = random.Random(seed)
    step = size / n
    out = []
    for j in range(n):
        for i in range(n):
            color = REPORT[((i * 7 + j * 13) % len(REPORT))]
            cx, cy = (i + 0.5) / n, (j + 0.5) / n
            d = (((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5) / 0.7071
            f = (1.12 - 0.72 * (d ** 0.85)) * (0.92 + 0.16 * rng.random())
            out.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" '
                       'fill="%s"/>' % (x + i * step, y + j * step,
                                        step + 0.6, step + 0.6,
                                        shade(color, f)))
    return "".join(out)


def defs(extra=""):
    return f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BG_0}"/><stop offset=".55" stop-color="{BG_1}"/>
    <stop offset="1" stop-color="{BG_2}"/>
  </linearGradient>
  <radialGradient id="glowTile" cx=".5" cy=".5" r=".5">
    <stop offset="0" stop-color="{BLUE}" stop-opacity=".30"/>
    <stop offset="1" stop-color="{BLUE}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowGrid" cx=".5" cy=".5" r=".5">
    <stop offset="0" stop-color="{CYAN}" stop-opacity=".16"/>
    <stop offset="1" stop-color="{CYAN}" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="cellFill" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BLUE_L}" stop-opacity=".20"/>
    <stop offset="1" stop-color="{CYAN}" stop-opacity=".07"/>
  </linearGradient>
  <linearGradient id="cellHot" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{BLUE_L}" stop-opacity=".48"/>
    <stop offset="1" stop-color="{CYAN}" stop-opacity=".22"/>
  </linearGradient>
  <linearGradient id="beam" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{BLUE_L}" stop-opacity=".30"/>
    <stop offset="1" stop-color="{CYAN}" stop-opacity=".10"/>
  </linearGradient>
  <linearGradient id="lineG" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{BLUE_L}"/><stop offset="1" stop-color="{CYAN}"/>
  </linearGradient>
  <linearGradient id="word" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#bfe3ff"/>
  </linearGradient>
  <pattern id="paper" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M40 0H0V40" fill="none" stroke="#1b2531" stroke-width="1" opacity=".55"/>
  </pattern>
  <filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
    <feDropShadow dx="0" dy="16" stdDeviation="26" flood-color="#05080c"
                  flood-opacity=".62"/>
  </filter>
  <clipPath id="tileClip"><rect x="0" y="0" width="{TILE_SPAN:.0f}"
      height="{TILE_SPAN:.0f}" rx="20"/></clipPath>
  {extra}
</defs>"""


def grid_and_tile(gx, gy, cell, stride, tile_x, tile_y, tile, mosaic_n,
                  captions=False):
    """九宫格（重叠 25%）+ 两条连线/光锥 + 前景瓦片，坐标全参数化。"""
    cells = [(gx + i * stride, gy + j * stride) for j in range(3) for i in range(3)]
    hot = (gx + 2 * stride, gy + 1 * stride)          # 中右格 = 被放大的那一块
    span = 2 * stride + cell
    o = []
    o.append('<circle cx="%.0f" cy="%.0f" r="%.0f" fill="url(#glowGrid)"/>'
             % (gx + span / 2, gy + span / 2, span * 0.95))
    o.append('<circle cx="%.0f" cy="%.0f" r="%.0f" fill="url(#glowTile)"/>'
             % (tile_x + tile / 2, tile_y + tile / 2, tile * 1.25))
    for (x, y) in cells:
        if (x, y) == hot:
            continue
        o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="10" '
                 'fill="url(#cellFill)" stroke="%s" stroke-opacity=".34"/>'
                 % (x, y, cell, cell, BLUE_L))
    hx, hy = hot
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="10" '
             'fill="url(#cellHot)" stroke="#dff1ff" stroke-opacity=".92" '
             'stroke-width="2"/>' % (hx, hy, cell, cell))
    pad = cell * 0.26
    o.append('<g clip-path="url(#hotClip)" transform="translate(%.1f,%.1f)">%s</g>'
             % (hx, hy, mosaic(pad, pad, cell - 2 * pad, mosaic_n)))
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="14" '
             'fill="none" stroke="%s" stroke-opacity=".38" stroke-dasharray="9 8"/>'
             % (gx - 10, gy - 10, span + 20, span + 20, CYAN))
    x0, y0 = hx + cell, hy
    x1, y1 = hx + cell, hy + cell
    tx0, ty0 = tile_x, tile_y
    tx1, ty1 = tile_x, tile_y + tile
    o.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" '
             'fill="url(#beam)"/>' % (x0, y0, tx0, ty0, tx1, ty1, x1, y1))
    for (ax, ay, bx, by) in ((x0, y0, tx0, ty0), (x1, y1, tx1, ty1)):
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                 'stroke="url(#lineG)" stroke-width="3" stroke-linecap="round" '
                 'opacity=".95"/>' % (ax, ay, bx, by))
    for (px, py) in ((x0, y0), (x1, y1), (tx0, ty0), (tx1, ty1)):
        o.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#eaf6ff" '
                 'opacity=".95"/>' % (px, py, max(3.0, cell * 0.025)))
    o.append('<g filter="url(#soft)"><rect x="%.1f" y="%.1f" width="%.1f" '
             'height="%.1f" rx="20" fill="#11161d"/></g>'
             % (tile_x, tile_y, tile, tile))
    o.append('<g clip-path="url(#tileClip)" transform="translate(%.1f,%.1f) '
             'scale(%.4f)">%s</g>'
             % (tile_x, tile_y, tile / TILE_SPAN, mosaic(0, 0, TILE_SPAN, mosaic_n)))
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="20" fill="none" '
             'stroke="#ffffff" stroke-opacity=".16" stroke-width="2"/>'
             % (tile_x, tile_y, tile, tile))
    o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" '
             'stroke="%s" stroke-width="%.1f" stroke-opacity=".95"/>'
             % (tile_x + tile * 0.26, tile_y + tile * 0.22, tile * 0.52, tile * 0.52,
                HIT, max(2.0, tile * 0.009)))
    if captions:
        o.append('<text x="%.0f" y="%.0f" font-family="%s" font-size="22" fill="%s" '
                 'text-anchor="middle">瓦片内容 · 红框 = 命中框</text>'
                 % (tile_x + tile / 2, tile_y + tile + 44, FONT, MUTED))
        o.append('<text x="%.0f" y="%.0f" font-family="%s" font-size="22" fill="%s" '
                 'text-anchor="middle">瓦片索引 · 每格 512px · 重叠 25%%</text>'
                 % (gx + span / 2, gy + span + 42, FONT, MUTED))
    return "".join(o)


def banner_body():
    o = [grid_and_tile(gx=120, gy=245, cell=200, stride=150, tile_x=960,
                       tile_y=300, tile=400, mosaic_n=16, captions=True)]
    o.append('<text x="120" y="150" font-family="%s" font-size="88" '
             'font-weight="700" letter-spacing="1" fill="url(#word)">'
             'ImageSearchTool</text>' % FONT)
    o.append('<text x="124" y="198" font-family="%s" font-size="27" fill="%s">'
             '二值法粗筛 + ResNet 精排 · 本地混合图库检索</text>' % (FONT, MUTED))
    o.append('<rect x="1256" y="116" width="224" height="46" rx="23" fill="%s" '
             'fill-opacity=".10" stroke="%s" stroke-opacity=".40"/>' % (BLUE, BLUE_L))
    o.append('<text x="1368" y="146" font-family="%s" font-size="22" fill="%s" '
             'text-anchor="middle">v3.2 · AGPL-3.0</text>' % (FONT, REPORT[0]))
    chips = ["二值粗筛 64×64", "ResNet18 · 512 维", "LSH 候选 → 精排",
             "侧车 .npy 索引", "GPU 前向 0.41 ms"]
    cx = 120.0
    for c in chips:
        w = 34 + sum(21.5 if ord(ch) > 0x2000 else 11.6 for ch in c)
        o.append('<rect x="%.1f" y="820" width="%.1f" height="44" rx="22" fill="%s" '
                 'fill-opacity=".09" stroke="%s" stroke-opacity=".34"/>'
                 % (cx, w, BLUE, BLUE_L))
        o.append('<text x="%.1f" y="849" font-family="%s" font-size="22" fill="%s" '
                 'text-anchor="middle">%s</text>' % (cx + w / 2, FONT, REPORT[0], c))
        cx += w + 14
    o.append('<text x="1480" y="849" font-family="%s" font-size="21" fill="%s" '
             'text-anchor="end">本地运行 · 无云依赖</text>' % (FONT, DIM))
    return "".join(o)


def svg_open(w, h, extra=""):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d" role="img" aria-label="ImageSearchTool">'
            % (w, h, w, h)) + defs(extra)


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "brand")
    os.makedirs(out_dir, exist_ok=True)
    body = banner_body()

    banner = (svg_open(1600, 900, '<clipPath id="hotClip"><rect x="0" y="0" '
               'width="200" height="200" rx="10"/></clipPath>')
              + '<rect width="1600" height="900" fill="url(#bg)"/>'
              + '<rect width="1600" height="900" fill="url(#paper)" opacity=".5"/>'
              + body + '</svg>')

    s = 0.7111
    social = (svg_open(1280, 640)
              + '<rect width="1280" height="640" fill="url(#bg)"/>'
              + '<rect width="1280" height="640" fill="url(#paper)" opacity=".5"/>'
              + '<g transform="translate(%.1f,0) scale(%.4f)">%s</g>'
              % ((1280 - 1600 * s) / 2, s, body) + '</svg>')

    logo = (svg_open(760, 200, '<clipPath id="hotClip"><rect x="0" y="0" '
                'width="46" height="46" rx="6"/></clipPath>')
            + '<rect width="760" height="200" fill="url(#bg)"/>'
            + '<rect width="760" height="200" fill="url(#paper)" opacity=".45"/>'
            + grid_and_tile(gx=40, gy=52, cell=46, stride=34.5, tile_x=214,
                            tile_y=50, tile=100, mosaic_n=4)
            + '<text x="344" y="100" font-family="%s" font-size="44" '
              'font-weight="700" letter-spacing=".5" fill="url(#word)">'
              'ImageSearchTool</text>' % FONT
            + '<text x="346" y="138" font-family="%s" font-size="17" fill="%s">'
              '二值法粗筛 + ResNet 精排 · 瓦片索引 · 命中框回传</text>'
              % (FONT, MUTED) + '</svg>')

    icon = (svg_open(512, 512, '<clipPath id="hotClip"><rect x="0" y="0" '
                'width="88" height="88" rx="8"/></clipPath>')
            + '<rect width="512" height="512" rx="96" fill="url(#bg)"/>'
            + '<rect width="512" height="512" rx="96" fill="url(#paper)" opacity=".45"/>'
            + grid_and_tile(gx=40, gy=155, cell=88, stride=66, tile_x=300,
                            tile_y=185, tile=160, mosaic_n=12)
            + '<rect x="6" y="6" width="500" height="500" rx="92" fill="none" '
              'stroke="#ffffff" stroke-opacity=".10" stroke-width="2"/></svg>')

    for name, text in (("banner.svg", banner), ("social.svg", social),
                       ("logo.svg", logo), ("icon.svg", icon)):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("  %-11s %7.1f KB" % (name, os.path.getsize(p) / 1024))
    print("输出目录:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
