# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 全局配置 dataclass：建库/检索全部可调参数与默认值 + 参数一致性校验基准
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
全局配置：集中管理“二值法粗筛 + ResNet精排”混合检索系统的所有可调参数。

所有参数均可在 CLI 上通过 --xxx 覆盖，也可以在代码里直接改 Config 默认值。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 默认支持的图片扩展名（小写）
DEFAULT_EXTENSIONS: tuple = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)


@dataclass
class Config:
    # ---------------------------------------------------------------
    # 第一阶段：二值法粗筛（纯 CPU，毫秒级）
    # ---------------------------------------------------------------
    # 二值化前统一压缩到的边长（正方形），64 = 64x64 指纹 = 4096 bit
    coarse_size: int = 64
    # 高斯模糊核大小（奇数），用于降噪
    coarse_blur: int = 5
    # 是否启用轮廓 Hu 矩（7 维，旋转/缩放不变形状特征）
    use_hu: bool = True
    # 是否启用二值图像指纹（展平位图，汉明距离）
    use_fp: bool = True
    # 两类粗筛特征在综合分里的权重（总和不必为 1，仅相对大小有意义）
    hu_weight: float = 0.35
    fp_weight: float = 0.65
    # 若为 True，二值化后把“白像素多”的图像取反，
    # 使黑底白字与白底黑字类图像指纹一致（对照片类图库建议保持 False）
    invert_binary: bool = False

    # ---------------------------------------------------------------
    # 第二阶段：ResNet 精排（深度学习语义特征）
    # ---------------------------------------------------------------
    # 可选：resnet18 / resnet34 / resnet50 / resnet101 / resnet152
    # 大图库内存敏感时推荐 resnet18（512 维）；精度优先且库小时可用 resnet50
    model: str = "resnet18"
    # 设备：auto=有 CUDA 用 GPU 否则 CPU；也可显式写 cuda / cpu
    device: str = "auto"
    # GPU 推理使用 FP16 半精度（显存减半、速度更快）
    fp16: bool = True
    # 特征提取批大小；0 表示自动（cuda:64 / cpu:16）
    batch: int = 0

    # ---------------------------------------------------------------
    # 检索行为
    # ---------------------------------------------------------------
    # 粗筛阶段保留的候选数（推荐 200~500，太小可能漏召回）
    coarse_k: int = 300
    # 最终返回的结果数
    top_k: int = 10
    # 若查询图本身就在索引里，是否把它从结果中剔除（找相似而非找自己）
    exclude_self: bool = True

    # ---------------------------------------------------------------
    # 索引构建行为
    # ---------------------------------------------------------------
    # 是否预构建全库 ResNet 特征索引（True：检索最快；
    # False：每次查询只对粗筛候选实时抽特征，省内存但每查必算）
    store_fine: bool = True
    # 按文件 MD5 去重（同一张图重复出现只索引一次）
    dedup: bool = True
    # 粗筛特征提取的并行线程数（0=自动，自动=min(8, CPU核数)）
    workers: int = 0
    # ResNet 精排的“图像解码/预处理”并行线程数（0=自动）。
    # 解码在 CPU 侧与 GPU 前向天然重叠，多线程可显著提升吞吐
    decode_workers: int = 0
    # PNG 解码器："cv2"（默认，libpng 全尺寸，比 Pillow 快 ~1.3x；
    # 个别坏 iCCP 文件会有零星 stderr 警告）/ "pillow"（安静、较慢）
    png_decoder: str = "cv2"
    # 屏蔽 libpng 的 iCCP/cHRM stderr 噪音（只吞已知噪音行，其余原样转发）。
    # 目的：终端不再刷屏、不淹没真实错误；不影响解码结果与构建流程。
    silence_png_warnings: bool = True
    # 大图(>12MP)解码并发上限：这类图 RGB 峰值 36MB~100MB+/张，
    # 过高并发会推高内存(页回收反而降速)。14 物理核/16GB 内存机建议 16~20
    big_decode_conc: int = 16
    # 瓦片(局部)建库：第一级“同时解码的原图数”上限(大图还受上面并发门约束)
    tile_decode_slots: int = 18
    # 瓦片(局部)建库：瓦片不足一批时的最长等待毫秒(批发送间隔)。
    # 越小 GPU 批越碎(1-2 行小批唤醒多)，越大批越整但延迟略高
    tile_flush_ms: int = 20
    # 索引存储格式：False=npz（旧，读取时整体解压进内存）；
    # True=侧车 .npy（可 mmap 懒加载：444k 瓦片库实测加载 3.6s-><1s、
    # 常驻内存 1.3GB->约 0.3GB）。仅影响“新建/重建”索引的写盘格式，
    # 已有索引可用 compact 命令就地转换。
    fast_load: bool = False
    # 预处理缓存（L2 内存 + L3 磁盘）：缓存“PIL 处理后的 224 裁剪 + 粗筛指纹 +
    # md5”，重复建库时**完全跳过读原图与解码**（命中约 1.5ms/张，且逐位一致）。
    # 实测 CPU 侧 85ms/张 vs GPU 0.41ms/张 —— 这是把 GPU 真正喂饱的关键。
    prep_cache: bool = True
    # torch 推理线程数（0=保持 torch 默认；CPU 单线程前向通常最优，
    # GPU 场景该值无影响，由 CUDA 流自动调度）
    torch_threads: int = 0
    # 支持的图片扩展名
    extensions: tuple = field(default_factory=lambda: DEFAULT_EXTENSIONS)

    def coarse_feature_dim(self) -> int:
        """粗筛特征信息（供打印/存档用）"""
        dims = []
        if self.use_hu:
            dims.append(7)
        if self.use_fp:
            dims.append(self.coarse_size * self.coarse_size)
        return sum(dims)


