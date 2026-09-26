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
    # 特征提取批大小（同时决定融合建库的"分块"粒度）；0 表示自动
    # （cuda:256 / cpu:16）。实测（600 张真实图，CPU 核·秒恒定 87~94）：
    #   批 16/32/64/128/192/256/384 → wall 15.79/11.46/9.15/7.95/7.35/6.95/7.43 s
    # 即批越大解码线程池越吃得饱（并行核 5.5→13.5），256 为最优点；
    # 注意瓦片路径**不跟随**此项（见 tile_fwd_batch，它另有最优）。
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
    # 解码前按内容 md5 预过滤重复副本（True=推荐）：同内容文件只处理第一张，
    # 其余跳过解码+切块+特征（结果与“处理后被 add_results 丢弃”完全等价，
    # 已用 600 张样本验证：块 md5 / 指纹 / Hu 逐位一致）。真实图库实测有
    # 2,169/39,854 张（5.4%）属于此类。
    dedup_prefilter: bool = True
    # 瓦片索引：块 md5 复用「整文件哈希对象」（True=推荐）。
    #   True ：每张原图只哈希一次，N 块用 md5.copy()+update(box) 派生（实测 15.4×，
    #          取值与逐块 md5(整文件字节+框) 逐位一致，索引无需重建）
    #   False：旧行为（每块重新哈希 + 拷贝整份文件字节），仅作回退/对照用
    tile_md5_reuse: bool = True
    # 粗筛特征提取的并行线程数（0=自动，自动=min(8, CPU核数)）
    workers: int = 0
    # ResNet 精排的“图像解码/预处理”并行线程数（0=自动）。
    # 解码在 CPU 侧与 GPU 前向天然重叠，多线程可显著提升吞吐
    decode_workers: int = 0
    # PNG 解码器："libdeflate"（**默认**，2026-09-26 起：libdeflate 解 IDAT +
    # 原生 SIMD 反滤波，直出紧凑 RGB；实测纯 inflate 比 cv2 全解码快，
    # 只覆盖 8bit 非交错 RGBA（按像素覆盖 100%），其余与依赖缺失一律自动回退 cv2，
    # 逐位一致性已验证 0 位差；需要 hybrid_search/native 下的 _pngfast 扩展与
    # libdeflate.dll，构建见 devtools/build_native.py，实测见 docs/perf-plan.md 第五节）
    # / "cv2"（OpenCV 捆绑 libpng，回退目标）/ "imagecodecs"（libpng 1.6.58
    # + zlib-ng 2.3.3：实测大 PNG 快 21%、小 PNG 快 11%，输出与 cv2 逐位一致，
    # 需 pip install imagecodecs，未安装时自动回退 cv2）/ "pillow"（安静但慢 0.70~0.74x）
    png_decoder: str = "libdeflate"
    # libdeflate 路径的每线程 scratch 缓冲上限（MB）：复用可省大图反复分配的
    # 12~14% 开销，但 18 个解码线程同时持有时会占内存，故设上限，超过的图逐张新建。
    png_fast_scratch_mb: float = 32.0
    # 屏蔽 libpng 的 iCCP/cHRM stderr 噪音（只吞已知噪音行，其余原样转发）。
    # 目的：终端不再刷屏、不淹没真实错误；不影响解码结果与构建流程。
    silence_png_warnings: bool = True
    # cv2 解码直出 RGB（IMREAD_COLOR_RGB，OpenCV ≥4.5.5）：省掉一次全图
    # BGR→RGB 拷贝。实测 40 张真实 PNG（均 4.06 MB）99.9→89.7 ms/张（-10.2%）、
    # 逐位一致；只有非域缩放档受益（REDUCED_* 没有 RGB 变体）。False=回退旧行为。
    cv2_rgb_direct: bool = True
    # 归一化搬 GPU（True=推荐，仅 CUDA 生效）：transform 只做
    # Resize(256)+CenterCrop(224)+ToTensor（即 uint8/255），(x-mean)/std 在 GPU 上
    # 以 float32 完成（放在 autocast 之前），省掉 CPU 侧每张 15 万次浮点乘加；
    # 预处理缓存随之只存 round(t*255) 的 PNG（比"逆向归一化"更简单且逐位无损）。
    # False=回退旧行为（CPU 归一化 + 缓存存逆向归一化结果）。
    norm_on_gpu: bool = True
    # 大图(>12MP)解码并发上限：这类图 RGB 峰值 36MB~100MB+/张，
    # 过高并发会推高内存(页回收反而降速)。14 物理核/16GB 内存机建议 16~20
    big_decode_conc: int = 16
    # 瓦片(局部)建库：第一级“同时解码的原图数”上限(大图还受上面并发门约束)
    tile_decode_slots: int = 18
    # 瓦片(局部)建库：瓦片不足一批时的最长等待毫秒(批发送间隔)。
    # 越小 GPU 批越碎(1-2 行小批唤醒多)，越大批越整但延迟略高
    tile_fwd_batch: int = 64
    # （说明）瓦片前向批大小单独一项：整图路径实测"块越大越好"（4096 核·秒
    # 不变、并行核 5.5→13.5），而瓦片路径实测 batch=256 会拖垮吞吐（22.9s vs
    # 11.8s）——两者解耦，各自取各自的最优。
    tile_flush_ms: int = 20
    # 索引存储格式：False=npz（旧，读取时整体解压进内存）；
    # True=侧车 .npy（可 mmap 懒加载：444k 瓦片库实测加载 3.6s-><1s、
    # 常驻内存 1.3GB->约 0.3GB）。仅影响“新建/重建”索引的写盘格式，
    # 已有索引可用 compact 命令就地转换。
    # 默认 True（2026-09-26 收敛为默认行为，UI 不再单列开关：纯收益，无代价）。
    fast_load: bool = True
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


