# 性能优化计划与实测归档（含被证伪的假设）

> 本文件是"该往哪里优化"的唯一事实来源：每条都带**本机实测数字**、**开关名**、
> **验证方式**。被证伪的假设保留在案（附反证数据），避免以后重复踩。
> 靶子图库：`<图库根目录>`（约 4 万 个图片文件 / 约 3.8 万 张入整图索引 / 约 44.4 万 块瓦片）；
> 测试机：14 核 20 线程 / 16GB / RTX 4060 Laptop（sm_89, 8GB）/ ZHITAI TiPlus7100 NVMe。

---

## 一、已落地：P0 + P1（逐项实测）

基准台：`python -E devtools/ab_build_bench.py --mode {whole,tiles} --n 540 --dup 60`
（固定样本 600 张 = 540 索引内 + 60 对"内容完全相同"的重复文件；模型加载与计时分离；
交替/多次重复取均值。每项都做了**索引内容一致性验证**。）

| 步骤 | 开关（Config） | 瓦片 wall | 瓦片 CPU | 结论 |
| :--- | :--- | ---: | ---: | :--- |
| 基线（优化前） | — | 23.11 s | 218.7 核·秒 | 26.0 张/s |
| **P0-a** 块 md5 复用 | `tile_md5_reuse=True` | 22.89 → **14.40 s** | 218.8 → **166.7** | **−37% wall / −24% CPU**；哈希微基准 **15.1×**；块 md5/指纹/Hu 逐位一致 |
| **P0-b** cv2 直出 RGB | `cv2_rgb_direct=True` | 14.62 → **14.18 s** | 170.5 → **159.7** | −3.0% wall / −6.4% CPU；PNG 解码 −3.2%（配对测量）；80 张逐位一致 |
| **P1-a** 重复内容预过滤（瓦片） | `dedup_prefilter=True` | 13.93 → **12.41 s** | 162.0 → **146.0** | −10.9% wall / −9.9% CPU；跳过 60/600 张；索引内容一致 |
| **P1-a2** 重复内容预过滤（整图） | 同上 | 9.55 → **9.08 s** | 99.7 → **90.5** | −4.9% wall / −9.3% CPU；fp/hu/md5s/paths 逐位一致 |
| **P1-b** 归一化搬 GPU | `norm_on_gpu=True` | 12.73 → **11.75 s** | 144.7 → **137.5** | −7.7% wall / −5.0% CPU；特征**逐位一致**（最大差 0）；整图 −1.7%/−2.4%；缓存往返逐位一致 |
| **P1-c** 批大小 64→256 | `batch`（自动值） | 9.15 → **6.95 s** | 89.0 → 93.6 | **−24% wall**、吞吐 65.6→86.3 张/s（详见第二节） |
| **P1-d** 瓦片 tick 20→120 ms | `tile_flush_ms` | 12.41 → 12.18 | 146.0 → 147.7 | **无收益（噪声内）→ 保持默认 20 ms** |

**最终态（全部优化开启）**

| | 优化前 | 优化后 | 变化 |
| :--- | ---: | ---: | ---: |
| 瓦片建库（600 张样本） | 23.11 s / 26.0 张/s / 218.7 核·秒 | **11.68 s / 51.4 张/s / 141.0 核·秒** | **−49.5% wall / +98% 吞吐 / −35.5% CPU** |
| 整图建库（600 张样本） | 11.23 s / 53.4 张/s / 111.8 核·秒 | **6.93 s / 86.7 张/s / 94.8 核·秒** | **−38.3% wall / +62% 吞吐** |
| 真实图库·瓦片全量 | 1843.5 s / 21.7 张/s（2026-09-12） | **1033.6 s / 38.6 张/s**（2026-09-26） | **−44%** |
| 真实图库·整图全量 | 543.2 s / 69.4 张/s（2026-09-12） | **483.8 s / 82.4 张/s**（2026-09-26） | **−11%** |

**验证方法（重要）**：瓦片索引的落盘顺序 = 线程完成顺序（每次运行不同），因此
**不能逐位比对数组**，必须按「原图路径 + 框」建键做**顺序无关**比对
（`--compare-tiles`）；整图索引顺序确定，可直接逐位比对（`--compare-arrays`）。
精排特征在任何两次运行之间都有 FP16 量级差异（余弦偏差 ~3e-7、最大绝对差 ~2e-4），
这是 GPU autocast 的非确定性，**不是**改动引起的（用"同代码两次运行"对照证明）。

---

## 二、P1-c 的结论：不是"分块屏障"，是"批太小喂不饱解码池"

原本的假设是"投 64 张 → `task_q.join()` 全等 → 前向"这个屏障造成 GPU 空窗。实测：

- 流水线拆分（`devtools/probe_pipeline_split.py 400`）：**消费端只用 1.92 s / wall 5.94 s（32%），
  68% 在等解码批次** → GPU 本来就闲着，拆屏障省不出时间。
- 改测**批大小**（CPU 核·秒几乎恒定 87~94，说明做的是同一份活）：

| batch | 16 | 32 | 64 | 128 | 192 | **256** | 384 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wall(s) | 15.79 | 11.46 | 9.15 | 7.95 | 7.35 | **6.95** | 7.43 |
| 并行核 | 5.51 | 7.66 | 9.69 | 11.72 | 12.76 | **13.47** | 12.76 |
| 张/s | 38.0 | 52.3 | 65.6 | 75.5 | 81.7 | **86.3** | 80.8 |

→ 块越大，18~20 个解码线程越吃得饱；**256 是最优点**。已把 CUDA 自动批改为 256。
瓦片路径相反（batch=128 略差、**256 塌到 22.9 s**），故新增 `tile_fwd_batch`（默认 64）
把两条路径解耦。**结论：不做"滚动调度/拆屏障"那个高风险重写。**

---

## 三、P2 修订（本次重点）

### ❌ 删除：「超大 PNG 单张并行解码」

**证伪证据**（`gui_index_20260926-012457` 报告 + 建库顺序构成分析）：

1. 瓦片流水线**已经是图级 18 路并行**（`tile_decode_slots=18`）：一张 0.4~0.8 s 的大 PNG
   只占 1 个槽位，其余 17 个槽照常跑别的图，不构成阻塞。
2. 报告分段吞吐与"内容混合度"强相关（28 vs 68 张/s），CPU 核数随内容在 6.5~14.9 浮动
   → 瓶颈是**聚合解码算力**，不是单张解码串行化。把一张图解快，只是腾出一个槽位，总量不变。
3. "3000 多张时明显下滑"也不是流程退化：按建库顺序每 2000 张切桶统计，
   第 2001-4000 张那一段是**几乎纯小 JPEG 目录**（PNG 占比 0.1%、平均 1.75 MP、
   估算解码 28 ms/张），前后都是重 PNG 区（如第 1 桶 47% >12MP、估算 193 ms/张）。
   报告里 51~103 s（≈1770-5232 张）出现 68.4 张/s 的"爆发"就是这个轻量区，
   之后回到 28~49 张/s 才是正常水平。**整图报告同源同因**（40~80 s 出现 114 张/s 爆发）。

### ✅ 替换为：「PNG 解码库替换（可选开关，已实现）」

实测（`devtools/bench_png_decoders.py`，交替配对，单线程口径）：

| 实现 | 大 PNG（30 张最大，1.28 GB） | 随机 PNG（30 张） | 随机 PNG（10 张） | 逐位一致 |
| :--- | ---: | ---: | ---: | :--- |
| cv2（现状 libpng） | 506.1 ms | 52.1 ms | 43.6 ms | — |
| **imagecodecs png（libpng 1.6.58 + zlib-ng 2.3.3）** | **434.5 ms（1.16×）** | **47.9 ms（1.09×）** | 39.4 ms（1.11×） | ✅ 30/30 |
| libvips 8.18（pyvips） | 451.2 ms（1.12×） | — | 42.1 ms（1.04×） | ✅ 30/30 |
| imagecodecs spng（libspng 0.7.4） | 569.9 ms（0.89×） | — | 59.0 ms（0.74×） | 29/30（1 张失败） |
| Pillow 10.1 | 687.0 ms（0.74×） | 75.1 ms（0.69×） | 58.6 ms（0.74×） | ✅ 30/30 |

纯 DEFLATE 解压（PNG 解码 84% 的时间在这里）：Python zlib **1.2.13 → 317 MB/s**；
imagecodecs zlib 1.3.2 → 279 MB/s；**zlib-ng 2.3.3 → 426 MB/s（1.34×）**。

> ⚠️ **上面的 zlib-ng 1.34× 是"小缓冲微基准"口径，不能外推到大图**：换成真实 IDAT 流 +
> 总量口径后（见第五节），imagecodecs 包装的 zlib-ng **反而比 Python zlib 慢**（411.6 vs 492.4 MB/s），
> 原因是它的"输出大小未知"实现要反复扩容重试。请以第五节为准。

**已实现**：`cfg.png_decoder ∈ {"cv2", "imagecodecs", "pillow"}`（CLI `--png-decoder imagecodecs`；
未安装 imagecodecs 时自动回退 cv2）。**默认仍是 cv2**，因为真实收益是：

| 场景 | 结果 |
| :--- | :--- |
| 格式矩阵逐位验证（114 张：RGB8 / RGBA8 / RGBA16 / 灰度 / 灰度+alpha / 调色板） | **114/114 逐位一致** → 无需重建索引 |
| 代表样本（600 张，40% PNG）单线程 PNG 解码 | cv2 145.0 ms/张 → imagecodecs 136.9 ms/张（**仅 1.06×**） |
| 该样本 PNG 解码占建库 CPU 比例 | **25.1%** → 理论省 1.4% |
| 建库 A/B（瓦片） | 11.81 s → 11.68 s（**−1.1% wall / −1.3% CPU**） |
| 建库 A/B（整图） | 6.70 s → 6.74 s（**噪声内，无差异**） |

> ⚠️ 灰度**不走** imagecodecs：cv2 的 `IMREAD_GRAYSCALE` 用 libpng 自己的 rgb→gray
> 定点转换，与 `cvtColor` 的舍入在彩色图上差 **±1 级**（RGB 差 0）。建库主路径的灰度
> 是对**已解码 RGB** 再 cvtColor，所以只要 RGB 逐位一致就完全一致 —— 灰度因此保持 cv2。
>
> 结论：这是一个**零风险可选加速**（约 1%），不是必须项；换 libspng 反而更慢，libvips
> 收益相近但依赖更重（`pyvips-binary` 自带 libvips 8.18.6 wheel，可装但不推荐）。

### 仍然保留的 P2 候选（未做，需重建索引 / 改语义）

| 项 | 预期 | 代价 |
| :--- | :--- | :--- |
| **libdeflate 自建 PNG 解码路径**（已实现，opt-in） | 解码 **1.14~1.15×**；建库 **−4%~−9% CPU**、瓦片 −3.8% 墙钟 | 需构建原生扩展（`devtools/build_native.py`）+ libdeflate.dll；默认仍 cv2 |
| 反滤波 SSE2 Paeth/Average（进一步） | 若能把 Paeth 从 6.3 → ~2 ms/MP，解码可望到 ~1.4× | 需 libpng 级 SIMD Paeth（易错，必须逐位验证）；收益递减 |
| cv2 预处理替代 torchvision(PIL)（PIL 预处理 ~4.5 ms/块） | −10~15% | 已实测 cosine 漂到 0.9945 → **必须重建 + 重跑召回评估** |
| 更激进的 JPEG 域缩放（现 ≤2560 全解、≤5120 半解） | −8~10% | 工作图尺度变化 → 指纹/瓦片几何与语义变更 |
| L3 有损 JPEG 缓存（重建期绕开 PNG 解码） | 重建期 4~10× | ~6 GB 磁盘 + 质量评估；首建仍需完整解码 |
| 命中框坐标系修复（53.2% 的图红框按 1/2、1/4 偏位） | 功能正确性 | 非性能项，见 README「已知问题」 |

---

## 四、被证伪 / 不适用的假设（保留存档）

| 假设 | 反证 |
| :--- | :--- |
| **显存做 L1 缓存**能缓解 GPU 空转 | 瓶颈在 CPU 解码供给（消费端只占 wall 32%）；8GB 显存里 ResNet18@256 批只占 1~2GB，显存不是约束 |
| **GPU 解码 PNG** | PNG 无硬件解码单元（NVDEC/NVJPG 无 PNG；nvImageCodec 的 GPU 扩展只有 nvjpeg/nvjpeg2k/nvtiff）；项目自身实测 **nvJPEG 比 CPU cv2 慢**（3.7MB JPEG：69.0 vs 38.6 ms） |
| **DALI** | GPU 路径只覆盖 JPEG；不产出本项目的 OTSU/Hu/MD5 |
| **DataLoader(num_workers/pin_memory)** | 自建线程池无 IPC 开销、解码线程释放 GIL 已够用；只有 pin_memory 那一半适用，实测收益 ~1% |
| **超大 PNG 单张并行解码** | 见第三节（图级已 18 路并行，吞吐随内容波动） |
| **瓦片 tick 调大**能提高吞吐 | 20→120 ms：12.41→12.18 s，噪声内 |
| **Pillow 比 cv2 快**（旧报告结论） | 实测 Pillow 只有 cv2 的 0.69~0.74×；旧结论应视为测量误差 |
| **每块 md5 必须重算整文件** | `md5(data+tag) == md5(data).copy().update(tag)` 逐位等价，15.1× |
| **归一化必须在 CPU** | GPU float32 归一化与 CPU 结果**逐位一致**（最大差 0） |
| **GDeflate（nvCOMP）能加速图库 PNG 解码** | 实测外来 IDAT 解码 **0/48**（`code=10 not NVCOMP_NATIVE`）；且其输出 CPU zlib 读不了 → 只能用于自建格式；含 D2H 端到端仅 1.35× CPU libdeflate，缓存还大 1.62 倍（见第五节组二） |
| **imagecodecs 能把 libdeflate 用在 PNG 上** | `deflate_decode` 文档明写"输出大小未知时用 zlib"，实测只吃 zlib 流；`_spng` 未链接 libdeflate（符号扫描确认） |
| **换 libdeflate 只是"换个库"** | inflate 本身 3.10×（1526 vs 492 MB/s），但 cv2 不暴露 inflate、libpng/libspng 需重编译 → 必须自建解码路径（Cython 反滤波）才能吃到 |

---

## 五、DEFLATE 解码方案逐组评测（2026-09-26）

> 背景：图库 PNG 的 IDAT 就是 zlib/DEFLATE 流。预分类实测（`devtools/classify_deflate_targets.py`）：
> **14,877 张 / 58.4 GB / 97,622 MP**（解出约 300~400 GB），以 **RGBA 8bit 为主（78.7%）**；
> 其余 24,966 张 / 74.7 GB 为 jpg/jpeg。PNG 解码是建库 CPU 的大头，所以逐个试"换熵解码器"。

### 组一：libdeflate 1.25 —— ✅ 有效（3.1~3.4×），但必须自建解码路径

统一口径（`devtools/bench_deflate_final.py`，真实 IDAT 流 · 每变体独立整轮遍历 · 轮间交替顺序 ·
64 张 / 1.275 GB / 3 轮 / 单张最大解出 65 MB）：

| 实现 | 吞吐 MB/s | 相对 Python zlib（预分配） | 相对现状 cv2 全解码 | 逐位一致 |
| :--- | ---: | ---: | ---: | :--- |
| Python zlib 1.2.13（输出大小未知） | 447.8 | 0.91× | 1.23× | 64/64 |
| Python zlib 1.2.13（`bufsize=n` 预分配） | 492.4 | 1.00× | 1.35× | 64/64 |
| imagecodecs zlib 1.3.2 | 295.4 | 0.60× | 0.81× | 64/64 |
| imagecodecs zlib-ng 2.3.3 | 411.6 | 0.84× | 1.13× | 64/64 |
| **libdeflate 1.25（scratch 复用）** | **1526.2** | **3.10×** | **4.18×** | 64/64 |
| libdeflate 1.25（每张新建缓冲） | 1343.1 | 2.73× | 3.68× | 64/64 |
| cv2 全 PNG 解码（`IMREAD_COLOR_RGB`，现状） | 365.4 | 0.74× | 1.00× | — |

独立复跑（`devtools/bench_deflate_group.py`，另一套样本组织方式）互相印证：
Python zlib **448.4**、imagecodecs zlib 298.3、zlib-ng 415.1、libdeflate 新建 **1316.4**、
libdeflate scratch **1542.1** MB/s，**逐位 64/64**，且 5 个变体峰值 RSS 全为 **386 MB（零增长）**。

**关键量：libdeflate 纯 inflate 比现状 cv2 的整个 PNG 解码（inflate+反滤波+色彩）还快 4.18×**，
所以换 inflate 的天花板很高（反滤波零成本时 4.18×；保留 16% 反滤波开销的推演 = 55.8 → 24.1 ms/张，
建库 CPU 约 **−17%**）。

落地路线（本机已具备：`cython` / `cffi` / MinGW `gcc`，无 MSVC/camke/ninja）：

| 路线 | 说明 |
| :--- | :--- |
| **Cython 扩展**（推荐） | ctypes 调 libdeflate inflate → C 层逐行反滤波（filter 3/4 行内串行，可多行并行）→ 只处理 RGB8/RGBA8，其他格式回退 cv2 |
| 重建 libpng/libspng 并链接 libdeflate | 需要 cmake/msvc；且现成 `imagecodecs._spng` **没有**链接 libdeflate（已用符号扫描确认） |
| 只换 Python 侧 inflate | 收益极小：建库解码走的是 cv2(C++)，Python 侧 inflate 只在缓存/工具链里 |

PNG 行滤波分布（`devtools/probe_png_filters.py`，64 张 / 150,944 行 / 1.27 GB 扫描线）：

| 滤波类型 | 行占比 | 字节占比 | 反滤波能否向量化 |
| :--- | ---: | ---: | :--- |
| Sub (1) | 55.77% | 58.88% | ✅ cumsum |
| Paeth (4) | 24.84% | 24.29% | ❌ 行内串行（需 C 层） |
| Up (2) | 8.44% | 7.44% | ✅ cumsum |
| Average (3) | 8.22% | 8.31% | ❌ 行内串行（需 C 层） |
| None (0) | 2.73% | 1.08% | ✅ 直通 |

分档看：**RGBA 各档几乎 100% 可向量化**（1-4MP/12+MP 都是 100%），RGB 各档约 70% 是 Paeth/Average。
即"图库主体（RGBA）好做，RGB/灰度需要 C 层兜底"。

### 组一·落地：libdeflate 自建 PNG 解码路径（已实现为 opt-in 开关，实测见下表）

`hybrid_search/png_fast.py` + `hybrid_search/native/`（Cython 包装 + C/SIMD 反滤波内核）。
构建：`python -E devtools/build_native.py`（Cython 生成 C，再用 MinGW gcc 直接编 .pyd；
本机无 MSVC/cmake 也能编）。开关：`png_decoder="libdeflate"`（CLI `--png-decoder libdeflate`），
**默认仍是 cv2**；缺原生模块/libdeflate.dll、或遇到非 8bit/非交错/调色板/16bit/灰度/tRNS/非法滤波
一律回退 cv2（回退路径与现状完全一致）。

覆盖与一致性（`devtools/verify_png_fast.py` + `devtools/test_png_filters.py`）：

| 项 | 结果 |
| :--- | :--- |
| 合成单元测试（5 种滤波 × RGB/RGBA × 22 种宽度含块边界与尾部 × 混合滤波） | **264/264 逐位一致**，SIMD 与纯标量实现结果相同；非法滤波/长度不符正确返回负值 |
| 真实样本逐位一致（多批随机 + 分层，合计 800+ 张） | **全部一致，0 张位差** |
| 覆盖率 | 按张数 **98.3%**，按像素 **100%**（回退仅灰度+alpha/坏头） |
| 建库索引语义 | 瓦片：块 md5/指纹/Hu **逐位一致**；整图：fp/hu/md5s/paths **逐位一致**（features 仅 FP16 前向的 2.2e-4 量级抖动，同代码复现亦然） |

解码级性能（`devtools/bench_png_fast.py`，颜色类型×像素档分层，每变体整轮独立遍历，3 轮）：

| 档位 | cv2 ms/张 | 新路径 ms/张 | 加速比 |
| :--- | ---: | ---: | ---: |
| RGBA/1-4MP | 18.06 | 12.64 | **1.43×** |
| RGBA/4-12MP | 37.28 | 29.94 | **1.25×** |
| RGBA/0-1MP | 6.17 | 4.86 | **1.27×** |
| RGBA/12+MP | 164.64 | 141.94 | 1.16× |
| RGB/0-1MP | 4.48 | 3.65 | 1.23× |
| RGB/1-4MP | 13.36 | 11.45 | 1.17× |
| RGB/4-12MP / 12+MP | 62.48 / 112.52 | 58.27 / 105.56 | 1.07× |
| **合计** | **52.37** | **46.04** | **1.14×（按像素加权 1.15×）** |

建库 A/B（`devtools/ab_build_bench.py`，固定样本 600 = 540 索引内 + 60 重复注入）：

| 路径 | 墙钟 | CPU 秒 | 吞吐 |
| :--- | ---: | ---: | ---: |
| 瓦片 cv2 → libdeflate | 13.89 → **13.36 s（−3.8%）** | 146.0 → 140.5（−3.8%） | 43.2 → 44.9 张/s |
| 整图 cv2 → libdeflate（首轮） | 6.96 → 6.87 s（−1.3%） | 93.6 → 87.7（−6.3%） | 86.3 → 87.3 张/s |
| 整图（同配置复跑一次） | 6.96 → **6.64 s（−4.6%）** | 93.6 → 85.2（−9.0%） | 86.3 → 90.3 张/s |

> ⚠️ **为什么没有达到"4.18× 上限"的预期**（实测 1.14×，建库 −4%~−9%）：
> 1. 反滤波是**行内串行**的：Paeth/Average（本图库约 33% 字节）是延迟受限循环（≈5 周期/像素），
>    SIMD 只能覆盖可向量化的 Sub/Up（≈67% 字节）。实测纯 Sub 图 0.44 ms/MP、Paeth 主导图
>    6.3 ms/MP，差 14 倍；把 Paeth 改成无分支**没有**改善（本来就不是分支预测问题）。
> 2. 4.18× 那个上限测于"inflate 友好"的样本（大量 Sub 行）；换成代表性分层样本后，
>    cv2 全解码 ≈ 21 ms/MP，libdeflate 纯 inflate ≈ 3~7 ms/MP。
> 3. libpng 是**逐行流式**解码（固定大小缓冲），我们是一次性解出整幅扫描线 →
>    大图要额外付整幅缓冲的页错误开销（已用线程本地缓冲复用把这段压掉，实测贡献约 1.14→1.15）。
> 4. 结论：这条路径**语义安全、确实更快，但收益是"个位数百分比"**，适合作为可选加速项；
>    想再往前一步，瓶颈已不在熵解码，而在**反滤波**（需要 libpng 级 SSE2 Paeth）或省掉整幅缓冲。

### 组一·内存特征与优化（跟随工程旋钮，边测边改）

探针：`devtools/probe_memory_profile.py`（父进程观测建库子进程 RSS 曲线/线程数，零侵入；
可对照 `--png-decoder`、`--prep-cache`、`--decode-workers` 等现有旋钮）。

首轮特征（瓦片建库，540+60 样本）：

| 配置 | 峰值 RSS | 峰值出现位置 | 均值 RSS |
| :--- | ---: | ---: | ---: |
| cv2（无缓存） | 2089 MB | 38.6% | 1522 MB |
| libdeflate（无缓存） | 2610 MB（**+25%**） | **95.2%** | 1617 MB |
| cv2 + 预处理缓存 | 2144 MB | 45.5% | 1380 MB |
| libdeflate + 预处理缓存 | 2615 MB | 93.8% | 1399 MB |

**诊断**：峰值出现在 93~95% 进度 = 内存**单调爬升**。原因是 18 个常驻解码线程把
inflate scratch 各自长到"自己见过的最大图"后**永不释放**（32 MB × 18 ≈ 576 MB，与 +520 MB 吻合）。

**已改**：`png_fast` 增加**全局 scratch 预算**（`set_scratch_budget_mb`，默认 256 MB）——
预算内线程复用（保住"反复分配省 12~14%"的收益），超预算的大图改为逐张分配、用完即弃。
复测：峰值 **2610 → 2545 MB（相对 cv2 由 +25% 降到 +16%）**，均值 1617 → **1397 MB（−14%）**；
解码速度与逐位一致性不受影响（合成 264/264、真实样本 0 位差、样本内解码 1.22×）。

**待改**：原生层的**行缓冲**（`pf_row_acquire`，每线程 2×wb，12MP RGBA 约 96 MB/线程）
未被该预算覆盖，是残余爬升的主因 → 需要给原生内核加"行缓冲上限/可不缓存"开关。

> 注：探针输出的是**整进程墙钟**（含模型加载与启动），run-to-run 波动可达 12%，
> 提速结论请以 `ab_build_bench` 的对照为准（瓦片 −3.8%、整图 −4.6%）。

### 🔴 未解决阻塞项：libdeflate 旁路在真实多线程建库下堆损坏（2026-09-26 收尾时定位）

**现象（可复现）**：`ab_build_bench.py --mode tiles --png-decoder libdeflate` **静默死掉**，
无输出、无 traceback、无结果 JSON；退出码 **`-1073740940` = 0xC0000374 `STATUS_HEAP_CORRUPTION`（堆损坏）**。
同一套里 `--png-decoder cv2` 完全正常（产出 `rg_cv2.json`）。复现记录见
`perf_reports/suite_20260926-210538/3_1.log`（0 字节）与 `perf_reports/regression.log`。

**为什么之前没暴露**：单线程口径全过——合成单元测试 264/264、真实样本逐位一致 200/200
（本次回归同样是 200/200、0 位差、覆盖按像素 100%）。也就是说**正确性在单线程下成立，
问题只出现在多线程/长时间真实负载**（18 路解码池、不同尺寸图交替）。

**当前处置**：`png_decoder` 默认仍是 `cv2`，libdeflate 保持 opt-in；在这些验证通过前
**不建议在任何正式建库中使用**：
1. 缩小复现（`--decode-workers 1` → 4 → 18 二分，确认是否与线程数相关）；
2. 重点排查候选：AVX2/SSSE3 打包与 Up 内核的越界写、`pf_row_acquire` 的预算记账与
   `tmp_prev/tmp_cur` 释放集合、`__declspec(thread)` 行缓冲在多线程下的行为；
3. 建议加"指针/长度断言 + `_CrtCheckMemory` 等价物（MinGW 下用 `-fsanitize=address` 重编）"
   跑到崩溃点；ASan 版本最容易一击定位；
4. 通过后再跑：瓦片/整图 A/B（含索引语义比对）+ 峰值 RSS 复测。

### 组二：GDeflate（NVIDIA nvCOMP 5.3）—— ❌ 不适用于本图库

环境：`nvidia-nvcomp-cu12==5.3.0.16` 自带官方 Python 绑定（`nvidia.nvcomp`），
GPU = RTX 4060 Laptop（8 GB，驱动 610.88），nvCOMP 5.3.0 / CUDA 12.9 运行时。

| 判别问题 | 实测结论 |
| :--- | :--- |
| Q1 能否直接解图库里的 IDAT（外来 zlib/deflate 流）？ | **不能**：zlib 流 0/48、raw deflate 0/48；报错 `code=10 The passed data was not compressed with NVCOMP_NATIVE bitstream kind` |
| Q2 GDeflate 输出能否被 CPU zlib 读？ | **不能**：NVCOMP_NATIVE / RAW / WITH_UNCOMPRESSED_SIZE 三种位流，zlib(wbits=15) 与 raw inflate 全部报错 |
| Q3 只做自建缓存格式时吞吐多少？ | 见下表（48 张 / 0.97 GB / 3 轮，逐位 48/48、144/144、144/144 全一致） |

| 口径 | 吞吐 | 单张中位 | 对照 CPU libdeflate |
| :--- | ---: | ---: | ---: |
| GDeflate GPU 纯解码（数据已在显存） | **4.71 GB/s** | 3.0 ms | 2.98× |
| GDeflate + H2D | 3.78 GB/s | 3.3 ms | 2.39× |
| GDeflate + H2D + D2H（CPU 要像素） | 2.13 GB/s | 4.4 ms | **1.35×** |
| CPU libdeflate（同一批自建格式数据） | 1.58 GB/s | 6.3 ms | 1.00× |
| CPU zlib（同一批数据） | 0.43 GB/s | 27.2 ms | 0.27× |
| GDeflate GPU 压缩（写缓存侧） | 0.31 GB/s | 18.9 ms | — |
| zlib 压缩（写缓存侧） | 0.03 GB/s | 458.3 ms | — |

pinned 传输带宽：H2D **12.05 GB/s**、D2H **12.26 GB/s**（256 MB，即 PCIe 已是硬上限）。
缓存体积（同批像素）：zlib 压缩比 5.3:1 vs GDeflate **3.28:1 → 缓存盘占用是 zlib 的 1.62 倍**。

**判读**：GDeflate 只能吃自己压的数据，而图库 14,877 张 PNG 的 IDAT 是外来 zlib 流 ——
**作为"替换现有解码器"的方案直接出局**。若要为它建全分辨率像素缓存（解出 300~400 GB，
压缩后仍约 100 GB 且比 zlib 大 62%），还要把反滤波也搬到 GPU（filter 3/4 行内串行），
换来"含 D2H 时仅 1.35× CPU libdeflate"的端到端收益 —— **不划算**。

### 口径修正记录（重要，避免以后误引）

1. **MB/s 数学错误（已修）**：`rate = total_raw / sum(时间)` 在 3 轮时把时间累加了 3 次，
   把吞吐低估约 3 倍（同一实现 142 → 448 MB/s 就是它）。**相对比值不受影响**，绝对数已全部重跑修正。
2. **真实 IDAT ≠ Python 重压流**：`devtools/bench_inflate_isolate.py` inflate 的是
   `zlib.compress(payload, 6)` 的流，Huffman/LZ77 结构与图库真实 IDAT 不同（同实现可差 3×），
   **该脚本的绝对值作废**，只保留"输出缓冲分配"这一变量的对照结论。
3. **imagecodecs 的 `deflate_decode` 不是 libdeflate 快路径**：其文档明写"输出大小未知时用 zlib"，
   实测它连 raw deflate 都拒绝、只吃 zlib 流 —— 所以 imagecodecs 无法把 libdeflate 用到 PNG 上。
4. **只比中位数会骗人**：cv2 的"中位解码时间"和 inflate 的"总字节/总时间"不是同口径，
   必须都用总量口径（本节的 365.4 / 1526.2 均为总量口径）。
5. **同名文件**：图库里不同目录的同名图很常见，按 basename 做基准键会误报"逐位不一致"；
   必须按完整路径。

复现命令见第七节。

---

## 六、JPEG 解码器对照（C2 · 2026-09-28）—— ❌ 建库级净负，不采纳

**问题**：JPEG 解码占整图建库 CPU 约 1/4（实测见下），换一个更快的 libjpeg-turbo 构建能否把这块拿下来？
**候选**：PyTurboJPEG（libjpeg-turbo 3.2.0，conda-forge 预编译 DLL，纯 Python 解包，不跑安装器）、
imagecodecs jpeg（同为 libjpeg-turbo 3.2.0）与 imagecodecs mozjpeg（4.1.5）；
**基线**：cv2 捆绑的 libjpeg-turbo 3.0.3（现状）。

### 6.1 全库普查（只读表头，约 2.5 万张 JPEG）

| 项 | 实测 | 说明 |
| :--- | :--- | :--- |
| 编码类别 | 基线 96.3% / **渐进 3.7%** | 上一轮"样本 0% 渐进"是抽样偏差，全库实测有 3.7% |
| EXIF 方向 | **100% = 1** | 换库不需要自己做 numpy 转正（但仍对非 1 一律回退，见 6.3） |
| 采样比 | 4:4:4 **79.9%** / 4:2:0 17.2% / 灰度 1.1% | 以 4:4:4 为主，没有色度上采样开销 |
| DCT 缩放档 | 全解 49.9% / 1/2 40.0% / 1/4 10.1% / 1/8 0.1%（**占像素 90.8%**） | **关键**：一半张数、九成像素走采样域缩放 |

最后一行决定了候选的成败：**imagecodecs / mozjpeg 不支持 DCT 域缩放解码**，
只能全尺寸解，在这三个档上像素量多 2~8 倍，因此结构性吃亏（实测 0.88×）。

### 6.2 解码级：TurboJPEG 更快，且逐位一致

配对 A/B（按缩放档分层抽样、每张图同轮跑全部解码器、顺序按 图号+轮号 轮转、3 轮取逐张比值中位）：

| 解码器 | ms/张 | 配对中位比 | 逐位一致（基准=现状） |
| :--- | ---: | ---: | :--- |
| cv2 现状（libjpeg-turbo 3.0.3） | 53.76 | 1.000× | 46/46 |
| **TurboJPEG 3.2.0（同缩放档）** | **48.17** | **1.125×** | **46/46** |
| imagecodecs jpeg 3.2.0（全尺寸） | 74.43 | 0.884× | 15/46（形状不同，其余档位不可比） |
| imagecodecs mozjpeg（全尺寸） | 74.15 | 0.882× | 15/46（同上） |

分档：全解 **1.15×** / 1/2 **1.19×** / 1/4 **1.07×** / 1/8 **1.10×** —— 每一档都更快。
（口径与第五节一致：单线程、按图独立配对；整轮总时长比与配对中位比同向。）

### 6.3 逐位校验：0 位差，但查出一个真实契约差异

逐位一致是换库的准入线（指纹/Hu/ResNet 输入由像素直接决定，有位差就要重建索引）：
真实图库 **365/365**、合成格式矩阵 **67/67**（渐进/灰度/4:4:4/4:2:2/4:2:0/4:4:0/CMYK/极小图）、
18 路线程池压测 340 张 0 位差、EXIF 1..8 全部按预期回退。

**过程中查出的差异（重要）**：**截断的 JPEG** 两者行为不同 ——
cv2 的 `imdecode` 返回 `None`（文件被跳过、不入索引），TurboJPEG 却会补边解出半张图。
若不拦，这类文件会**从"不入索引"变成"入索引"**，索引内容就变了。
解法是零成本的结构守卫：**尾部不带 EOI（`FFD9`）一律回退 cv2**。
实测全库只有 69 张缺 EOI，且它们在 cv2 下都能正常解出 —— 即这条守卫在当前语料上不改变任何一张结果。
（先试过 TurboJPEG 的 `TJFLAG_STOPONWARNING`，实测对 libjpeg 的 "Premature end of JPEG file" 警告**无效**，
截断文件照样解出，不能替代该守卫。）

### 6.4 建库级配对 A/B：净负，且可复现

零侵入 A/B（monkey-patch 替 `decode_rgb`，不动主代码；固定 600 张样本、4 轮交替、生产配置 `png_decoder=libdeflate`）：

| 轮 | 现状墙钟 | TurboJPEG 墙钟 | Δwall | ΔCPU |
| :--- | ---: | ---: | ---: | ---: |
| 1–4 | 7.6 / 7.1 / 7.0 / 7.2 s | 9.6 / 9.3 / 9.2 / 8.9 s | **+28.6%（中位）** | **+22.5%（中位）** |

**索引数组逐位一致**（`fp`/`hu`/`features` 最大绝对差 0）。三次独立复跑（含 `png_decoder=cv2`、
含页错误计数）结论同向：ΔCPU **+20%~+22.5%**、Δwall **+24%~+29%**。

**阶段归因把回归钉死了**（同一 A/B 内同时给 `decode_rgb` / `read_bytes` / 前向装计时器）：

| 阶段 | 变化 |
| :--- | :--- |
| JPEG 解码 | 22.8 → 18.4 核秒（**1.24× 变快**） |
| 读盘 | 2.4 → 2.1 核秒（不变） |
| 前向 | 0.7 → 0.7 核秒（不变） |
| **PNG 解码（一行没改）** | **69.8 → 99.9 核秒（+43%）** ← 回归全部落在这里 |

PNG 占解码 CPU 的 **75.4%**，所以它被拖慢 43% 足以吃掉 JPEG 省下的那点。

**这不是解码器本身慢**：把建库流水线整个剥掉、只留 18 路线程池的隔离实验里，
纯 JPEG 负载下 TurboJPEG **Δwall −16.0% / ΔCPU −20.3%**，混合负载 **−8.5% / −11.8%**（PNG 未被拖慢）。
所以机制是**进程内交互**。已排除的方向：内存页错误只多 6%（不足以解释 +21% CPU）；
`READ/FORWARD` 均未变。**根因尚未定论** —— 留作后续：怀疑方向是
本项目用户态与 `tj3Init`/`tj3Destroy`（PyTurboJPEG 每张新建/销毁句柄）带来的分配器/GIL 交互，
但未取得直接证据。

### 6.5 结论

**不采纳**（判据是"<5% 收益不采纳"，而这里是**负**收益）。
解码级 1.125×、逐位一致这两条是成立的，若将来要重开这条线，**必须先用持久句柄的 ctypes 绑定
（而非 PyTurboJPEG 的每张 init/destroy）重做 6.4 的建库级 A/B**，只看建库级结果。

**顺带记录的口径坑**：`devtools/ab_build_bench.py` 自己的 `--png-decoder` 默认是 `cv2`，
而生产默认是 `libdeflate` —— 两边不一致会让结论跑偏，做 JPEG 类 A/B 时应显式指定。

---

## 七、复现命令

```bat
:: 建库 A/B 基准台（固定样本、模型加载与计时分离、内容一致性比对）
python -E devtools/ab_build_bench.py --mode tiles --label base --n 540 --dup 60
python -E devtools/ab_build_bench.py --compare-tiles base p0a      :: 顺序无关比对
python -E devtools/ab_build_bench.py --mode whole --label w --n 540 --dup 60
python -E devtools/ab_build_bench.py --compare-arrays w w2

:: 微基准（交替配对，消时钟漂移）
python -E devtools/micro_bench.py md5-tile     :: P0-a
python -E devtools/micro_bench.py png-rgb      :: P0-b

:: PNG 解码库对比 / 正确性验证 / 熵解码占比
python -E devtools/bench_png_decoders.py 30 2
python -E devtools/verify_png_decoder.py 2500 40
python -E devtools/probe_entropy_split.py 24
python -E devtools/probe_order_mix.py 2000 4  :: 建库顺序的图片构成（解释吞吐波动）
python -E devtools/probe_pipeline_split.py 400

:: 缓存端到端一致性（冷 vs 热，逐位一致）
python -E devtools/verify_prep_cache.py 240

:: DEFLATE 方案评测（2026-09-26）：权威口径 / 独立复跑 / GDeflate / 前置分类 / 滤波分布
python -E devtools/bench_deflate_final.py 8 3 256   :: 统一口径（真实 IDAT，权威）
python -E devtools/bench_deflate_group.py 8 3 256   :: 独立复跑 + 缓冲策略 + 峰值 RSS
python -E devtools/bench_gdeflate.py 6 3 192        :: GDeflate（nvCOMP）可用性 + 吞吐 + 传输
python -E devtools/classify_deflate_targets.py      :: 全库预分类（PNG 分档 / 非 PNG）
python -E devtools/probe_png_filters.py 8 256       :: PNG 行滤波类型分布（自建解码路径可行性）
python -E devtools/probe_cv2_vs_inflate.py 6 192 2  :: cv2 全解码 vs 纯 inflate 的总量口径上限
python -E devtools/bench_inflate_isolate.py 6 3 192 :: ⚠️ 已作废（重压流口径），仅留存档

:: libdeflate 自建 PNG 路径（落地版）：构建 → 单元测试 → 一致性 → 性能图 → 建库 A/B
python -E devtools/build_native.py --force          :: Cython + MinGW gcc 编 _pngfast
python -E devtools/test_png_filters.py              :: 合成 264 用例（含 SIMD vs 标量交叉校验）
python -E devtools/verify_png_fast.py 2500 40 800   :: 格式矩阵 + 大样本逐位对齐
python -E devtools/bench_png_fast.py 6 3 256        :: 解码级逐档性能图（HTML/JSON）
python -E devtools/ab_build_bench.py --mode tiles --label t_ldf --n 540 --dup 60 --png-decoder libdeflate
python -E devtools/ab_build_bench.py --mode whole --label w_ldf --n 540 --dup 60 --png-decoder libdeflate

:: JPEG 解码器对照（C2，2026-09-28）：取 DLL → 普查 → 解码级性能图 → 逐位校验 → 建库 A/B → 隔离对照
python -E devtools/fetch_turbojpeg.py --check          :: conda-forge 纯 Python 解包 turbojpeg.dll（不跑安装器）
python -E devtools/probe_jpeg_census.py                :: 全库 JPEG 表头普查（缩放档 / EXIF / 基线-渐进）
python -E devtools/bench_jpeg_decoders.py 15 3         :: 解码级配对 A/B + 逐位校验 + HTML 性能图
python -E devtools/verify_jpeg_decoder.py 3000 40      :: 逐位校验（真实 + 合成 + 18 路压测 + 截断/损坏回退）
python -E devtools/ab_jpeg_turbo.py --mode whole --rounds 4 --n 540 --dup 60 --png-decoder libdeflate
                                                       :: 零侵入建库级配对 A/B（含阶段归因）
python -E devtools/ab_jpeg_mt.py 200 18 3              :: 剥掉流水线的 18 路解码对照（区分"交互"与"解码器慢"）
```

## 七、环境注意（踩过的坑）

1. `python -E` 是本仓库脚本/基准的**推荐运行方式**：本机曾存在用户级
   `PYTHONPATH=<第三方 python 目录>`（Siemens NX 留下），它让 `_ctypes.pyd` 从 NX 的
   Python 加载 → `import ctypes`/`numpy` 直接失败（已删除，原值备份在用户目录
   `.dsh/` 下）。NX Open 脚本请按进程设置 `PYTHONPATH`，不要设全局。
2. **numpy 必须是 1.26.x**：torch 2.4.0+cu124 与 numpy 2.x 不兼容
   （`Could not load numpy`/`Could not infer dtype of numpy.float32`）。
   装第三方库（如 imagecodecs）时注意别被连带升级。
3. `.ps1` 文件在本机被安全软件锁死（写得进、读不出），需要脚本时用
   `-EncodedCommand` 或 `.log` 承载脚本体。
4. **PowerShell 的 `cd` 不影响 `[System.IO.File]` 的相对路径**：.NET 用进程启动目录解析，
   所以脚本里做文件替换要用绝对路径（子进程 `python` 反而继承 PowerShell 的当前位置）。
5. **长基准要脱离会话跑**：`Invoke-CimMethod Win32_Process Create` + `cmd /c ... > log 2>&1`
   可完全脱离（父进程结束不影响），配合脚本内 Tk 置顶小窗看进度；
   脚本里用 `BENCH_LINGER` 控制跑完后的窗口滞留秒数（0 = 立刻关，便于串跑多个基准）。
6. 重定向到文件的 stdout 是**块缓冲**，跑到一半看不到内容属正常；判定进度请看
   `perf_reports/*.json` 是否出现，或看进程 CPU 时间。
7. 本机无 MSVC / cmake / ninja；有 **MinGW gcc**（`<MinGW 安装目录>\bin\gcc.exe`，已在 PATH）、
   `cython`、`cffi` —— 要做 C 扩展请按 MinGW 路线（`setuptools` 指定
   `--compiler=mingw32`），或直接用 ctypes + 现成 DLL。
