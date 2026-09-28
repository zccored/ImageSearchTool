# 01 · 完整交接（106 轮会话全量上下文）

> **用途**：新对话的**唯一权威交接文档**。包含本会话全部技术结论、代码状态、性能数据、环境坑、待办与用户已做决定。
> **读取顺序**：先读本文件 → 再读 `00_INDEX.md`（各文档用途索引）→ 需要细节时读 `02_SESSION_LOG.md`（逐轮时间线）。
> **生成时间**：2026-09-27（会话上下文接近上限时打包，未提交、未发 Release）

---

## 一、项目与工作方式

- **项目**：ImageSearchTool · 全栈图库管理器 v3.2beta，路径 `<仓库根>`（本机绝对路径已隐去）
- **远端**：`https://github.com/zccored/ImageSearchTool`，**git remote 名是 `ImageSearchTool`（不是 origin）**，分支 `main`
- **靶子图库**：`<图库根>`（约 4 万个图片文件 → 整图索引约 3.8 万张 + 瓦片索引约 44 万块，量级与公开 README 一致；精确数字已隐去）。**严禁改动靶子里面的内容**（只读；索引写独立目录）
- **检索架构**：两阶段 —— 粗筛（OTSU 64×64 打包指纹 + Hu-7）→ 精排（ResNet18 512d，L2 归一化余弦）
- **工作纪律（用户明确要求）**：
  1. 每项改动都要有**性能图/数字**作判据；性能结论只认 `ab_build_bench` 配对 A/B（探针的输出是整进程墙钟、波动可达 15%）
  2. 不要删除任何东西（文件/索引/数据）；不擅自强推、不覆盖用户线上改动
  3. 选项要用**对话框**给用户点（用户打字慢）
  4. 报告要**如实**，包括自己的操作失误

---

## 二、本会话做了什么（结论分级）

### ✅ 已定论并被采纳
| 结论 | 数据 |
| :--- | :--- |
| **libdeflate 1.25 解 zlib 流 3.10×** | 真实 IDAT、统一口径：libdeflate 1526.2 MB/s vs Python zlib 492.4 / imagecodecs zlib 295.4 / zlib-ng 411.6；逐位 64/64 |
| imagecodecs 的 `deflate_decode` **不是** libdeflate 快路径 | 其文档"输出大小未知时回退 zlib"，实测连 raw deflate 都拒；`_spng.pyd` 未链接 libdeflate |
| **GDeflate（nvCOMP 5.3）出局** | 外来 PNG IDAT **0/48 可解**（`code=10 not NVCOMP_NATIVE`）；其输出 CPU zlib 也读不了；自建格式才可用（GPU 纯解码 4.71 GB/s，含 H2D+D2H 仅 2.13 GB/s=1.35× CPU libdeflate），缓存还大 1.62 倍 |
| **libdeflate 旁路落地（opt-in → 后改默认开启）** | 自建原生扩展（Cython + SSE2/SSSE3/AVX2 反滤波）+ 自编 libdeflate 1.24 DLL；只接管 8bit 非交错 **RGBA**，其余与依赖缺失自动回退 cv2 |
| 解码级收益 | 分层样本 **1.14×**（像素加权 1.15×；RGBA 档 1.16~1.43×）；**不是**早先乐观的 4.18×（那是 inflate 友好样本） |
| 建库级收益（CPU 降、墙钟基本不动） | 瓦片 −3.8% wall/CPU；3.7 万张整图 **CPU −7.55%**、墙钟 +0.27%；旁路全库一跑 CPU 5811.8 核秒 vs cv2 6469.3（**−10.2%**）、墙钟 +3.1% |
| **缓存编码 PNG → libdeflate-6** | 读 **3.02×**、写 1.01×、体积 +8%；冷热建库索引逐位一致；缓存命中 **90 ms → 4 ms/张** |
| **多线程堆损坏已定位并修复**（重要） | `STATUS_HEAP_CORRUPTION 0xC0000374` / `0xC0000005`；根因 **MinGW 静态 TLS（`__declspec(thread)`）编进动态加载的 .pyd**；三种 SIMD 级别同样崩、`PF_NOCACHE=1` 即恢复正常 → 默认改为逐张 malloc/free（并不慢） |
| 长尾结构（PNG） | p50 67.5 / p95 556 / p99 1046.7 / max 1800 ms；**p99/p50=15.5**；**最慢 5% 吃掉 28.9% 解码 CPU**；≥4MP 仅 5.4% 张数 |
| 耗时分解（整图建库 600 张） | 解码 **77.94 核秒 = 88.9% CPU（7.52 核）**；主线程前向仅 **1.81 核秒**（GPU 均值 10~22%，**不是瓶颈**） |

### ❌ 被证伪 / 已回退（**不要再重复尝试**）
| 假设 | 反证 |
| :--- | :--- |
| **加权准入信号量**（大图按权重扣配额） | 事件驱动仿真：B=18×6=108 反而墙钟 **+59%**、有效核数 16.52→10.39；且权重按 p99 会**低估**大图 2.6 倍 |
| "18 路只跑 8.7~9.8 核 = 长尾把池子卡住" | 独立解码探针实测有效核数 **15.47**；建库只到 7.52 核是因为**解码只占建库 CPU 的一部分**，不是池子闲着 |
| **cv2.setNumThreads(2)** | 首测 −11.7% 墙钟是**噪声**；配对复跑（瓦片 3 次+整图 2 次，交替顺序）真实效应 ≤1%（瓦片 −0.7%、整图 −1.5%）。**默认保持 0，不固化** |
| 交叉引用（libdeflate inflate + stored 重封装 + cv2 反滤波） | **0.63×**，比现状 cv2 还慢 1.6 倍（level-0 重封装 + libpng 解 stored 不是 memcpy 级） |
| zlib-ng / imagecodecs / Pillow 换库 | imagecodecs 大 PNG 仅 1.16×，Pillow 0.70~0.74×；zlib-ng 经 imagecodecs 包装反而比 Python zlib 慢（0.84×） |
| nvJPEG 加速 JPEG | 实测 **比 CPU cv2 更慢**（3.7 MB JPEG：69.0 vs 38.6 ms） |
| 大图 RGB 也走旁路（本轮最后一步） | 全库实测 CPU 降但墙钟 +3.1%，**净负收益 → 已回退** |
| progressive JPEG 拖慢 | C3 普查：样本 **95% baseline、0% progressive** |

### ⏳ 待办（用户已排优先级，见第五节）

---

## 三、代码状态（极其重要）

### 已提交并发布（v3.2.0-beta，tag `v3.2.0-beta` = commit `1d97c07`）
```
1d97c07 fix: 侧车索引读取回退 + ResNet 权重下载不再用 tqdm（GUI 无效句柄致 WinError 1）
136bc6a devtools: GDeflate 基准加 CUDA 守卫
bc543f6 merge: 并入线上 README 更新（README 以线上为准）
c173bcd chore: peer_launcher 去掉写死的本机绝对路径
2f1d3cf docs: 更新 README 与性能归档；移除含本机路径的性能图并占位符化
455b490 perf: P0/P1 优化落地
dd9afc3 ui: PNG 解码器(旁路) 下拉 + 默认 libdeflate；两个选项收敛为默认
71c9da0 cache: 缓存编码 PNG→libdeflate-6 + store 侧车落盘降级修复
2021834 png: libdeflate 旁路 + 全局 scratch 预算
46e41be native: libdeflate 旁路原生扩展 + 自编 libdeflate 1.24 DLL
```
- **Release**：`https://github.com/zccored/ImageSearchTool/releases/tag/v3.2.0-beta`，4 个资产：`libdeflate.dll`(127,843B) / `_pngfast.cp312-win_amd64.pyd`(325,067B，**仅 CPython 3.12**) / `build_native.py` / `build_libdeflate.py`
- **用户已决定：本轮（09-27）实验是净负收益 → 不发 v3.3.0-beta，不合并**

### 工作区状态（未提交）
- **本轮实验已回退**：`_BIG_BYPASS_PX`（大图旁路）与 `cv2_threads`（config/engine/cli/ab_build_bench）全部清除，残留 0 处；`_BYPASS_CT = (_CT_RGBA,)` 已恢复
- 与 `1d97c07` 相比仅 4 个文件有差异（`cli.py`/`config.py`/`engine.py`/`png_fast.py`，内容已等同或更干净）+ **21 个未跟踪 devtools 脚本**（用户口径：**不提交 devtools**）
- **`branch/` 目录**：本次交接文档所在处（用户要求放在仓库内，注意决定是否提交）

### 关键实现位置
| 文件 | 作用 |
| :--- | :--- |
| `hybrid_search/png_fast.py` | 旁路解码：分块解析 + libdeflate(ctypes) inflate + 线程本地 scratch + 全局预算 256MB + 回退统计；`_BYPASS_CT=(RGBA,)` |
| `hybrid_search/native/_pngfast.pyx` + `png_filter_simd.c/.h` | 原生反滤波内核（Sub=移位相加、Up=向量加、RGBA→RGB=pshufb；filter 3/4 保持标量） |
| `devtools/build_native.py` | Cython→C→MinGW gcc 编 `.pyd`（无需 MSVC/cmake）；`--force` 重建 |
| `devtools/build_libdeflate.py` | 编 libdeflate.dll（给源码 tar.gz 路径）+ `--check` 行为验证 |
| `hybrid_search/prep_cache.py` | 缓存条目编码 = libdeflate-6（`head.codec`，签名含编解码标签，旧缓存自动失效） |
| `hybrid_search/store.py` | 侧车 `.npy`；npz 缺失时回退侧车；mmap 锁文件时降级 npz |
| `hybrid_search/fine.py` | ResNet 权重加载 `progress=False`（避免 GUI 无控制台句柄时 tqdm 崩） |
| `hybrid_search/io_utils.py` | `set_png_decoder`（cv2/libdeflate/imagecodecs/pillow）、cv2 RGB 直出、域缩放 |

### 调试开关（编进 DLL）
- `PF_SIMD=0|1|2` → 强制标量 / SSE2 / AVX2
- `PF_NOCACHE=1` → 完全不用 TLS 行缓冲（堆损坏时用它绕开）
- `PF_ROWNS=1` → 启用 TLS 行缓冲缓存（默认关闭）

---

## 四、性能数据总表（引用时注明口径）

| 项目 | 数值 | 口径 |
| :--- | ---: | :--- |
| libdeflate / Python zlib（纯 inflate） | 3.10×（1526 vs 492 MB/s） | 真实 IDAT、每变体整轮遍历 |
| 旁路解码 vs cv2 | 1.14×（像素加权 1.15×） | 分层 PNG 样本、主队列平均 |
| 瓦片建库 A/B | −3.8% wall / −3.8% CPU | 600 张固定样本（540+60 重复） |
| 整图建库 A/B | 墙钟 +0.27%、**CPU −7.55%** | 全库（约 3.8 万张真实图） |
| 旁路全库（基准台） | 492.83 s / 76.5 张/s / CPU 5,811.8 核秒 / 11.79 核 / GPU 均值 10.5% | 同口径 cv2 参照 478.13 s / 78.8 张/s / 6,469.3 核秒 |
| 缓存命中 | 90 → **4 ms/张**（20×） | `verify_prep_cache.py` |
| 缓存编码对照 | 读 3.02×、写 1.01×、体积 +8% | 256×256 载荷、100 张 |
| PNG 长尾 | p50 67.5 / p99 1046.7 / max 1800 ms；p99/p50 15.5；最慢 5% 占 28.9% CPU | 540 张 / 18 路 |
| JPEG 长尾 | p50 40.8 / p99 248.5 / max 307.5 ms；p99/p50 **6.08** | 540 张 / 18 路 |
| 建库耗时分解 | 解码 77.94 核秒（88.9% CPU、7.52 核）；前向 1.81 核秒 | 整图 600 张 |
| cv2_threads 0/1/2 | 首测 13.11/12.65/11.57 s（**噪声**）；配对复跑 =0: 11.23/12.18/12.90，=2: 12.04/12.09/12.38 → ≤1% | 瓦片 |

---

## 五、待办清单与用户已做决定

> **2026-09-28 补记（第二次交接）**：C2 已完成并定档"不采纳"（见下表）；
> 另修正两处上一轮的数字：**索引内 JPEG 是约 2.3 万张**（"2.5 万"是扫描到的文件数）、
> **渐进式 JPEG 占 3.7%**（旧记"样本 0%"是抽样偏差）；全库 EXIF 方向 **100% = 1**。
> 新增 6 个 devtools 脚本（`probe_jpeg_census` / `fetch_turbojpeg` / `bench_jpeg_decoders` /
> `verify_jpeg_decoder` / `ab_jpeg_turbo` / `ab_jpeg_mt`），按既有口径**不入库**。
> 另记录一个口径坑：`devtools/ab_build_bench.py` 的 `--png-decoder` 默认是 `cv2`，
> 而生产默认是 `libdeflate`，做配对 A/B 时要显式指定，否则结论会跑偏。

**用户已定**：不发 Release（本轮净负）→ 先回退+校验（**已完成，全绿**）→ 再按下面清单推进。

| # | 事项 | 状态/决定 |
| :--- | :--- | :--- |
| A1 | 归档本轮 PNG 结论（长尾、加权准入证伪、耗时分解、cv2_threads 证伪）进 `docs/perf-plan.md` 第五节 | **已完成**（第五节「2026-09-27 补充」+ 5 项证伪清单） |
| A2 | 同口径全库 A/B（基准台 `--png-decoder cv2`，4 万张 ≈8 分钟），挤掉 GUI vs 基准台的口径噪声 | 待做（用户此前未选） |
| A4 | `cv2_threads` 结论定档为"实验性、默认 0"（代码注释） | 待做（改动已回退，需在文档里说明） |
| C1+C3 | JPEG 零成本侦察 | **已完成**（p99/p50 6.08；渐进式实为 **3.7%**，旧记"样本 0%"是抽样偏差） |
| C2 | JPEG 解码器对照（cv2 vs PyTurboJPEG vs imagecodecs）+ 逐位校验 + 性能图 | **已完成 → ❌ 不采纳**（解码级 TurboJPEG 1.125× 且逐位一致，但建库级 +20~22.5% CPU 净负；详见 `docs/perf-plan.md` 第六节） |
| C4 | 更激进 DCT 缩放阈值（>1600 用 1/2） | **已完成 → ❌ 实测否决**（只省 15 核秒 = 0.26% 建库 CPU，却要重建全库；详见第三节末） |
| — | 缓存覆盖率（原第二优先） | **已完成 → ✅ 已 100%**（索引内全部命中、0 未命中；"修缓存键"无收益，关闭；详见第七节） |
| D1 | 旁路峰值内存 +31% 收敛（尺寸分流 / 分段 inflate 受限） | 未做 |
| D2 | 预处理约 5.6 核（torchvision/PIL Resize+Crop）→ 换 cv2 | 未做；代价 cosine 漂 0.9945 需重建 |
| D3 | 发布流程踩坑归档（代理/远端名/sslBackend/游离 HEAD 恢复） | 未做 |

---

## 六、环境与踩坑（省下次试错）

1. **代理**：本机 Clash 在 **`127.0.0.1:7897`**（7890 不在听）；系统 `ProxyEnable=0`，所以必须按命令传：`$env:HTTPS_PROXY='http://127.0.0.1:7897'`
2. **git 必须 `-c http.sslBackend=openssl`**：默认 schannel 走代理会 `failed to receive handshake`（gh 用 Go TLS 则正常）
3. **远端名是 `ImageSearchTool`**；push 用带凭据 URL：`https://x-access-token:$(gh auth token)@github.com/zccored/ImageSearchTool.git`
4. **rebase 冲突后恢复**：`.git/rebase-merge` 残留会让新 rebase 拒绝启动；`reset --hard` 报"Could not reset index file" 时先 `git checkout -f main`（HEAD 处于游离态），提交对象其实都在
5. **PowerShell 陷阱**：`.NET` API 用进程工作目录（不是 `cd` 的目录）→ 一律绝对路径；here-string/转义**两次污染源码**（`cli.py`、`ab_build_bench.py`）→ **改代码用 edit 工具按行改，别用字符串替换脚本**
6. **`python -E` 必须**（用户级 `PYTHONPATH` 曾指向一个第三方 python 目录，导致 `_ctypes` 崩溃；已删除，备份留在用户目录 `.dsh/` 下）；**numpy 必须 1.26.x**（torch 2.4.0+cu124）
7. **本机无 MSVC/cmake/ninja**；有 MinGW gcc 13.1、Cython 3.2.4、cffi
8. **报告/日志**：`perf_reports/`（索引见该目录 README.md）；`docs/perf-plan.md` 为性能归档正文

---

## 七、给新对话的接手指令

1. 先读 `00_INDEX.md` 与 `02_SESSION_LOG.md` 建立背景，再动手
2. **先做 C2（JPEG 解码器对照）**：`cv2` vs `PyTurboJPEG` vs `imagecodecs`，配对 A/B + 逐位校验 + 性能图；依据 C1/C3 结论（JPEG 单张便宜、无 progressive），**预期上限只有 5~15%**
3. 若 C2 收益不足，按数据优先做 **缓存覆盖率**（命中即 20×）与 **C4 阈值评估（先只给"解码耗时随缩放档变化"的数，不动语义）**
4. 随后做 **A1 归档**；**不要**重新尝试加权准入、cv2_threads、交叉引用、GDeflate、大图 RGB 旁路（均已被证伪并记录在案）
5. 一切**未提交**：提交/发版前必须先问用户；`branch/` 目录是否入库也需用户确认
