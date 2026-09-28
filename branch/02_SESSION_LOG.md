# 02 · 会话时间线 + 性能数据出处

> **用途**：追溯"每个结论是怎么来的、当时用什么口径、报告文件在哪"。
> 阶段划分按 106 轮的实际推进顺序；末节是**数据出处清单**（合并原计划中的 `03_PERF_DATA.md`）。

## 阶段 1 · PNG/DEFLATE 方案逐组评测
- 预分类：PNG **约 1.5 万张 / 约 58 GB**（RGBA 8bit 78.7%）；JPEG **约 2.5 万张 / 约 75 GB**（比 PNG 字节量更大）
- 组一 libdeflate：统一口径实测 **3.10×**；发现两个测量陷阱（MB/s 在 3 轮时被低估 3×；"输出大小未知"口径不可比）→ 全部重跑修正
- 组二 GDeflate：**外来 IDAT 0/48 可解**，输出 CPU 也读不了 → 出局
- 关键鉴别：`imagecodecs.deflate_decode` 并非 libdeflate 快路径；`_spng.pyd` 未链接 libdeflate

## 阶段 2 · libdeflate 落地（自建解码路径）
- 探针发现反滤波与 inflate 同贵 → 写原生内核（SSE2/SSSE3/AVX2）
- 验证体系：合成 264 用例（SIMD vs 标量交叉）、真实样本逐位 0 位差、格式矩阵、索引语义一致
- 结论：解码 **1.14×**，与早先 4.18× "上限"的差距归因于样本难度与"整幅缓冲"设计

## 阶段 3 · 修 bug 与回归（本阶段最重要）
- **多线程堆损坏**（`0xC0000374`/`0xC0000005`，进程静默消失）：二分 SIMD 级别均崩 → `PF_NOCACHE=1` 恢复 → 根因 **MinGW 静态 TLS 编进动态加载 .pyd**；默认改逐张分配
- `fast_load=True` 引入的两个回归：基准台读 npz 而索引写侧车（钉死基准台 `fast_load=False`）；同进程 mmap 导致 `os.replace` WinError 5（store 降级 npz）
- 发布后用户报错两例：`gallery.coarse.npz 不存在`（**我的回归**，store 已加侧车回退）；`加载 ResNet 失败 WinError 1`（torchvision tqdm 写无效句柄 → `progress=False`）

## 阶段 4 · 内存与缓存优化
- 内存特征探针：峰值出现在 93~95% 进度 = 单调爬升（18 线程 TLS 缓冲不释放）→ 全局 scratch 预算 256MB → 峰值 +25%→+16%、均值 −14%
- 缓存编码 PNG → libdeflate-6：读 3.02×、写 1.01×、体积 +8%、冷热逐位一致、命中 20×
- 两个 UI 选项收敛为默认（`fast_load`、`silence_png_warnings`），撤掉勾选框

## 阶段 5 · 性能调优尝试与证伪（本轮的教训）
- 用户提供了一份 cv2 优化清单 → 逐条对照：多数**已实现或不可行**（`IMREAD_COLOR_RGB` 早已做、`imdecodeBatch` 无 Python API、OpenCV 本就用 libjpeg-turbo、异步流水线早已有）
- `cv2_threads` 三档 A/B 首测 −11.7% → **配对复跑证伪**（≤1%）：教训是"波动 15% 与效应同量级时，单次 A/B 无意义"
- 长尾探针：PNG **p99/p50=15.5**、最慢 5% 占 **28.9%** 解码 CPU
- 加权准入仿真：**B=108 反而 +59% 墙钟** → 方案否决
- 耗时分解：解码 **88.9% CPU（7.52 核）**、主线程前向仅 1.81 核秒 → "消失的核"归因完成
- 大图 RGB 旁路：全库 CPU −10.2% 但**墙钟 +3.1%** → 净负 → **已回退**

## 阶段 6 · JPEG 侦察（会话末尾）
- C1 长尾：JPEG p50 **40.8 ms**、p99/p50 **6.08**（比 PNG 温和）
- C3 普查：样本 **95% baseline、0% progressive** → "渐进式慢 1.5~2×"不成立
- 判断：JPEG 换解码器上限 5~15%；**真正划算的是缓存覆盖与 DCT 缩放档评估**

---

## 数据出处清单（口径与报告位置）

| 数据 | 脚本 | 报告/日志 |
| :--- | :--- | :--- |
| 熵解码层对照（3.10× 等） | `devtools/bench_deflate_final.py`、`bench_deflate_group.py` | `perf_reports/deflate_final_*.html/json`、`deflate_libdeflate_*.html/json` |
| GDeflate 结论 | `devtools/bench_gdeflate.py`、`probe_gdeflate.py` | `perf_reports/deflate_gdeflate_*.html/json` |
| 缓存编码对照 | `devtools/probe_cache_codec.py` | `perf_reports/` 输出 + 控制台 |
| 缓存冷热一致性 | `devtools/verify_prep_cache.py` | 控制台（90→4 ms/张） |
| 旁路逐位一致性 | `devtools/verify_png_fast.py`、`test_png_filters.py` | `perf_reports/verify_png_fast_*.json` |
| 解码级性能图 | `devtools/bench_png_fast.py` | `perf_reports/png_fast_*.html/json` |
| 建库 A/B | `devtools/ab_build_bench.py` | `<基准工作目录>\results\*.json` |
| 内存特征 | `devtools/probe_memory_profile.py` | `perf_reports/mem_profile_*.json` |
| PNG 长尾 | `devtools/probe_decode_tail.py` | `perf_reports/decode_tail_*.json` |
| JPEG 长尾 | 同上（第三个参数 `jpg`） | `perf_reports/jpeg_tail.log` |
| 加权准入仿真 | `devtools/sim_weighted_admission.py` | `perf_reports/sim_admission_*.json` |
| 耗时分解 | `devtools/probe_pipeline_attrib.py` | `perf_reports/pipeline_attrib.log` |
| 全库旁路跑 | `ab_build_bench.py --n 40000 --dup 0` | `perf_reports/full_gallery_ldf.log` |

**口径警示**：探针输出的是**整进程墙钟**（含模型加载），run-to-run 波动可达 15%；速度结论只认
`ab_build_bench` 的配对 A/B。另有三个**已作废**口径（3 轮 MB/s 低估、Python 重压流、按 basename 做键），
详见 `perf_reports/README.md` 与 `docs/perf-plan.md`「口径修正记录」。
