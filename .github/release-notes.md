# ImageSearchTool · Windows x64 (CUDA) 打包版

> 本包由 **GitHub Actions 自动构建**（`build-release-win64-cuda.yml`），
> 从对应的 tag 源码打包，非本地手工产物。

## 下载与安装

1. 下载本页的**全部**分卷（`...zip.001` / `...zip.002`）；
2. 合并成一个 zip：
   ```bat
   copy /b ImageSearchTool-<tag>-win64-cuda.zip.001 + ImageSearchTool-<tag>-win64-cuda.zip.002 ImageSearchTool-<tag>-win64-cuda.zip
   ```
3. 解压得到 `ImageSearch/`，双击 `ImageSearchGUI.exe`。

- 目标机**无需安装 Python / 依赖**（运行时已内置）
- 三个入口：`ImageSearchGUI.exe`（图形界面）、`ImageSearchCLI.exe`（命令行）、
  `ImageSearchWeb.exe`（Web 界面，需系统 WebView2 运行时）
- 各分卷 SHA256 见本页附件 `SHA256SUMS.txt`

---

# 本次更新

## 瓦片（局部）检索召回修复 —— 不再漏图

旧路径靠 **LSH 桶取候选 + 指纹硬过滤**，桶外或指纹不同的正确瓦片会被直接丢掉。
现在改为 **全库瓦片 × 全部查询块的有界分块余弦评分**，按原图取最大分 ——
这是**在已有特征下的精确排名**，不存在桶截断带来的漏召回。

**开关语义跟着变了（重要）**：

| 开关 | 现在的行为 |
| :--- | :--- |
| `--cand lsh` / `--cand coarse` | **仅保留兼容**，实际也走全覆盖评分 |
| `--lsh-*` 系列 | **不再改变召回结果** |
| `coarse_k` | 仍影响**整图**检索；不再限制瓦片候选的原图数量 |
| `top_k` | 瓦片输出条数由它决定 |

**性能口径随之变化**（靶子库 455,175 块）：

| 路径 | 首查（含模型加载） | 常驻热查 |
| :--- | ---: | ---: |
| 全覆盖评分（当前） | 约 **4.10 s** | 约 **0.80 s / 次** |
| 旧 LSH（**已废弃，漏图**） | 2.2 s | ~1 s |

> ⚠ 旧 LSH 的历史延迟**不是等正确性对照**，别拿来比较。

- **旧索引可直接使用，无需重建**
- 正在运行的程序需**重启**以载入修复

## 性能优化（本机实测，逐项开关与数据见 `docs/perf-plan.md`）

新增 `docs/perf-plan.md` —— 性能优化的**唯一事实来源**：每条都带本机实测数字、开关名与
验证方式，并且**保留被证伪的假设**（附反证数据），避免以后重复踩。

已落地的主要项：

| 项 | 开关 | 效果 |
| :--- | :--- | ---: |
| 块 md5 复用 | `tile_md5_reuse=True` | **−37% wall / −24% CPU** |
| 重复内容预过滤（瓦片） | `dedup_prefilter=True` | −10.9% wall |
| 重复内容预过滤（整图） | 同上 | −4.9% wall |
| 归一化搬 GPU | `norm_on_gpu=True` | −7.7% wall |
| 批大小 64 → 256 | `batch`（自动） | **−24% wall**，吞吐 65.6 → 86.3 张/s |
| cv2 直出 RGB | `cv2_rgb_direct=True` | −3.0% wall |
| 瓦片 tick 20 → 120 ms | `tile_flush_ms` | **无收益（噪声内）→ 保持默认 20 ms** |

新增 OpenCV 内部线程数配置 `opencv_threads`（进程级，首次任务前生效）。

## 交接协议 v2：整图 / 子图 可选或都做

与「全栈图库管理器」的跨进程交接升级到 **schema v2**：

- request 新增 `modes`：`["full","tiles"]`（缺省/老请求，两个都做）/ `["full"]` / `["tiles"]`
- 本侧按 `full → tiles` 固定顺序连做，逐阶段上报进度，result 里给出
  `modes` / `stages[]` / `total_added` / `total_tiles_added`
- CLI 可覆盖：`python main.py ingest <request.json> --modes full,tiles`
- 两边都没勾 → 由 img_server 侧暂存 `handoff/pending_<时间戳>.json`（本程序不读不写）
- 协议全文：`docs/HANDOFF_PROTOCOL.md`

## 新增 `devtools/` 基准与回归脚本（61 个）

A/B 建库基准（`ab_build_bench.py`）、各类 `probe_*` / `bench_*`，
以及 `verify_*` 自检（含 `verify_handoff_modes.py`、`verify_tile_index.py`）。

---

# 上次更新（v3.3.0-beta）

> 上一版发布说明的要点，方便对照。

- **Web 版界面并入主线**：新增 `gui_web.py` / `ImageSearchWeb.exe`（基于 pywebview）
- **路径治理**：统一索引/缓存/输出路径处理，修掉一批相对路径踩坑
- **CLI 订正**：参数与默认值对齐，`--help` 文案重写
- **README 重写**：结构重整，补打包版下载指引
- **两卷分发**：从 v3.3.0-beta 起提供可直接双击运行的 Windows 打包版
- 全包敏感信息扫描：无本机路径 / 用户名 / 令牌

---

## 许可证

**AGPL-3.0-only** —— 见 [LICENSE](https://github.com/zccored/ImageSearchTool/blob/main/LICENSE)。
