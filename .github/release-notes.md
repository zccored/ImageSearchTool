# ImageSearchTool · Windows x64 (CUDA) 打包版

> 本包由 **GitHub Actions 自动构建**（`build-release-win64-cuda.yml`），
> 从对应的 tag 源码打包，非本地手工产物。
>
> **分支说明**：本版本来自 `refactor/v3.5.1-beta-zccored-2026.11.22`，
> 属于缓存池/断点检查点重构的**实验性预发行**，不建议直接替换稳定版。

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

# 本次更新（v3.5.1-beta）

## 断点特征检查点（本版核心）

- 全量建库中断后，下一次启动可按路径复用已完成的 **512 维精排特征**，
  不再解码、不再跑 ResNet 前向；
- 检查点按路径键存储：512 维 fp32 特征 + 粗筛 `hu/fp/md5`，约 **2 KB/张**；
- manifest 原子提交、逐条 CRC32、entries SHA-256、路径/stat 校验；
- 语义抽样：64 条经 u16 张量缓存重算，余弦 min **0.99999976**；
- 任一校验失败自动拒绝检查点，回退到 u16 热回放/冷构建，不混空间；
- 从现有 37,782 条完整索引迁移检查点：**2.7 s / 104.4 MB**；
- 全量热构建实测：**6.31 s**（对照冷构建 337.64 s，约 53×）。

> 严格说明：本版做的是“全量结构校验 + 64 条语义抽样”，
> 尚未实现逐条重算全部 37,782 条的 strict-audit。

## u16 预处理缓存

- 新增 `prep_cache_dtype=u16`：与 fp16 同体积（约 301 KB/张），
  但把近邻打分误差从 fp16 的 **1.41e-3** 降到 **3.58e-7**；
- 1,500 条真实张量：fp16 near-tie 翻转 4 次，u16 为 **0 次**；
- 1,000 张真实图：u16 热缓存 **1.10 s（907 张/s）**，
  与无缓存参考 top1 20/20、top10 200/200；
- 缓存目录与 fp16/fp32 物理隔离：`prep_cache_torch_u16`；
- GUI 新增「预处理缓存精度：u16 / fp32 / fp16」，默认 u16。

## GUI / 配置

- 新增「断点特征检查点」开关（默认开启）；
- 缓存精度选项与检查点开关均通过 GUI 参数读取；
- 现有完整索引可直接用 `devtools/seed_feature_checkpoint.py` 迁移检查点；
- 校验工具：
  - `devtools/validate_feature_index.py`
  - `devtools/validate_feature_checkpoint.py`

## 多盘性能天平（占位，未实现）

- 检查点 Entry 预留 `disk_hint` 字段；
- 设计稿见 `docs/feature_checkpoint_design.md` 第 7 节；
- 当前版本仍为单盘调度，不做按盘权重分配。

## 已知限制

- 检查点签名绑定 `model / prep / resize / crop / coarse 参数 / feature_dim`；
  改模型或预处理会整体失效并回退冷构建；
- 图片 size/mtime 变化、路径删除会让对应条目失效；
- 增量入库仍只处理新图，检查点只对“中断的全量构建/中断的增量构建”有意义；
- 瓦片索引不使用整图特征检查点；
- 本版是实验分支，稳定生产请继续使用 `main` / `v3.5.0-beta`。

## 验证

- 完整索引结构校验：37,782 条，512 维，范数 ∈ [0.99999988, 1.00000012]，
  路径/特征顺序一致，无 NaN/Inf、无零行；
- 检查点结构校验：37,782 条，CRC/SHA/stat 全部通过；
- 全量热构建：6.31 s，临时前缀验证，不修改生产 `gallery.*`；
- 生产 `gallery.*` 索引文件未改动。

> 详细数据见仓库 `perf_reports/feature_checkpoint_full.md`。
