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

# 本次更新（v3.5.0-beta）

- **CPU/GPU 双路同源**：CPU 与 GPU 共用同一套 torch resize/crop 算子（特征余弦 0.9999997）。
- **批量 GPU 预处理**：解码线程只交 RGB，GPU 单线程批量 resize/crop；单图预处理 17.9 → 3.69 ms。
- **预分配批次**：`forward_rgb` 预分配 `[n,3,224,224]` 替代 stack（52 ms → 6.4 ms）。
- **实测**：10k −14.3% 墙钟；全量 37,682 张 408.96 → 334.13 s（−18.3%）。
- **语义保护**：`prep` 进 meta/fatal；`add`/`add_tiles` 自动对齐；瓦片查询白化与混合守卫修复。
- **界面/CLI**：新增 `prep` / `prep_batch`；`--prep` / `--prep-batch`。
- **GUI 默认值**：预处理默认 `torch`、批大小默认 `64`（3000 张实测 43.6 → 33.5 s）；
  可视化面板隐藏时不再产生逐帧事件（`viz` 开关）。


> 完整说明见仓库 `docs/paper/dual_path_10k.md`。