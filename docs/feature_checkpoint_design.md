# 断点特征检查点与完整特征校验（设计稿）

日期：2026-11-26  
目的：全量建库中断后，下一次启动直接复用前 N 张的 **512 维精排特征**，不再解码、不再 GPU 前向。  
前提：**先通过完整图库特征校验**；任何一项失败都必须放弃检查点，回退到 u16 张量热回放或全冷构建，绝不能混用。

## 1. 为什么不能直接复用现有 u16 缓存

当前 `prep_cache_torch_u16` 存的是 `[3,224,224]` 预处理张量 + 粗筛记录：

- 命中后仍要跑一次 ResNet 前向（37k 约 20–30 s）；
- 只能跳过读盘/解码/粗筛，不能跳过前向；
- 数据结构是“图像缓存”，不是“建库检查点”。

断点续建需要的是 **特征检查点**：

- 每条：路径 + 文件 stat + 512 维 fp32 特征 + 粗筛 `hu/fp/md5` + CRC；
- 37k 约 **75 MB**（2 KB/张），远小于 u16 缓存（10.6 GB）；
- 检查点通过校验后，前 N 张直接进入索引，不跑 GPU。

## 2. 存储布局

```
<索引目录>/feature_checkpoint/
  manifest.json          # 签名的提交清单（原子替换）
  entries.bin            # 追加式定长/变长记录
  validation.json        # 最近一次完整校验报告（可选）
```

### manifest.json

```json
{
  "version": 1,
  "build_id": "20261126-...",
  "signature": {
    "model": "resnet18",
    "prep": "torch",
    "resize": 256,
    "crop": 224,
    "feature_dim": 512,
    "feature_dtype": "float32",
    "coarse_size": 64,
    "blur": 5,
    "binarize_norm": "none"
  },
  "n_committed": 5073,
  "path_prefix_sha256": "...",
  "entries_sha256": "...",
  "created_at": "...",
  "updated_at": "..."
}
```

- `n_committed` 是**唯一可信前缀长度**；`entries.bin` 中超出部分视为未提交尾部，直接忽略。
- `entries_sha256` 覆盖 `entries.bin` 前 `n_committed` 条记录，防止截断/篡改。
- manifest 用临时文件 + `os.replace` 原子提交。

### 每条记录

| 字段 | 说明 |
|---|---|
| `path_key` | `sha1(normcase(abs(path)) + size + mtime_ns)`，20 B |
| `path` | 原始路径（UTF-8，便于诊断） |
| `stat` | size + mtime_ns |
| `feature` | 512 × float32 |
| `hu` | 7 × float32 |
| `fp` | 512 B |
| `md5` | 32 B hex |
| `crc32` | 整条记录（不含 crc 本身） |

## 3. 完整校验门槛（全部通过才允许 resume）

### A. 结构完整性（全量，不只抽样）

1. manifest 存在、`version` 匹配、`signature` 与当前 `Config`/模型/预处理完全一致；
2. `n_committed` 在 `[0, 当前路径数]` 范围内，且 records 数量 ≥ n；
3. 前 `n_committed` 条记录逐条 CRC 通过；
4. `entries_sha256` 与重算值一致；
5. 路径顺序与当前 `paths[:n_committed]` 逐项一致；
6. 无重复路径、无重复 md5（或明确允许同内容多路径）；
7. 每条 feature：维度 512、float32、无 NaN/Inf、L2 范数 ∈ [0.9998, 1.0002]；
8. 每条 hu/fp/md5 形状与长度正确；
9. 文件 stat 键一致；文件已变更则整条不可信。

### B. 语义一致性（跨空间校验）

结构校验只能证明“数据没坏”，不能证明“特征还是同一个模型空间”。因此还要：

1. **签名校验**：model / prep / resize / crop / feature_dim / coarse 参数任一变化 → 直接拒绝；
2. **抽样重算**：从检查点随机抽 K 条（建议 64），从 u16 张量缓存取对应 224 张量，走当前 `forward_gpu` 重算特征；
   - 与检查点特征的余弦 **min ≥ 0.9999**；
   - 分数差、范数、NaN 检查；
3. **严格模式（可选）**：对全部检查点特征重算一遍（37k × 0.53 ms ≈ 20–30 s），余弦/误差全部通过才允许 resume；
   - 这仍比解码全库（约 340 s）便宜；
   - 如果用户要求“绝对不混空间”，用严格模式。

### C. 混合空间安全

resume 后的最终索引由两部分组成：

- 检查点特征（前 N 张，旧构建产生）；
- 新计算特征（N 之后，当前进程产生）。

必须保证两者同一模型空间：

- 签名一致 + K 条抽样重算通过；
- 新计算部分使用同一个 `ResNetExtractor`、同一 `prep`、同一 `norm_on_gpu`；
- 保存前再对最终 `fine` 做一次全量结构校验（维度/范数/NaN/路径对齐）。

## 4. Resume 流程（只在显式开启时）

1. 读取 manifest，跑 A+B；
2. 校验失败：记录原因，删除/忽略检查点，回退到现有 u16 热回放或全冷构建；
3. 校验通过：
   - 前 N 张：直接使用检查点 feature + 粗筛 rec，跳过 decode/coarse/forward；
   - 第 N 张之后：正常冷处理；
   - 每完成一批，追加 entries + 原子提交 manifest；
4. 全部完成后，保存正式索引，再跑一次完整 feature 校验；
5. 校验失败：不覆盖旧索引，保留现场。

## 5. 现状与基线

现有全量索引 `F:\视频\.gallery_index\gallery` 已通过校验：

```json
{
  "ok": true,
  "n": 37782,
  "feature_dim": 512,
  "norm_min": 0.9999998807907104,
  "norm_max": 1.0000001192092896,
  "bad_chunks": 0,
  "zero_rows": 0
}
```

校验器：`devtools/validate_feature_index.py`  
报告：`perf_reports/feature_validation_full.json`

## 6. 实现顺序

1. `feature_checkpoint.py`：manifest + entries 读写、原子提交、CRC/hash；
2. `validate_feature_checkpoint.py`：A+B 校验，输出 JSON；
3. `build` 增加显式 `resume=True` 分支：只在前 N 张跳过；
4. GUI 增加“继续上次构建”按钮，构建前后显示检查点覆盖率/校验结果；
5. 全量测试：中断 → 校验 → resume → 对比无中断构建的 fine 矩阵逐路径余弦。

## 7. 多盘性能天平（占位，未实现）

未来完整图库可能跨多块硬盘，且每块盘的随机读/顺序读/延迟不同。当前检查点的 Entry 已保留
`disk_hint` 字段（现在写空字符串），足以在不改格式的前提下记录来源盘。

后续实现时建议：

- 每个图库根配置 `disk_id / read_bps / latency_ms`；
- 调度器按盘加权分配解码 worker，避免慢盘队头阻塞；
- 检查点记录 `disk_hint`，resume 时按盘恢复未完成队列；
- 性能天平目标：各盘有效吞吐 × 权重接近，而不是简单平均分配线程。

本版本只做单盘路径，不引入调度逻辑。

## 8. 当前实现与实测（2026-11-26）

已实现：

- `hybrid_search/feature_checkpoint.py`：path-keyed 记录、CRC32、manifest 原子提交、签名校验、结构校验、64 条语义抽样；
- `engine.build/add` 自动写入/复用；GUI「断点特征检查点」开关（默认开）；
- `devtools/validate_feature_index.py`：完整索引结构校验；
- `devtools/validate_feature_checkpoint.py`：检查点结构校验；
- `devtools/seed_feature_checkpoint.py`：从现有完整索引迁移，无需重算。

实测（F: 生产索引 37,782 条）：

| 项 | 结果 |
|---|---:|
| 迁移检查点 | 37,782 条 / 2.7 s / 104.4 MB |
| 全量热构建 | 6.31 s（对照冷构建 337.64 s） |
| 语义抽样 | 64 条，cos_min 0.9999997616 |
| 结构校验 | ok，0 条变更/缺失 |

尚未实现：

- `strict-audit`：对全部 N 条用 u16 张量缓存重算比对；
- 多盘性能天平：仅 `disk_hint` 字段 + 本文第 7 节占位。

报告：`perf_reports/feature_checkpoint_full.md`
