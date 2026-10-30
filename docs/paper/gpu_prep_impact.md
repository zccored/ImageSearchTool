# D4 影响评估：把 Resize/Crop 搬到 GPU（`_make_fused_prep` + `_forward`）

> 结论先行：**可行、收益约 10–15%，但会改变精排特征（余弦 0.9977），必须带索引语义标记与回归验证；不建议直接全量改。**

## 1. 现状与调用链

```
_make_fused_prep.prep(path)            # 线程池里执行
  └─ read_bytes → decode_rgb            (CPU，69%)
     → transform: Resize(256, bilinear) + CenterCrop(224) + ToTensor   (CPU，23%)
     → 返回 (tensor[3,224,224] float 0..1, payload)
engine._fused_ingest.on_batch(...)
  └─ ex._forward(tensors)               # stack → .to(device) → 归一化 → ResNet
```

- `_forward` 已支持 **uint8 张量 + GPU LUT 归一化**（`_tile_scale_lut`），说明"归一化上 GPU"这条腿已经存在。
- 目前 CPU 侧真正贵的是 **Resize/Crop（8.2 ms/张，p90 39.7 ms）**。

## 2. 实测（`docs/paper/gpu_prep_probe.py`，60 张）

| 项 | 结果 |
| :--- | ---: |
| CPU Resize+Crop | 中位 **8.18 ms**（p90 39.69） |
| GPU Resize+Crop（含 H2D） | 中位 **1.53 ms**（p90 9.96） |
| 省时 | **6.65 ms/张（81%）** |
| 精排特征余弦 | min **0.9908**，mean **0.9977** |
| 显存峰值（单张） | 641 MiB |

特征不是逐位一致：PIL 双线性 vs `F.interpolate` 双线性的采样差异，导致约 0.3% 的余弦偏移。
**这足以让"旧索引 + 新查询"跨空间**，必须先解决索引语义标记，否则不能上线。

## 3. 显存与带宽（回答"要不要预算"）

| 方案 | 每批上传量 | 结论 |
| :--- | ---: | :--- |
| 现状（只传 224×224 张量） | **147 MB / 256 张** | 不需要预算 |
| 全幅 RGB、batch=256 | 中位 1.67 GB，p99 最坏 **19.7 GB** | 必然 OOM，**不能用** |
| 全幅 RGB、固定 chunk=8 | 中位 211 MB，p99 2.5 GB，极端 3.3 GB | 8 GB 卡可容纳，**不需要预算** |

即：**不做 GPU 预处理 → 不需要预算；做了 → 用固定 chunk（8）而不是动态预算**，这与"设预算没必要"的判断一致。

带宽：10,000 张 × 6.5 MB ≈ 65 GB；PCIe 4.0 ×8 实测约 5–8 GB/s → 8–13 s，
而 CPU 侧可省 66 s → 净赚；37,683 张 ≈ 240 GB → 30–48 s，省 245 s，仍净赚。

## 4. 对接口的影响（改 `_make_fused_prep` + `_forward`）

| 影响面 | 具体变化 | 风险 |
| :--- | :--- | :--- |
| `prep` 返回值 | 由 `tensor[3,224,224]` 变为 `uint8 HWC` 原图（或缩到 ≤2048 的 RGB） | 预处理缓存（`_PREP_CACHE`）体积变大：2.2 MP ≈ 6.5 MB/张，缓存命中率不变但内存上涨 |
| `_forward` 入参 | 需要接受 uint8 列表并按 chunk 做 resize/crop/normalize | 现有 `_tile_scale_lut` 可复用；fp16/autocast 路径要单独验证 |
| `frame_sink` | 帧回调目前拿 `_center_quad_sample(rgb)`；改后仍可在 prep 内取，不影响 | 低 |
| `add`/`add_tiles` | 增量提取同样走新路径；若索引是旧 transform 建的，新旧行会跨空间 | **必须加 meta 标记 + fatal 校验** |
| 瓦片路径 | `tile_index` 已是 uint8 + LUT；若统一改 GPU resize，需保证与整图一致 | 中 |
| 检索查询侧 | `extract_one` 必须与新 transform 一致，否则查询/库不同空间 | **高** |
| 确定性 | GPU 插值 + fp16 非确定性叠加，同图两次特征可能差 1e-4 | 可接受（与现状同级） |
| 现有索引 | 旧索引不能直接用于新代码；要么重建，要么 meta 标记后禁用 GPU 路径 | **高** |

## 5. 建议的落地顺序（如果要做）

1. 先加 **meta 字段** `prep: "cpu" | "gpu"`，进 `engine._check_cfg_compat` 的 fatal_keys（与 `binarize_norm` 同级）；
2. 做 **1,000 张原型**（flag 开关）：对比 CPU/GPU 两条路径的 `MR@300`、`R@10`、top-1 一致率；
3. 只有检索指标在噪声内，才考虑全量；否则保持 CPU 路径；
4. 若上线：固定 chunk=8（不引入动态预算），并在 `whiten` 侧车行数校验里同步检查 prep 标记。

## 6. 我的建议

- **D5 维持"不设预算"**：现状 147 MB/批，确实不需要；GPU 尖峰是前向太短（p50 1–20%），不是显存问题。
- **D3 暂缓**：收益 10–15% 但会改特征语义，需要 meta + 回归测试；在论文/检索指标未定稿前不值得引入。
- 若一定要做，先做第 5 节的 1,000 张原型，用 `MR@300` 是否变化来决定，而不是用墙钟决定。
