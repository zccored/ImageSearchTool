# OpenVINO iGPU 后端（实验，默认 auto）

## 结果（2026-11-28，NAS i5-1235U Iris Xe 80EU）

| 项目 | 结果 |
|---|---|
| Intel 驱动 | 32.0.101.7088，`GPU` 设备可用 |
| OpenVINO | 2026.4.1 |
| ResNet18 IR | FP16，`model/resnet18.xml`，输出 512 维 |
| 纯前向探针 | batch=8：**465.5 张/s** |
| 主程序 128 张冷构建 | **127/128** 成功，特征 **92.4 张/s**，耗时 6.0s |
| 原 torch CPU | 约 20 张/s |

## 启用

GUI 构建页：

- 推理后端：`auto`（推荐）
- OpenVINO 设备：`auto`（NAS 自动 `GPU`；无 GPU 回退 `CPU`）

CLI：

```powershell
ImageSearchCLI.exe build <gallery> --prefix <prefix> `
  --backend auto --openvino-device auto `
  --batch 64 --decode-workers 12 --prep cpu
```

`auto` 规则：

1. 有 CUDA（本机 RTX 4060）→ torch CUDA；
2. 否则 OpenVINO 可用（NAS Iris Xe）→ OpenVINO GPU；
3. 否则 torch CPU。

所以本机不会误选 UHD 核显；NAS 自动走 iGPU。

## 精度

- OpenVINO FP16 与 torch FP32 特征余弦：**0.99998–0.99999**；
- 建库与查询统一走 `stream_decode`；
- 单图查询特征与索引特征约 **0.99+**（GPU batch 形状造成的微小差异）。

## 当前边界

- 前端预处理/解码仍是 CPU，128 张冷构建 6.0s 中真正 ResNet 前向占比很小，
  当前端到端瓶颈已转到解码/粗筛，而不是 GPU；
- OpenVINO 特征检查点语义抽样仍可能拒绝复用（安全回退重算）；
- INT8 / ONNX Runtime 量化尚未接入；当前是 FP16。

## 打包

`image-search.spec` 使用 `collect_all("openvino")` + `model/` 数据；NAS 包为
`ImageSearch`（v4），包含 OpenVINO 运行库、插件和 IR。
