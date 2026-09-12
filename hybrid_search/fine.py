# -*- coding: utf-8 -*-
"""
第二阶段：ResNet 语义精排器。

- 支持 resnet18 / 34 / 50 / 101 / 152（torchvision 预训练权重）。
- 去掉分类头（fc -> Identity），取全局池化后的语义特征：
    resnet18/34 -> 512 维；resnet50/101/152 -> 2048 维。
- 特征做 L2 归一化：之后“余弦相似度”等价于向量点积，一次矩阵乘法完成打分。
- GPU 用 torch.autocast 走 FP16（权重保持 fp32），CPU 全 fp32。

吞吐设计（榨干 CPU + GPU）：
  * “解码/预处理线程池（CPU） + 双缓冲队列 + 主线程前向” 流水线：
    解码与推理重叠，GPU 侧不会干等磁盘/解码；多核 CPU 解码线性提速。
  * 解码走 OpenCV 快速路径（EXIF 检测后回退 Pillow），单次读盘。
  * 推理前向按批进行；CPU 场景维持单线程前向（实测多线程反降速）。
  * torch 推理线程数可配（torch_threads，0=保持 torch 默认）。

内存参考（float32 全库索引）：
  resnet18 512 维： 5000 张 ≈ 10 MB；200 万张 ≈ 4 GB
  resnet50 2048 维：5000 张 ≈ 41 MB；200 万张 ≈ 16 GB
16GB 内存机器跑 200 万级图库：选 resnet18，或 store_fine=False 按需实时抽特征。
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import Config
from .io_utils import LOGGER, decode_rgb, read_bytes

_PREP_QUEUE_DEPTH = 3        # 双缓冲再加深：队列里最多同时存几批已预处理张量


_CUDA_FALLBACK_WARNED = False


def _resolve_device(cfg: Config) -> str:
    """
    解析推理设备：
      auto    -> 有 CUDA 用 cuda，否则 cpu；
      cuda    -> 若当前 torch 无 CUDA（未装 CUDA 版/驱动不可用），
                 自动回退 CPU 并提示一次，而不是让检索直接失败；
      cpu     -> 强制 CPU。
    """
    global _CUDA_FALLBACK_WARNED
    if cfg.device == "cpu":
        return "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 —— 未装 torch 时按 CPU 处理，错误留给加载处
        cuda_ok = False
    if cfg.device == "auto":
        return "cuda" if cuda_ok else "cpu"
    if cfg.device == "cuda" and not cuda_ok:
        if not _CUDA_FALLBACK_WARNED:
            _CUDA_FALLBACK_WARNED = True
            LOGGER.warning(
                "设备指定为 cuda，但当前 torch 无法使用 CUDA"
                "（未安装 CUDA 版 torch，或驱动不可用），已自动回退 CPU 继续；"
                "安装 CUDA 版后即可用 GPU，例如：\n"
                "  pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu124")
        return "cpu"
    return cfg.device


def _auto_decode_workers(cfg: Config, cuda: bool) -> int:
    """
    解码线程数：显式值 > 0 用之；0=自动：
      GPU 场景解码是喂饱 GPU 的关键 -> 上限 20（解码/熵解码每张单核，
      20 核机器全核解码，供给最紧的 JPEG 熵解码与 PNG 解压）；
      CPU-only 场景解码与前向争抢核心 -> 上限 8。
    """
    if cfg.decode_workers > 0:
        return cfg.decode_workers
    cores = os.cpu_count() or 4
    return max(1, min(20, cores)) if cuda else max(1, min(8, cores))


def _load_weights(models, model_name: str):
    """
    取最新预训练权重枚举（ResNet18_Weights.DEFAULT 形式），
    避免传字符串触发的弃用警告。
    """
    cls_name = model_name[0].upper() + model_name[1:] + "_Weights"
    cls = getattr(models, cls_name, None)
    if cls is not None and hasattr(cls, "DEFAULT"):
        return cls.DEFAULT
    return models.get_model_weights(model_name)


_PRE_DOWNSCALE_PX = 8_000_000       # 像素数超过该值先降采样
_PRE_DOWNSCALE_SIDE = 2048          # 重型图降采样后的最长边
_VIZ_QUAD = 16                      # 可视化象限分辨率：16×16 RGB 采样块


def prep_cv2(rgb: np.ndarray, mean, std) -> "object":
    """numpy RGB -> ResNet 输入张量（cv2 实现，语义对齐 torchvision）。

    等价于 `Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize`：
      * 短边缩到 256（cv2.INTER_AREA 面积平均，缩小时比 PIL BILINEAR 更快更清晰）；
      * 居中裁 224×224（偏移用 round((len-224)/2)，与 torchvision 一致）；
      * 归一化后转 (3,224,224) float32 CHW。

    为什么默认不用：实测大图（≤2048 边长）只快 1.67×（8.9ms→5.3ms），
    占单张图片总 CPU 成本的 ~2%，但特征余弦相对 torchvision 路径漂到 0.9945，
    与既有索引不一致，得不偿失（见 devtools/bench_prep_big.py）。保留此实现
    以便将来在“全量重建”场景下按需启用。
    """
    import cv2
    import torch

    h, w = rgb.shape[:2]
    side = min(h, w)
    scale = 256.0 / max(side, 1)
    if abs(scale - 1.0) > 1e-6:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (max(1, int(round(w * scale))),
                               max(1, int(round(h * scale)))),
                         interpolation=interp)
    h, w = rgb.shape[:2]
    if h < 224 or w < 224:                     # 极端小图：补齐到 224（居中）
        pad_y = max(0, 224 - h)
        pad_x = max(0, 224 - w)
        rgb = cv2.copyMakeBorder(rgb, pad_y // 2, pad_y - pad_y // 2,
                                 pad_x // 2, pad_x - pad_x // 2,
                                 cv2.BORDER_REPLICATE)
        h, w = rgb.shape[:2]
    y0 = int(round((h - 224) / 2))
    x0 = int(round((w - 224) / 2))
    crop = np.ascontiguousarray(rgb[y0:y0 + 224, x0:x0 + 224])
    t = torch.from_numpy(crop).permute(2, 0, 1).float().div_(255.0)
    t = t.sub_(torch.as_tensor(mean).view(3, 1, 1)).div_(
        torch.as_tensor(std).view(3, 1, 1))
    return t.contiguous()


def _center_quad_sample(rgb: np.ndarray) -> np.ndarray:
    """取图片中心 224×224 窗口（与 ResNet 预处理一致的采样视野），
    降采样成 16×16×3 的 RGB 象限块 —— 供 GUI 可视化“ResNet 正在采样这张图”。"""
    import cv2
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    window = rgb[y0:y0 + side, x0:x0 + side]
    quad = cv2.resize(window, (_VIZ_QUAD, _VIZ_QUAD),
                      interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(quad)


_PREP_CACHE = None                  # 由 engine 注入（见 set_prep_cache）


def set_prep_cache(cache) -> None:
    """全局挂载预处理缓存：build-fine / 查询等解码路径也能命中（省一次解码）。"""
    global _PREP_CACHE
    _PREP_CACHE = cache


def _prepare_one(path: str, transform, frame_sink=None) -> Optional:
    """
    单个工作项：单次读盘 -> 快速解码 RGB -> torchvision 预处理 -> CPU 张量。
    frame_sink(path, quad16x16x3)：可选可视化回调（解码 worker 线程内调用，
    异常被吞，不影响前向）。
    """
    if _PREP_CACHE is not None and frame_sink is None:
        hit = _PREP_CACHE.get(path)      # 命中：跳过读盘+解码+预处理
        if hit is not None:
            return hit[0]
    data = read_bytes(path)
    if data is None:
        return None
    rgb = decode_rgb(data)
    if rgb is None:
        return None
    if frame_sink is not None:
        try:
            frame_sink(path, _center_quad_sample(rgb))
        except Exception:  # noqa: BLE001 —— 可视化失败不得影响任务
            pass
    # 重型图（如 1 亿像素长图）先在解码层降一次采样到 ≤2048 边长，
    # 避免后续对全尺寸像素做 Resize（省 10~100 倍内存与时间）。
    if rgb.shape[0] * rgb.shape[1] > _PRE_DOWNSCALE_PX:
        import cv2
        scale = _PRE_DOWNSCALE_SIDE / max(rgb.shape[:2])
        rgb = cv2.resize(rgb,
                         (max(1, int(rgb.shape[1] * scale)),
                          max(1, int(rgb.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    try:
        return transform(Image.fromarray(rgb))
    except Exception:  # noqa: BLE001 —— 个别格式怪异，交给调用方跳过
        return None


class ResNetExtractor:
    """ResNet 特征抽取器：并行解码流水线 + 批量前向 + 归一化。"""

    def __init__(self, cfg: Config):
        import torch.nn as nn
        import torchvision.models as models

        if not hasattr(models, "get_model"):
            raise RuntimeError("torchvision 版本过旧（需 >= 0.13 的 get_model API）")

        self.cfg = cfg
        self.device = _resolve_device(cfg)
        LOGGER.info("精排：设备=%s，模型=%s", self.device, cfg.model)

        # torch 推理线程数（0=保持 torch 默认；CPU 单线程前向通常最优）
        if cfg.torch_threads > 0:
            import torch
            torch.set_num_threads(cfg.torch_threads)

        import warnings
        with warnings.catch_warnings():
            # 屏蔽 torchvision 内部对 weights 枚举的过时提示噪音
            warnings.filterwarnings(
                "ignore", message="Arguments other than a weight enum.*",
                category=UserWarning)
            weights = _load_weights(models, cfg.model)
            model = models.get_model(cfg.model, weights=weights)
        # fc 的输入维 == 全局池化特征维；移除分类头后前向即返回特征
        self.feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.use_fp16 = bool(cfg.fp16 and self.device == "cuda")
        if self.device == "cuda":
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True      # 固定输入尺寸下自动挑最快卷积内核
        if self.use_fp16:
            LOGGER.info("精排：已启用 FP16（autocast）")
        self.batch = cfg.batch if cfg.batch > 0 else (64 if self.device == "cuda" else 16)
        self.decode_workers = _auto_decode_workers(cfg, self.device == "cuda")
        LOGGER.info("精排：解码流水线线程=%d，前向批大小=%d",
                    self.decode_workers, self.batch)

        import torchvision.transforms as transforms
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        # 兼容保留 torchvision 版本（语义等价，作为对照/兜底）
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def prep(self, rgb: np.ndarray):
        """numpy RGB -> ResNet 输入张量。

        保留 cv2 快路径备查：实测大图仅快 1.67×（8.9ms→5.3ms，占单张 CPU
        成本 ~2%），但特征余弦会漂到 0.9945（与既有索引不一致），故**不使用**；
        如需启用请配合全量重建，见 devtools/bench_prep_big.py。
        """
        return prep_cv2(rgb, self._mean, self._std)

    # ------------------------------------------------------------------
    def _forward(self, tensors) -> Optional[np.ndarray]:
        """一组 CPU 张量 -> 归一化前特征矩阵（单次前向，autocast 可选）。"""
        import torch
        try:
            with torch.no_grad():
                batch_t = torch.stack(tensors, dim=0).to(self.device)
                if self.use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feats = self.model(batch_t)
                else:
                    feats = self.model(batch_t)
                return feats.float().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            LOGGER.warning("精排批次前向失败（%d 张）已跳过: %r", len(tensors), e)
            return None

    def extract_one(self, path: str) -> Optional[np.ndarray]:
        """单张同步抽取（查询图常用）：不经流水线，延迟最低。"""
        t = _prepare_one(path, self.transform)
        if t is None:
            return None
        feats = self._forward([t])
        if feats is None:
            return None
        out = np.asarray(feats[0], dtype=np.float32)
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-8 else out

    # ------------------------------------------------------------------
    def stream_decode(self, paths: List[str], prep=None, progress=None,
                      on_batch=None):
        """
        与 extract_batch 相同的并行解码流水线，但把“每批前向”开放给调用方：
          prep(path)            -> (tensor|None, payload|None)  自定义解码器
                                  （默认：读盘+解码+torchvision 预处理）
          on_batch(ok, tensors, payloads)
                                -> (ok_sub, rows_np|None)       每批回调
                                   rows 已 L2 归一化，ok_sub 与其逐位对齐；
                                   rows=None 表示整批丢弃。
        返回 (all_ok_paths, feats|None)，二者逐位对齐。
        融合建库用它：单遍解码同时喂粗筛特征 + ResNet 前向。
        """
        import torch

        if prep is None:
            def prep(p):
                t = _prepare_one(p, self.transform)
                return t, None
        if on_batch is None:
            def on_batch(ok, tensors, _payloads):
                feats = self._forward(tensors)
                if feats is None:
                    return [], None
                normed = feats.astype(np.float32)
                norms = np.linalg.norm(normed, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                normed /= norms
                return ok, normed

        if not paths:
            return [], None
        t0 = time.time()
        n = len(paths)
        n_batches = (n + self.batch - 1) // self.batch
        prep_q: "queue.Queue" = queue.Queue(maxsize=_PREP_QUEUE_DEPTH)

        def producer():
            task_q: "queue.Queue" = queue.Queue()
            results: List = [None] * n
            stop = threading.Event()

            def worker():
                while not stop.is_set():
                    try:
                        idx, path = task_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        r = prep(path)
                        if not isinstance(r, tuple) or len(r) != 2:
                            r = (r, None)
                    except Exception as e:  # noqa: BLE001
                        LOGGER.debug("解码失败 %s: %r", path, e)
                        r = (None, None)
                    results[idx] = r
                    task_q.task_done()

            threads = [threading.Thread(target=worker, daemon=True)
                       for _ in range(self.decode_workers)]
            for t in threads:
                t.start()
            try:
                for start in range(0, n, self.batch):
                    chunk = paths[start:start + self.batch]
                    for i, p in enumerate(chunk):
                        task_q.put((start + i, p))
                    task_q.join()
                    ok, tensors, payloads = [], [], []
                    for i, p in enumerate(chunk):
                        t, pl = results[start + i]
                        results[start + i] = None
                        if t is not None:
                            ok.append(p)
                            tensors.append(t)
                            payloads.append(pl)
                    prep_q.put((start, len(chunk), ok, tensors, payloads))
                prep_q.put(None)
            except Exception as e:  # noqa: BLE001
                prep_q.put(("ERR", e))
            finally:
                stop.set()

        thr = threading.Thread(target=producer, daemon=True)
        thr.start()

        all_feats: List[np.ndarray] = []
        all_ok: List[str] = []
        got = 0
        processed = 0
        while got < n_batches:
            item = prep_q.get()
            got += 1
            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == "ERR":
                LOGGER.error("解码流水线异常: %r", item[1])
                raise RuntimeError(f"解码流水线失败: {item[1]!r}")
            start, chunk_n, ok, tensors, payloads = item
            processed += chunk_n
            if progress:
                try:
                    progress(processed, n)
                except Exception:  # noqa: BLE001 —— 进度回调异常不得影响任务
                    pass
            if not tensors:
                continue
            try:
                ok_sub, rows = on_batch(ok, tensors, payloads)
            except Exception as e:  # noqa: BLE001 —— 回调失败整批跳过
                LOGGER.warning("融合批次处理失败（%d 张）已跳过: %r", len(ok), e)
                ok_sub, rows = [], None
            if rows is not None and rows.shape[0] > 0:
                all_feats.append(np.asarray(rows, dtype=np.float32))
                all_ok.extend(ok_sub)
        thr.join(timeout=5)

        if not all_feats:
            LOGGER.warning("精排：%d 张全部失败", len(paths))
            return [], None
        out = np.concatenate(all_feats, axis=0).astype(np.float32)
        LOGGER.info("精排：%d/%d 张成功（%d 维），吞吐 %.1f 张/秒",
                    len(all_ok), len(paths), out.shape[1],
                    len(all_ok) / max(time.time() - t0, 1e-6))
        return all_ok, out

    # ------------------------------------------------------------------
    def extract_batch(self, paths, progress=None, frame_sink=None):
        """
        流水线式分批抽取特征：
          daemon 解码线程（自管理，不用 concurrent.futures，程序退出时不会
          被 atexit 强制 join 而产生 KeyboardInterrupt 噪音）持续把
          “读盘→解码→Resize/Crop/Normalize”产成 CPU 张量送入有界队列，
          主线程按批取出前向，两段并行推进，内存只缓存少量批次。

        progress(done, total)：每处理完一批回调一次（含解码失败被剔除的图，
        done 会一直推进到 total，供 UI 估算完成进度与 ETA）。
        frame_sink(path, quad)：可选可视化回调，随每张图的解码并行触发
        （见 _prepare_one；异常不影响任务）。

        返回 (ok_paths, feats)：ok_paths 与 feats 逐位对齐（保持输入相对顺序），
        每行已 L2 归一化；全部失败返回 ([], None)，个别失败只剔除该图。
        """
        import torch

        if not paths:
            return [], None
        t0 = time.time()
        n_batches = (len(paths) + self.batch - 1) // self.batch
        n = len(paths)
        prep_q: "queue.Queue" = queue.Queue(maxsize=_PREP_QUEUE_DEPTH)

        def producer():
            """后台线程：投递解码任务并逐批汇总，双缓冲保障 GPU 不空等。"""
            task_q: "queue.Queue" = queue.Queue()
            results: List = [None] * n
            stop = threading.Event()

            def worker():
                while not stop.is_set():
                    try:
                        idx, path = task_q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        results[idx] = _prepare_one(path, self.transform,
                                                    frame_sink)
                    except Exception as e:  # noqa: BLE001
                        LOGGER.debug("解码失败 %s: %r", path, e)
                        results[idx] = None
                    finally:
                        task_q.task_done()

            threads = [threading.Thread(target=worker, daemon=True)
                       for _ in range(self.decode_workers)]
            for t in threads:
                t.start()
            try:
                for start in range(0, n, self.batch):
                    chunk = paths[start:start + self.batch]
                    for i, p in enumerate(chunk):
                        task_q.put((start + i, p))
                    task_q.join()                      # 本批解码完成
                    ok, tensors = [], []
                    for i, p in enumerate(chunk):
                        t = results[start + i]
                        results[start + i] = None      # 及时释放引用
                        if t is not None:
                            ok.append(p)
                            tensors.append(t)
                    prep_q.put((start, len(chunk), ok, tensors if tensors else None))
                prep_q.put(None)
            except Exception as e:  # noqa: BLE001
                prep_q.put(("ERR", e))
            finally:
                stop.set()

        thr = threading.Thread(target=producer, daemon=True)
        thr.start()

        all_feats: List[np.ndarray] = []
        ok_paths: List[str] = []
        got = 0
        processed = 0
        while got < n_batches:
            item = prep_q.get()
            got += 1
            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == "ERR":
                LOGGER.error("解码流水线异常: %r", item[1])
                raise RuntimeError(f"解码流水线失败: {item[1]!r}")
            start, chunk_n, ok, tensors = item
            processed += chunk_n
            if progress:
                try:
                    progress(processed, n)
                except Exception:  # noqa: BLE001 —— 进度回调异常不得影响任务
                    pass
            if not tensors:
                continue
            try:
                with torch.no_grad():
                    batch_t = torch.stack(tensors, dim=0).to(self.device)
                    if self.use_fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            feats = self.model(batch_t)
                    else:
                        feats = self.model(batch_t)
                    feats = feats.float().cpu().numpy()
                all_feats.append(feats)
                ok_paths.extend(ok)
            except Exception as e:  # noqa: BLE001 —— 单批前向失败不影响后续
                LOGGER.warning("精排批次前向失败（%d 张）已跳过: %r", len(ok), e)
        thr.join(timeout=5)

        if not all_feats:
            LOGGER.warning("精排：%d 张全部失败", len(paths))
            return [], None
        out = np.concatenate(all_feats, axis=0).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        out /= norms
        rate = len(ok_paths) / max(time.time() - t0, 1e-6)
        LOGGER.info("精排：%d/%d 张成功（%d 维），吞吐 %.1f 张/秒",
                    len(ok_paths), len(paths), out.shape[1], rate)
        return ok_paths, out
