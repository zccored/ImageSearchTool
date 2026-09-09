# -*- coding: utf-8 -*-
"""
通用 IO / 图像解码 / 日志工具。

解码策略（兼顾速度、内存与“安静”）：
  * PNG 一律走 Pillow —— OpenCV 内嵌 libpng 会对真实世界 PNG 的
    iCCP/cHRM/巨型 chunk 刷 stderr 警告（用户图库里大量出现），
    而 Pillow 自带 PNG 解码器无此噪音，且支持巨型长图；
  * 其余格式（JPEG/WebP/TIFF/BMP…）走 OpenCV 快速路径（EXIF 检测后回退 Pillow）；
  * 超大图保护：超过 MAX_IMAGE_PIXELS 的图直接跳过不解码；
    较大图（>12M 像素）的解码由全局信号量限制并发，防内存峰值爆掉；
  * 全局屏蔽 Pillow 的 DecompressionBombWarning（上限仍由 MAX_IMAGE_PIXELS 兜底）。
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
import threading
import warnings
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger("hybrid_search")

# ---- 超大图安全策略（进程级生效，也惠及 GUI 缩略图等直接调用 PIL 的地方）----
# 默认 PIL 上限约 89M 像素，真实大图库（长图/全景 PNG）常超限 → 提高到 256M
# 像素（RGB 全解码约 1GB，属可承受峰值）；再大视为异常文件跳过。
Image.MAX_IMAGE_PIXELS = 256 * 1024 * 1024
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
# 巨型图解码并发钳制：阈值 12MP 之上的大图同时解码数上限（默认 16）。
# 经验值：>12MP 图片 RGB 峰值 ~36MB/张、34MP ~100MB/张；
# 14 物理核/20 线程、16GB 内存机（建库时可用 ≥5GB）取 16~20。
# 运行期可经 set_big_decode_limit() 调整（GUI/CLI 的“大图解码并发”参数）。
_BIG_IMAGE_PX = 12_000_000          # 约 3500×3500
_big_decode_limit = [16]
_big_decode_active = [0]
_big_decode_cond = threading.Condition()


def set_big_decode_limit(n: int) -> None:
    """动态调整大图解码并发上限（引擎按 Config.big_decode_conc 调用）。"""
    with _big_decode_cond:
        _big_decode_limit[0] = max(1, int(n))


def _big_decode_enter() -> None:
    with _big_decode_cond:
        while _big_decode_active[0] >= _big_decode_limit[0]:
            _big_decode_cond.wait()
        _big_decode_active[0] += 1


def _big_decode_exit() -> None:
    with _big_decode_cond:
        _big_decode_active[0] -= 1
        _big_decode_cond.notify()
# 顺带把 OpenCV 自家日志提到 ERROR 级，压掉残余的 C 层杂音
try:
    cv2.setLogLevel(0)  # cv2.logging.LOG_LEVEL_ERROR
except Exception:  # noqa: BLE001 —— 版本差异时忽略
    pass


def setup_logging(verbose: bool = False) -> None:
    """配置根日志：统一时间/级别/消息格式，输出到 stderr。"""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-5s %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def collect_images(root_dir: str, extensions: Iterable[str], limit: Optional[int] = None,
                   sort: bool = True) -> List[str]:
    """
    递归收集 root_dir 下所有指定扩展名的图片路径。

    limit：只取前 N 张（小规模冒烟测试用）；sort：自然排序保证多次构建顺序一致。
    """
    exts = {e.lower() for e in extensions}
    out = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                out.append(os.path.join(dirpath, name))
    if sort:
        out.sort(key=natural_key)
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def natural_key(text: str):
    """自然排序键：'img2.jpg' < 'img10.jpg'。"""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def file_md5(path: str) -> str:
    """流式计算文件 MD5（大图库去重用，读取速度远快于解码）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_open(path: str) -> Optional[Image.Image]:
    """Pillow 打开 + EXIF 方向转正，失败返回 None。"""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            return im.copy()
    except Exception as e:  # noqa: BLE001 —— 解码失败原因多样，统一交给调用方跳过
        LOGGER.debug("解码失败 %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# 高速解码：整文件一次性读入内存，MD5 与解码共用同一份字节；
# 返回 numpy 数组不持有 PIL 对象，线程池里调用完全安全。
# ---------------------------------------------------------------------------
def read_bytes(path: str) -> Optional[bytes]:
    """一次性读入文件全部字节（失败返回 None）。"""
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        LOGGER.debug("读取失败 %s: %s", path, e)
        return None


def _probe(data: bytes) -> Optional[Tuple[str, Tuple[int, int], int]]:
    """
    只读文件头：返回 (PIL格式名, (宽, 高), EXIF方向1..8)。
    失败返回 None。该步骤不会触发像素解码，因此没有任何解码噪音。
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            orient = 1
            try:
                orient = int(im.getexif().get(0x0112, 1))
            except Exception:  # noqa: BLE001 —— 无/坏 EXIF 视为方向 1
                orient = 1
            return im.format, (im.width, im.height), orient
    except Exception:  # noqa: BLE001
        return None


def _apply_orientation(arr: np.ndarray, orient: int) -> np.ndarray:
    """
    把 EXIF 方向 2..8 用 numpy 直接实现（与 Pillow exif_transpose 语义一致），
    这样大图走 cv2 域缩放解码后也能就地转正，不必退回 Pillow 全尺寸解码。
    """
    if orient in (None, 1):
        return arr
    if orient == 2:                       # 水平镜像
        return arr[:, ::-1]
    if orient == 3:                       # 旋转 180°
        return np.rot90(arr, 2, axes=(0, 1))
    if orient == 4:                       # 垂直镜像
        return arr[::-1, :]
    if orient == 5:                       # 主对角线转置
        return np.swapaxes(arr, 0, 1)
    if orient == 6:                       # 顺时针 90°
        return np.rot90(arr, 3, axes=(0, 1))
    if orient == 7:                       # 反对角线转置
        return np.rot90(np.swapaxes(arr, 0, 1)[:, ::-1], 3, axes=(0, 1))
    if orient == 8:                       # 逆时针 90°
        return np.rot90(arr, 1, axes=(0, 1))
    return arr


# 解码输出目标最长边：超过即让解码器“域缩放”直接解小图（JPEG 采样缩放，
# 解码像素量按 1/4 / 1/16 / 1/64 下降，指纹/224 中心窗完全够用）
_DECODE_TARGET_SIDE = 2048


def _reduced_flag(w: int, h: int, gray: bool) -> int:
    """
    依据原图最长边选择 cv2 解码 flag：
      目标输出最长边 ~2048：全解 / 1/2 / 1/4 / 1/8 逐级选择。
    仅 JPEG 支持可靠的采样域缩放；其他格式返回全尺寸 flag。
    """
    max_side = max(w, h)
    if max_side <= _DECODE_TARGET_SIDE * 1.25:      # <=2560 全尺寸解码
        return cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    if max_side <= _DECODE_TARGET_SIDE * 2.5:       # ~5120 -> 1/2
        return (cv2.IMREAD_REDUCED_GRAYSCALE_2 if gray
                else cv2.IMREAD_REDUCED_COLOR_2)
    if max_side <= _DECODE_TARGET_SIDE * 5:         # ~10240 -> 1/4
        return (cv2.IMREAD_REDUCED_GRAYSCALE_4 if gray
                else cv2.IMREAD_REDUCED_COLOR_4)
    return (cv2.IMREAD_REDUCED_GRAYSCALE_8 if gray
            else cv2.IMREAD_REDUCED_COLOR_8)        # 更大 -> 1/8


def _pil_decoded(data: bytes, gray: bool) -> Optional[np.ndarray]:
    """纯 Pillow 解码（无 libpng 噪音），支持超大图；调用前已过 _probe 检查。"""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            if gray:
                im = im.convert("L")
            elif im.mode != "RGB":
                im = im.convert("RGB")
            return np.asarray(im, dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("Pillow 解码失败: %r", e)
        return None


def _decode_png(data: bytes, gray: bool,
                probe: Tuple[str, Tuple[int, int], int]) -> Optional[np.ndarray]:
    """
    PNG 专用：Pillow 全路径解码（OpenCV 的 libpng 会向 stderr 刷
    iCCP/cHRM/巨型 chunk 警告，而 Pillow 自带 PNG 解码器完全安静）。
    大图（>12M 像素）受信号量钳制并发，防多线程叠加把内存打爆。
    """
    w, h = probe[1]
    if w * h > _BIG_IMAGE_PX:
        _big_decode_enter()
        try:
            return _pil_decoded(data, gray)
        finally:
            _big_decode_exit()
    return _pil_decoded(data, gray)


def _decode_opencv(data: bytes, gray: bool,
                   probe: Tuple[str, Tuple[int, int], int]) -> Optional[np.ndarray]:
    """
    大图 JPEG/WebP 走 cv2 域缩放解码（输出约 2048 最长边），
    12MP 照片解码时间约为全尺寸的 1/4~1/8。

    注意：OpenCV（≥4.1 实测 4.11）的 imdecode 默认会应用 EXIF 方向
    （IMREAD_* 不带 IMREAD_IGNORE_ORIENTATION 时自动转正），full 与
    REDUCED_* 分支行为一致；因此这里不需要再手工旋转，
    Pillow 兜底路径的 exif_transpose 语义与之对齐。
    """
    fmt, (w, h), _orient = probe
    if fmt not in ("JPEG", "WEBP"):
        return None                     # 其他格式交给专用/兜底路径
    flag = _reduced_flag(w, h, gray)
    try:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    except Exception:  # noqa: BLE001 —— 解码异常走兜底
        return None
    if arr is None:
        return None
    if not gray:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


# PNG 解码器选择：'cv2'（默认，libpng 全尺寸解码，比 Pillow 快 ~1.3x，
# 代价：个别坏 iCCP 文件的 libpng 警告会出现在 stderr）/ 'pillow'（安静、慢）
_PNG_DECODER = "cv2"


def set_png_decoder(mode: str) -> None:
    """全局切换 PNG 解码器（由 Config.png_decoder 驱动，运行时生效）。"""
    global _PNG_DECODER
    if mode in ("cv2", "pillow"):
        _PNG_DECODER = mode


def _decode_png_cv2(data: bytes, gray: bool,
                    probe: Tuple[str, Tuple[int, int], int]) -> Optional[np.ndarray]:
    """
    PNG -> OpenCV/libpng 全尺寸解码（无域缩放收益，故不解 reduced）。
    输出与 Pillow 全尺寸语义一致（无色彩管理、原像素）。
    大图（>12M 像素）与 Pillow 路径一样受信号量钳制并发。
    失败/异常返回 None 交给 Pillow 兜底。
    """
    w, h, orient = probe[1][0], probe[1][1], probe[2]
    if orient != 1:
        return None                     # 带 EXIF 方向交给 Pillow exif_transpose
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    try:
        if w * h > _BIG_IMAGE_PX:
            _big_decode_enter()
            try:
                arr = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
            finally:
                _big_decode_exit()
        else:
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    except Exception:                    # noqa: BLE001 —— 解码异常走 Pillow 兜底
        return None
    if arr is None:
        return None
    if not gray:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def decode_gray(data: bytes) -> Optional[np.ndarray]:
    """字节 -> 灰度 uint8 数组。
    PNG：默认 OpenCV/libpng 全尺寸（快 ~1.3x；坏 iCCP 文件会有零星 stderr 警告），
    Pillow 兜底；JPEG/WebP 大图走 cv2 域缩放。"""
    probe = _probe(data)
    if probe is None:
        LOGGER.debug("无法识别图像头")
        return None
    fmt, (w, h), _ = probe
    if w * h > Image.MAX_IMAGE_PIXELS:
        LOGGER.debug("超大图跳过: %dx%d 像素超过上限", w, h)
        return None
    if fmt == "PNG":
        if _PNG_DECODER == "cv2":
            arr = _decode_png_cv2(data, gray=True, probe=probe)
            if arr is not None:
                return arr
        return _decode_png(data, gray=True, probe=probe)
    arr = _decode_opencv(data, gray=True, probe=probe)
    if arr is not None:
        return arr
    return _pil_decoded(data, gray=True)


def decode_rgb(data: bytes) -> Optional[np.ndarray]:
    """字节 -> RGB uint8 数组。
    PNG：默认 OpenCV/libpng 全尺寸（快 ~1.3x），Pillow 兜底；
    JPEG/WebP 大图走 cv2 域缩放。"""
    probe = _probe(data)
    if probe is None:
        LOGGER.debug("无法识别图像头")
        return None
    fmt, (w, h), _ = probe
    if w * h > Image.MAX_IMAGE_PIXELS:
        LOGGER.debug("超大图跳过: %dx%d 像素超过上限", w, h)
        return None
    if fmt == "PNG":
        if _PNG_DECODER == "cv2":
            arr = _decode_png_cv2(data, gray=False, probe=probe)
            if arr is not None:
                return arr
        return _decode_png(data, gray=False, probe=probe)
    arr = _decode_opencv(data, gray=False, probe=probe)
    if arr is not None:
        return arr
    return _pil_decoded(data, gray=False)


def load_gray(path: str) -> Optional[np.ndarray]:
    """路径 -> 灰度 uint8 数组（粗筛用，内部单次读盘）。"""
    data = read_bytes(path)
    return decode_gray(data) if data is not None else None


def load_rgb(path: str) -> Optional[np.ndarray]:
    """路径 -> RGB uint8 数组（ResNet 精排用，内部单次读盘）。"""
    data = read_bytes(path)
    return decode_rgb(data) if data is not None else None


def is_valid_image(path: str) -> bool:
    """轻量校验：能否被解码（用于索引前的预筛）。"""
    im = _safe_open(path)
    return im is not None


def human_bytes(n: float) -> str:
    """把字节数显示成友好单位。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def image_bytes_summary(image_paths: List[str]) -> int:
    """所有图片文件字节数合计（粗略内存预估用）。"""
    total = 0
    for p in image_paths:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total
