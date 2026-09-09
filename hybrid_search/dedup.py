# -*- coding: utf-8 -*-
"""重复图查验（一对多）：完全重复（MD5）+ 近似重复（指纹条带阻塞 + 汉明复核）。

为什么要复用索引：整图索引里已经存了每张图的 md5、64×64 二值指纹与 ResNet
特征，已入库条目可直接取用（零解码）；只有未入库文件才需要读盘解码取指纹。
因此 4 万张图库通常数秒出结果，而不是把每张图都重新解一遍。

近似重复的“阻塞”策略：把 4096 位指纹切成若干条带（band），只有共享至少一条
条带的图片才两两比对（标准 LSH 阻塞思想），避免 O(N²) 全比对；再用汉明比例
复核，最后用并查集把两两关系合并成“一对多”的组。

删除默认走 Windows 回收站（可还原）；失败不做任何破坏性动作。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coarse import _hamming_distance, extract_binary_features
from .config import Config
from .io_utils import decode_gray, read_bytes
from .store import IndexFiles

__all__ = ["DupMember", "DupGroup", "DupReport", "scan_duplicates",
           "recycle_paths", "move_paths"]

# 索引里可能被改动的几何参数（决定指纹能否与索引直接比对）
_FP_CFG_KEYS = ("coarse_size", "coarse_blur", "use_hu", "use_fp",
                "invert_binary")


@dataclass
class DupMember:
    path: str
    size: int = 0
    mtime: float = 0.0
    w: int = 0
    h: int = 0
    fmt: str = ""
    md5: str = ""
    indexed: bool = False
    exact_copy: bool = False        # 与组内基准内容完全相同（MD5 一致）
    hamming: float = 0.0            # 与组内“基准”（第一张）的汉明比例
    cos: Optional[float] = None     # 与基准的 ResNet 余弦（双方都有特征时）

    @property
    def pixels(self) -> int:
        return self.w * self.h

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class DupGroup:
    kind: str                       # exact（内容完全相同）/ near（近似）
    members: List[DupMember] = field(default_factory=list)
    gid: int = 0
    max_hamming: float = 0.0
    min_cos: Optional[float] = None

    @property
    def wasted_bytes(self) -> int:
        """除第一张（默认保留）之外可释放的字节数。"""
        return sum(m.size for m in self.members[1:])

    @property
    def keep(self) -> DupMember:
        return self.members[0]

    @property
    def all_exact(self) -> bool:
        return bool(self.members) and all(m.exact_copy for m in self.members)


@dataclass
class DupReport:
    groups: List[DupGroup] = field(default_factory=list)
    scanned: int = 0
    indexed_used: int = 0
    decoded: int = 0
    elapsed: float = 0.0
    threshold: float = 0.04
    cos_threshold: Optional[float] = None
    errors: List[Tuple[str, str]] = field(default_factory=list)
    truncated: bool = False

    @property
    def n_exact(self) -> int:
        return sum(1 for g in self.groups if g.kind == "exact")

    @property
    def n_near(self) -> int:
        return sum(1 for g in self.groups if g.kind == "near")

    @property
    def n_images(self) -> int:
        return sum(len(g.members) for g in self.groups)

    @property
    def wasted_bytes(self) -> int:
        return sum(g.wasted_bytes for g in self.groups)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def scan_duplicates(paths: Sequence[str], *,
                    prefix: Optional[str] = None,
                    threshold: float = 0.04,
                    include_near: bool = True,
                    workers: Optional[int] = None,
                    progress: Optional[Callable[[int, int, str], None]] = None,
                    cancel: Optional[Callable[[], bool]] = None,
                    max_pairs: int = 4_000_000) -> DupReport:
    """扫描重复图。

    threshold：近似重复判定的汉明比例上限（0.04 = 4096 位里差异 ≤164 位）。
    prefix：整图索引前缀；给了就复用其 md5/指纹/特征（大幅加速）。
    """
    t0 = time.time()
    rep = DupReport(threshold=threshold)
    paths = [os.path.abspath(p) for p in dict.fromkeys(paths)]
    n = len(paths)
    rep.scanned = n
    if n == 0:
        rep.elapsed = time.time() - t0
        return rep
    workers = workers or max(2, min(8, (os.cpu_count() or 4)))

    idx = _IndexReuse(prefix)
    cfg = idx.cfg or Config()

    md5s: List[str] = [""] * n
    fps: List[Optional[np.ndarray]] = [None] * n
    dims: List[Tuple[int, int, str]] = [(0, 0, "")] * n
    feat_rows: List[int] = [-1] * n          # 在精排特征矩阵里的行号（-1=无）
    indexed: List[bool] = [False] * n

    # ---- 1) 索引复用 + 尺寸探测（只读文件头，快）-----------------------
    if progress:
        progress(0, n, "复用索引/读文件头")
    todo: List[int] = []
    md5_only: List[int] = []
    for i, p in enumerate(paths):
        row = idx.row_of.get(os.path.normcase(p))
        if row is not None:
            indexed[i] = True
            md5s[i] = idx.md5s[row] if row < len(idx.md5s) else ""
            fps[i] = idx.fp_row(row)
            feat_rows[i] = row if idx.has_fine else -1
            if not md5s[i]:
                md5_only.append(i)      # 索引建库时没存 md5（--no-dedup）
        else:
            todo.append(i)
    rep.indexed_used = n - len(todo)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_probe_header, paths[i]): i for i in todo}
        done = 0
        for fut in futs:
            i = futs[fut]
            try:
                dims[i] = fut.result()
            except Exception as e:              # noqa: BLE001
                rep.errors.append((paths[i], f"{type(e).__name__}: {e}"))
            done += 1
            if progress and done % 200 == 0:
                progress(done, len(todo), "读文件头")
    # 已入库条目也补一下尺寸（预览与“保留最佳”都要用）
    need_dim = [i for i in range(n) if indexed[i] and dims[i][0] == 0]
    if need_dim:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_probe_header, paths[i]): i for i in need_dim}
            for fut in futs:
                i = futs[fut]
                try:
                    dims[i] = fut.result()
                except Exception:               # noqa: BLE001
                    pass

    # ---- 2) 未入库文件：md5 + 指纹（解码一次同时拿两样）----------------
    if md5_only:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_md5_of, paths[i]): i for i in md5_only}
            for fut in futs:
                try:
                    m = fut.result()
                    if m:
                        md5s[futs[fut]] = m
                except Exception as e:          # noqa: BLE001
                    rep.errors.append((paths[futs[fut]],
                                       f"{type(e).__name__}: {e}"))
    if todo:
        if progress:
            progress(0, len(todo), "解码未入库文件(取指纹+MD5)")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_md5_and_fp, paths[i], cfg, not md5s[i]): i
                    for i in todo}
            done = 0
            for fut in futs:
                i = futs[fut]
                done += 1
                if cancel and cancel():
                    rep.truncated = True
                    break
                try:
                    md5, fp = fut.result()
                    if md5:
                        md5s[i] = md5
                    if fp is not None:
                        fps[i] = fp
                        rep.decoded += 1
                except Exception as e:          # noqa: BLE001
                    rep.errors.append((paths[i], f"{type(e).__name__}: {e}"))
                if progress and done % 100 == 0:
                    progress(done, len(todo), "解码未入库文件")
    if rep.truncated:
        rep.elapsed = time.time() - t0
        return rep

    # ---- 3) 完全重复：按 md5 分组 --------------------------------------
    if progress:
        progress(0, 1, "按内容 MD5 分组")
    by_md5: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(md5s):
        if m:
            by_md5[m].append(i)

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for m, rows in by_md5.items():
        for k in range(1, len(rows)):
            union(rows[0], rows[k])

    # ---- 4) 近似重复：条带阻塞 -> 汉明复核 -> 并查集 -------------------
    if include_near:
        if progress:
            progress(0, 1, "近似重复：条带阻塞")
        rows = [i for i in range(n) if fps[i] is not None]
        if len(rows) >= 2:
            F = np.stack([fps[i] for i in rows]).astype(np.uint8)
            pairs = _band_pairs(F)
            if len(pairs) > max_pairs:
                rep.truncated = True
                pairs = set(list(pairs)[:max_pairs])
            if pairs:
                if progress:
                    progress(0, len(pairs), "近似重复：汉明复核")
                cand = np.fromiter((x for pr in pairs for x in pr),
                                   dtype=np.int64).reshape(-1, 2)
                d = _hamming_distance(np.bitwise_xor(F[cand[:, 0]],
                                                     F[cand[:, 1]])) \
                    / float(F.shape[1] * 8)
                keep = d <= threshold
                for a, b in cand[keep]:
                    union(rows[int(a)], rows[int(b)])

    # ---- 5) 组内整理 ---------------------------------------------------
    comp: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    groups: List[DupGroup] = []
    for members_idx in comp.values():
        if len(members_idx) < 2:
            continue
        rows: List[Tuple[int, DupMember]] = []
        for i in members_idx:
            try:
                st = os.stat(paths[i])
                size, mtime = int(st.st_size), float(st.st_mtime)
            except OSError:
                size, mtime = 0, 0.0
            w, h, fmt = dims[i]
            rows.append((i, DupMember(path=paths[i], size=size, mtime=mtime,
                                      w=w, h=h, fmt=fmt, md5=md5s[i],
                                      indexed=indexed[i])))
        # 默认保留“最像原图”的那张：像素最多 → 体积最大 → 最新
        rows.sort(key=lambda t: (-t[1].pixels, -t[1].size, -t[1].mtime))
        base_i = rows[0][0]
        base_fp = fps[base_i]
        base_feat = feat_rows[base_i]
        base_md5 = md5s[base_i]
        ms: List[DupMember] = []
        for i, m in rows:
            if fps[i] is not None and base_fp is not None:
                m.hamming = float(_hamming_distance(
                    np.bitwise_xor(fps[i][None, :], base_fp[None, :]))[0]) \
                    / float(len(fps[i]) * 8)
            m.cos = idx.cos_of(feat_rows[i], base_feat)
            m.exact_copy = bool(base_md5) and m.md5 == base_md5
            ms.append(m)
        md5_set = {m.md5 for m in ms if m.md5}
        kind = "exact" if len(md5_set) == 1 else "near"
        g = DupGroup(kind=kind, members=ms,
                     max_hamming=max((m.hamming for m in ms), default=0.0),
                     min_cos=min((m.cos for m in ms if m.cos is not None),
                                 default=None))
        groups.append(g)
    groups.sort(key=lambda g: -g.wasted_bytes)
    for k, g in enumerate(groups, 1):
        g.gid = k
    rep.groups = groups
    rep.elapsed = time.time() - t0
    if progress:
        progress(1, 1, "完成")
    return rep


# ---------------------------------------------------------------------------
# 索引复用
# ---------------------------------------------------------------------------
class _IndexReuse:
    """只读复用整图索引里的 md5/指纹/精排特征（不加载 torch 模型）。"""

    def __init__(self, prefix: Optional[str]):
        self.cfg: Optional[Config] = None
        self.row_of: Dict[str, int] = {}
        self.md5s: List[str] = []
        self._fp = None
        self._feats = None
        self.has_fine = False
        if not prefix:
            return
        files = IndexFiles(prefix)
        if not files.meta_exists():
            return
        try:
            meta = files.load_meta()
            st = files.load_coarse()
            self.md5s = list(st["md5s"])
            self._fp = st["fp"]
            self.row_of = {os.path.normcase(os.path.abspath(p)): i
                           for i, p in enumerate(st["paths"])}
            # 指纹参数必须与索引一致，否则汉明距离没有意义
            mc = meta.get("cfg", {})
            cfg = Config()
            if all(k in mc for k in _FP_CFG_KEYS):
                cfg.coarse_size = int(mc["coarse_size"])
                cfg.coarse_blur = int(mc["coarse_blur"])
                cfg.use_hu = bool(mc["use_hu"])
                cfg.use_fp = bool(mc["use_fp"])
                cfg.invert_binary = bool(mc["invert_binary"])
            self.cfg = cfg
            if files.fine_exists():
                fine = files.load_fine(mmap=True)
                self._feats = fine["features"]
                self.has_fine = self._feats is not None
        except Exception:                       # noqa: BLE001 —— 索引坏了就退化为纯解码
            self.row_of = {}
            self._fp = None
            self._feats = None

    def fp_row(self, row: int) -> Optional[np.ndarray]:
        if self._fp is None or row >= self._fp.shape[0]:
            return None
        return self._fp[row]

    def cos_of(self, row_a: int, row_b: int) -> Optional[float]:
        if self._feats is None or row_a < 0 or row_b < 0:
            return None
        if row_a >= self._feats.shape[0] or row_b >= self._feats.shape[0]:
            return None
        a = np.asarray(self._feats[row_a], dtype=np.float32)
        b = np.asarray(self._feats[row_b], dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return None
        return float(np.dot(a, b) / (na * nb))


def _probe_header(path: str) -> Tuple[int, int, str]:
    """只读文件头拿尺寸/格式（不整张解码）。"""
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h), (im.format or "").upper()
    except Exception:                           # noqa: BLE001
        return 0, 0, ""


def _md5_of(path: str) -> str:
    """只算文件 MD5（不解码）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_and_fp(path: str, cfg: Config,
                need_md5: bool) -> Tuple[str, Optional[np.ndarray]]:
    """读一次文件字节：MD5 + 解码灰度 -> 二值指纹。"""
    data = read_bytes(path)
    if data is None:
        return "", None
    md5 = hashlib.md5(data).hexdigest() if need_md5 else ""
    gray = decode_gray(data)
    if gray is None:
        return md5, None
    try:
        _bin, _hu, fp = extract_binary_features(gray, cfg)
    except Exception:                           # noqa: BLE001
        return md5, None
    return md5, fp


def _band_pairs(F: np.ndarray, n_bands: int = 64,
                max_group: int = 64) -> set:
    """条带阻塞：返回候选行对（组内下标）。

    4096 位指纹切成 n_bands 段，段内字节完全相同才成为候选。汉明比例 t 越小，
    单段全同的概率越低，但 64 段叠加后仍能覆盖绝大部分真实重复对。
    """
    n, nbytes = F.shape
    if nbytes == 0 or n < 2:
        return set()
    n_bands = max(1, min(n_bands, nbytes))
    band_bytes = max(1, nbytes // n_bands)
    n_bands = nbytes // band_bytes
    Fb = F.reshape(n, n_bands, band_bytes)
    dt = np.dtype((np.void, band_bytes))
    pairs = set()
    for b in range(n_bands):
        band = np.ascontiguousarray(Fb[:, b, :])
        view = band.view(dt).ravel()
        _uniq, inv = np.unique(view, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        sv = inv[order]
        bounds = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1], True])
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e - s < 2:
                continue
            idxs = order[s:e][:max_group]
            m = len(idxs)
            for a in range(m):
                ia = int(idxs[a])
                for c in range(a + 1, m):
                    pairs.add((ia, int(idxs[c])))
    return pairs


# ---------------------------------------------------------------------------
# 文件操作
# ---------------------------------------------------------------------------
def recycle_paths(paths: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """移入 Windows 回收站（可还原）。返回 (成功, [(失败路径, 原因)])。

    非 Windows 或 API 不可用时全部计入失败 —— 绝不做不可逆删除。
    """
    ok: List[str] = []
    bad: List[Tuple[str, str]] = []
    paths = [os.path.abspath(p) for p in paths]
    if os.name != "nt":
        return [], [(p, "当前系统不支持回收站删除") for p in paths]
    try:
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND),
                        ("wFunc", wintypes.UINT),
                        ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR),
                        ("fFlags", ctypes.c_ushort),
                        ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p),
                        ("lpszProgressTitle", wintypes.LPCWSTR)]

        FO_DELETE = 3
        FOF_ALLOWUNDO = 0x0040
        FOF_NOCONFIRMATION = 0x0010
        FOF_NOERRORUI = 0x0400
        FOF_SILENT = 0x0004

        missing = [p for p in paths if not os.path.exists(p)]
        for p in missing:
            bad.append((p, "文件不存在"))
        todo = [p for p in paths if os.path.exists(p)]
        if todo:
            buf = "\0".join(todo) + "\0\0"
            op = SHFILEOPSTRUCTW()
            op.hwnd = None
            op.wFunc = FO_DELETE
            op.pFrom = buf
            op.pTo = None
            op.fFlags = (FOF_ALLOWUNDO | FOF_NOCONFIRMATION
                         | FOF_NOERRORUI | FOF_SILENT)
            res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
            if res == 0:
                ok = todo
            else:
                # 逐个重试，能救回多少算多少
                for p in todo:
                    op2 = SHFILEOPSTRUCTW()
                    op2.hwnd = None
                    op2.wFunc = FO_DELETE
                    op2.pFrom = p + "\0\0"
                    op2.pTo = None
                    op2.fFlags = op.fFlags
                    r2 = ctypes.windll.shell32.SHFileOperationW(
                        ctypes.byref(op2))
                    if r2 == 0:
                        ok.append(p)
                    else:
                        bad.append((p, f"SHFileOperation={r2}"))
    except Exception as e:                      # noqa: BLE001
        return [], [(p, f"{type(e).__name__}: {e}") for p in paths]
    return ok, bad


def move_paths(paths: Sequence[str], dest_root: str,
               base_root: Optional[str] = None
               ) -> Tuple[List[str], List[Tuple[str, str]]]:
    """把文件移动到 dest_root，保留相对 base_root 的目录结构；重名加后缀。"""
    ok: List[str] = []
    bad: List[Tuple[str, str]] = []
    dest_root = os.path.abspath(dest_root)
    base = os.path.abspath(base_root) if base_root else ""
    for p in paths:
        src = os.path.abspath(p)
        try:
            if base and os.path.normcase(src).startswith(os.path.normcase(base)):
                rel = os.path.relpath(src, base)
            else:
                rel = os.path.basename(src)
            dst = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                stem, ext = os.path.splitext(dst)
                k = 2
                while os.path.exists(f"{stem}_{k}{ext}"):
                    k += 1
                dst = f"{stem}_{k}{ext}"
            shutil.move(src, dst)
            ok.append(src)
        except Exception as e:                  # noqa: BLE001
            bad.append((src, f"{type(e).__name__}: {e}"))
    return ok, bad
