# -*- coding: utf-8 -*-
"""全量建库断点特征检查点（path-keyed，512 维 fp32）。

与 `prep_cache_torch` 的区别：
  * 存的是**最终精排特征** + 粗筛记录，不是 224 张量；
  * 命中后不再 decode/coarse/GPU forward，直接进索引；
  * 按路径键复用，天然支持“中断后从任意已完成处继续”；
  * manifest 只提交完整前缀，尾部未提交记录忽略；
  * 校验失败一律拒绝，交给上层回退到 u16 张量缓存/冷构建。

多盘预留：Entry 中保留 `disk_hint` 字段（当前写空字符串），未来用于
“同一图库跨盘、按盘读写效率做性能天平”的调度；本版本不做任何调度。
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .coarse import CoarseRecord
from .io_utils import LOGGER

MAGIC = b"FCP1"
DIRNAME = "feature_checkpoint"
MANIFEST = "manifest.json"
ENTRIES = "entries.bin"
VERSION = 1


@dataclass
class FeatureEntry:
    path: str
    size: int
    mtime_ns: int
    feature: np.ndarray
    hu: np.ndarray
    fp: np.ndarray
    md5: str
    disk_hint: str = ""          # 预留：多盘性能天平用

    def rec(self) -> CoarseRecord:
        return CoarseRecord(path=self.path, md5=self.md5,
                            hu=self.hu.copy(), fp=self.fp.copy())


def make_signature(cfg, feature_dim: int) -> dict:
    """预处理/模型/粗筛契约；任一变化都让检查点整体失效。"""
    from .prep_torch import CROP, SIDE
    return {
        "v": VERSION,
        "model": str(getattr(cfg, "model", "")),
        "backend": str(getattr(cfg, "backend", "auto")),
        "openvino_device": str(getattr(cfg, "openvino_device", "auto")),
        "openvino_model": str(getattr(cfg, "openvino_model", "")),
        "prep": str(getattr(cfg, "prep", "cpu")),
        "resize": int(SIDE),
        "crop": int(CROP),
        "feature_dim": int(feature_dim),
        "feature_dtype": "float32",
        "coarse_size": int(getattr(cfg, "coarse_size", 64)),
        "blur": int(getattr(cfg, "coarse_blur", 5)),
        "binarize_norm": str(getattr(cfg, "binarize_norm", "none")),
        "invert_binary": bool(getattr(cfg, "invert_binary", False)),
        "use_hu": bool(getattr(cfg, "use_hu", True)),
        "use_fp": bool(getattr(cfg, "use_fp", True)),
    }


def _path_key(path: str) -> Optional[str]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    raw = f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class FeatureCheckpoint:
    """断点特征检查点；append 只写完整记录，commit 用 manifest 标记可信前缀。"""

    def __init__(self, prefix: str, cfg, feature_dim: int, enabled: bool = True):
        self.root = os.path.join(os.path.dirname(os.path.abspath(prefix)), DIRNAME)
        self.enabled = bool(enabled)
        self.sig = make_signature(cfg, feature_dim)
        self.feature_dim = int(feature_dim)
        self.manifest_path = os.path.join(self.root, MANIFEST)
        self.entries_path = os.path.join(self.root, ENTRIES)
        self.entries: Dict[str, FeatureEntry] = {}
        self.loaded_from_disk = False
        self.skipped_changed = 0
        self.skipped_missing = 0
        self.writes = 0
        self.bytes_written = 0
        self._n_records = 0
        self._hash = hashlib.sha256()
        self._bytes = 0
        self._fh = None
        if not self.enabled:
            return
        try:
            os.makedirs(self.root, exist_ok=True)
        except Exception as e:                            # noqa: BLE001
            LOGGER.warning("特征检查点目录不可用：%r", e)
            self.enabled = False

    # ---- 读 ----------------------------------------------------------
    def load(self, allowed: Optional[Iterable[str]] = None) -> Optional[Dict[str, FeatureEntry]]:
        """读取并校验结构；失败返回 None（调用方回退）。"""
        if not self.enabled or not os.path.exists(self.manifest_path):
            return None
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            if int(m.get("v", 0)) != VERSION or m.get("sig") != self.sig:
                LOGGER.info("特征检查点签名不匹配，忽略")
                return None
            n = int(m.get("n", 0))
            nbytes = int(m.get("bytes", 0))
            digest = str(m.get("sha256", ""))
            if n < 0 or nbytes <= 0:
                return None
            with open(self.entries_path, "rb") as fh:
                blob = fh.read(nbytes)
            if len(blob) != nbytes or hashlib.sha256(blob).hexdigest() != digest:
                LOGGER.warning("特征检查点 entries 哈希不匹配，拒绝")
                return None
            entries, off = self._parse(blob, n, allowed)
            if entries is None:
                return None
            self.entries = entries
            self._n_records = n
            self._bytes = nbytes
            self._hash = hashlib.sha256(blob)
            self.loaded_from_disk = True
            # 截掉上次崩溃留下的未提交尾部
            try:
                if os.path.getsize(self.entries_path) > nbytes:
                    with open(self.entries_path, "r+b") as fh:
                        fh.truncate(nbytes)
            except OSError:
                pass
            LOGGER.info("特征检查点载入：%d 条（跳过变更 %d / 缺失 %d）",
                        len(entries), self.skipped_changed, self.skipped_missing)
            return entries
        except FileNotFoundError:
            return None
        except Exception as e:                            # noqa: BLE001
            LOGGER.warning("特征检查点读取失败，拒绝使用：%r", e)
            return None

    def _parse(self, blob: bytes, n: int, allowed) -> Tuple[Optional[Dict[str, FeatureEntry]], int]:
        out: Dict[str, FeatureEntry] = {}
        off = 0
        allowed_set = set(map(str, allowed)) if allowed is not None else None
        for i in range(n):
            rstart = off
            try:
                path_len = struct.unpack_from("<I", blob, off)[0]; off += 4
                path = blob[off:off + path_len].decode("utf-8"); off += path_len
                size, mtime_ns = struct.unpack_from("<qq", blob, off); off += 16
                dim = struct.unpack_from("<H", blob, off)[0]; off += 2
                feat = np.frombuffer(blob, dtype="<f4", count=dim, offset=off).copy(); off += dim * 4
                hlen = struct.unpack_from("<B", blob, off)[0]; off += 1
                hu = np.frombuffer(blob, dtype="<f4", count=hlen, offset=off).copy(); off += hlen * 4
                flen = struct.unpack_from("<I", blob, off)[0]; off += 4
                fp = np.frombuffer(blob, dtype=np.uint8, count=flen, offset=off).copy(); off += flen
                mlen = struct.unpack_from("<B", blob, off)[0]; off += 1
                md5 = blob[off:off + mlen].decode("ascii"); off += mlen
                dlen = struct.unpack_from("<B", blob, off)[0]; off += 1
                disk_hint = blob[off:off + dlen].decode("utf-8"); off += dlen
                crc = struct.unpack_from("<I", blob, off)[0]; off += 4
            except Exception as e:                        # noqa: BLE001
                LOGGER.warning("特征检查点第 %d 条解析失败：%r", i, e)
                return None, off
            if zlib.crc32(blob[rstart:off - 4]) & 0xFFFFFFFF != crc:
                LOGGER.warning("特征检查点第 %d 条 CRC 不匹配，拒绝", i)
                return None, off
            if dim != self.feature_dim:
                LOGGER.warning("特征检查点第 %d 条维度 %d != %d，拒绝", i, dim, self.feature_dim)
                return None, off
            if allowed_set is not None and path not in allowed_set:
                continue
            try:
                st = os.stat(path)
            except OSError:
                self.skipped_missing += 1
                continue
            if st.st_size != size or st.st_mtime_ns != mtime_ns:
                self.skipped_changed += 1
                continue
            if feat.shape != (self.feature_dim,) or not np.isfinite(feat).all():
                LOGGER.warning("特征检查点第 %d 条含 NaN/Inf，拒绝", i)
                return None, off
            norm = float(np.linalg.norm(feat))
            if not (0.9998 <= norm <= 1.0002):
                LOGGER.warning("特征检查点第 %d 条范数 %.6f 异常，拒绝", i, norm)
                return None, off
            out[path] = FeatureEntry(path, int(size), int(mtime_ns), feat,
                                     hu, fp, md5, disk_hint)
        return out, off

    # ---- 写 ----------------------------------------------------------
    def append(self, paths: Sequence[str], features: np.ndarray,
               recs: Sequence[CoarseRecord], commit: bool = True) -> int:
        """追加已计算特征；已存在的路径跳过。返回新增条数。"""
        if not self.enabled:
            return 0
        add = 0
        if features.ndim != 2 or features.shape[0] != len(paths):
            return 0
        if self._fh is None:
            self._fh = open(self.entries_path, "ab")
        for path, feat, rec in zip(paths, features, recs):
            if path in self.entries:
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            if feat.shape != (self.feature_dim,) or not np.isfinite(feat).all():
                continue
            norm = float(np.linalg.norm(feat))
            if not (0.9998 <= norm <= 1.0002):
                continue
            hu = np.asarray(rec.hu if rec is not None and rec.hu is not None else np.zeros(7),
                            dtype="<f4")
            fp = np.asarray(rec.fp if rec is not None and rec.fp is not None else np.zeros(0, np.uint8),
                            dtype=np.uint8)
            md5 = (rec.md5 if rec is not None else "") or ""
            body = self._encode(path, st, feat, hu, fp, md5, "")
            crc = zlib.crc32(body) & 0xFFFFFFFF
            rec_bytes = body + struct.pack("<I", crc)
            try:
                self._fh.write(rec_bytes)
                self._fh.flush()
            except Exception as e:                        # noqa: BLE001
                LOGGER.warning("特征检查点写入失败：%r", e)
                break
            self._hash.update(rec_bytes)
            self._bytes += len(rec_bytes)
            self.entries[path] = FeatureEntry(path, int(st.st_size), int(st.st_mtime_ns),
                                              np.asarray(feat, dtype=np.float32),
                                              hu, fp, md5, "")
            self.writes += 1
            self.bytes_written += len(rec_bytes)
            self._n_records += 1
            add += 1
        if commit and add:
            self.commit()
        return add

    def _encode(self, path: str, st, feat, hu, fp, md5: str, disk_hint: str) -> bytes:
        pb = path.encode("utf-8")
        fb = np.asarray(feat, dtype="<f4").tobytes()
        hb = np.asarray(hu, dtype="<f4").tobytes()
        fpb = np.asarray(fp, dtype=np.uint8).tobytes()
        mb = md5.encode("ascii")
        db = disk_hint.encode("utf-8")
        return (struct.pack("<I", len(pb)) + pb +
                struct.pack("<qq", int(st.st_size), int(st.st_mtime_ns)) +
                struct.pack("<H", len(fb) // 4) + fb +
                struct.pack("<B", len(hb) // 4) + hb +
                struct.pack("<I", len(fpb)) + fpb +
                struct.pack("<B", len(mb)) + mb +
                struct.pack("<B", len(db)) + db)

    def commit(self) -> None:
        if not self.enabled or self._bytes <= 0:
            return
        m = {"v": VERSION, "sig": self.sig, "n": self._n_records,
             "bytes": self._bytes, "sha256": self._hash.hexdigest(),
             "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.manifest_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(m, fh, ensure_ascii=False)
            os.replace(tmp, self.manifest_path)
        except Exception as e:                            # noqa: BLE001
            LOGGER.warning("特征检查点 manifest 提交失败：%r", e)


    def verify_semantic(self, entries: Dict[str, "FeatureEntry"], extractor,
                        tensor_cache=None, sample: int = 64,
                        threshold: float = 0.9999) -> dict:
        """抽样重算：用 u16 张量缓存取 224 张量，走当前模型，比较检查点特征。

        只做 K 条，成本 K × 0.5 ms；没有张量缓存则返回 checked=0（调用方决定是否信任）。
        """
        vals = list(entries.values())
        if not vals:
            return {"ok": True, "checked": 0, "skipped": 0, "cos_min": None}
        if sample <= 0:
            sample = len(vals)
        step = max(1, len(vals) // sample)
        pick = vals[::step][:sample]
        ready = []
        skipped = 0
        for e in pick:
            if tensor_cache is None:
                skipped += 1
                continue
            try:
                hit = tensor_cache.get(e.path)
            except Exception:                             # noqa: BLE001
                hit = None
            if hit is None:
                skipped += 1
                continue
            t = hit[0]
            try:
                import torch
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
            except Exception:                             # noqa: BLE001
                skipped += 1
                continue
            ready.append((e, t))
        if not ready:
            return {"ok": True, "checked": 0, "skipped": skipped, "cos_min": None}
        try:
            feats = extractor._forward([t for _, t in ready])  # noqa: SLF001
            feats = np.asarray(feats, dtype=np.float32)
            feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
            ref = np.stack([e.feature for e, _ in ready]).astype(np.float32)
            cos = (feats * ref).sum(axis=1)
            ok = bool(np.isfinite(cos).all() and float(cos.min()) >= threshold)
            return {"ok": ok, "checked": len(ready), "skipped": skipped,
                    "cos_min": float(cos.min()), "cos_mean": float(cos.mean())}
        except Exception as e:                            # noqa: BLE001
            return {"ok": False, "checked": 0, "skipped": skipped,
                    "error": repr(e), "cos_min": None}

    def reset(self) -> None:
        """签名不匹配/损坏时丢弃旧代，避免旧记录混进新 manifest。"""
        self.close()
        for p in (self.manifest_path, self.entries_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        self.entries = {}
        self._n_records = 0
        self._bytes = 0
        self._hash = hashlib.sha256()
        self.loaded_from_disk = False

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:                             # noqa: BLE001
                pass
            self._fh = None

    def summary(self) -> str:
        return "特征检查点：%d 条（%.1f MB）/ 已载入 %s" % (
            len(self.entries), self._bytes / 1e6, self.loaded_from_disk)
