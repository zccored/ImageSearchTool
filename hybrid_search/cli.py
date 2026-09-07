# -*- coding: utf-8 -*-
"""
命令行入口：python main.py <子命令> [参数]

子命令：
  build      从零构建索引（粗筛 + 可选 ResNet 全库索引）
  add        增量入库（只处理新路径/新内容）
  build-fine 对已有粗筛索引补建 ResNet 全库索引
  search     以图搜图（二值法粗筛 + ResNet 精排两级流水线）
  eval       批量召回率评估（配合 make_test_dataset.py 生成的分组数据）
  stats      索引统计

所有命令都支持 --prefix 指定索引前缀（默认 ./gallery）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional

from .config import Config
from .engine import HybridEngine, Outcome
from .io_utils import LOGGER, setup_logging
from .visuals import save_contact_sheet


def _flag(a: argparse.Namespace, name: str, dflt=False):
    """Namespace 里可能没有的属性安全读取（不同子命令参数集不同）。"""
    return getattr(a, name, dflt)


def _val(a: argparse.Namespace, name: str, dflt=None):
    return getattr(a, name, dflt)


# ---------------------------------------------------------------------------
# 参数：每个子命令都注册 --prefix；建库/检索相关开关只在需要时注册
# ---------------------------------------------------------------------------
def _add_prefix(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--prefix", default="./gallery",
                    help="索引文件前缀（生成 <前缀>.meta.json/.coarse.npz/.fine.npz）")


def _add_feature_args(sp: argparse.ArgumentParser) -> None:
    """粗筛特征与精排模型相关的可调参数。"""
    sp.add_argument("--coarse-size", type=int, default=64, help="二值指纹边长")
    sp.add_argument("--blur", type=int, default=5, help="二值化前高斯模糊核（奇数）")
    sp.add_argument("--no-hu", action="store_true", help="关闭 Hu 矩特征")
    sp.add_argument("--no-fp", action="store_true", help="关闭二值指纹特征")
    sp.add_argument("--hu-weight", type=float, default=0.35, help="Hu 矩融合权重")
    sp.add_argument("--fp-weight", type=float, default=0.65, help="指纹融合权重")
    sp.add_argument("--invert-binary", action="store_true",
                    help="白像素过半时取反二值图")
    sp.add_argument("--model", default="resnet18",
                    help="ResNet 型号：resnet18/34/50/101/152")
    sp.add_argument("--device", default="auto", help="auto / cuda / cpu")
    sp.add_argument("--no-fp16", action="store_true", help="GPU 上禁用 FP16")
    sp.add_argument("--batch", type=int, default=0, help="精排批大小（0=自动）")
    sp.add_argument("--no-store-fine", action="store_true",
                    help="不预构建 ResNet 全库索引（查询时对候选实时抽特征）")
    sp.add_argument("--no-dedup", action="store_true", help="不做 MD5 内容去重")
    sp.add_argument("--workers", type=int, default=0,
                    help="粗筛特征提取并行线程数（0=自动<=8，1=串行）")
    sp.add_argument("--decode-workers", type=int, default=0,
                    help="精排解码/预处理并行线程数（0=自动<=8，CPU 解码与"
                         "GPU 前向重叠）")
    sp.add_argument("--torch-threads", type=int, default=0,
                    help="torch 推理线程数（0=保持默认）")
    sp.add_argument("--png-decoder", choices=["cv2", "pillow"], default="cv2",
                    help="PNG 解码器：cv2=libpng 全尺寸(快~1.3x；坏 iCCP 文件"
                         "零星 stderr 警告)；pillow=安静较慢")
    sp.add_argument("--ext", action="append", default=None,
                    help="额外支持的图片扩展名（可多次，如 .gif）")
    sp.add_argument("--coarse-k", type=int, default=300,
                    help="粗筛保留候选数")
    sp.add_argument("--no-exclude-self", action="store_true",
                    help="不剔除“查询图自身”这个结果")
    sp.add_argument("--no-progress", action="store_true",
                    help="关闭实时进度输出（默认开启：阶段/计数/百分比/吞吐/ETA）")


def _apply_feature_args(cfg: Config, a: argparse.Namespace) -> None:
    """把命令行值应用到 Config（仅设置本子命令出现过的参数）。"""
    cfg.coarse_size = _val(a, "coarse_size", cfg.coarse_size)
    cfg.coarse_blur = _val(a, "blur", cfg.coarse_blur)
    cfg.use_hu = not _flag(a, "no_hu")
    cfg.use_fp = not _flag(a, "no_fp")
    cfg.hu_weight = float(_val(a, "hu_weight", cfg.hu_weight))
    cfg.fp_weight = float(_val(a, "fp_weight", cfg.fp_weight))
    cfg.invert_binary = _flag(a, "invert_binary")
    cfg.model = _val(a, "model", cfg.model)
    cfg.device = _val(a, "device", cfg.device)
    cfg.fp16 = not _flag(a, "no_fp16")
    cfg.batch = int(_val(a, "batch", cfg.batch))
    cfg.store_fine = not _flag(a, "no_store_fine")
    cfg.dedup = not _flag(a, "no_dedup")
    cfg.workers = int(_val(a, "workers", cfg.workers))
    cfg.decode_workers = int(_val(a, "decode_workers", cfg.decode_workers))
    cfg.torch_threads = int(_val(a, "torch_threads", cfg.torch_threads))
    cfg.png_decoder = _val(a, "png_decoder", cfg.png_decoder)
    cfg.coarse_k = int(_val(a, "coarse_k", cfg.coarse_k))
    cfg.exclude_self = not _flag(a, "no_exclude_self")
    ext = _val(a, "ext")
    if ext:
        cfg.extensions = tuple(sorted(
            set(cfg.extensions)
            | {e if e.startswith(".") else "." + e for e in ext}))


# ---------------------------------------------------------------------------
# build / add / build-fine（带双阶段实时进度：粗筛 → ResNet）
# ---------------------------------------------------------------------------
def _mk_progress(a: argparse.Namespace, total_phases: int):
    """按 --no-progress 与终端情况创建渲染器；禁用时返回 None。"""
    if _flag(a, "no_progress"):
        return None
    from .progress import CliPhaseProgress
    return CliPhaseProgress(total_phases=total_phases)


def cmd_build(cfg: Config, a: argparse.Namespace) -> int:
    eng = HybridEngine(cfg)
    # 需 ResNet 时融合建库（fused），否则纯粗筛 —— 均为单阶段进度
    phases = 1
    p = _mk_progress(a, phases)
    try:
        n = eng.build(a.prefix, a.img_dir, limit=_val(a, "limit"),
                      force=a.force, progress=p)
    except FileExistsError as e:
        LOGGER.error("%s", e)
        if p:
            p.finish()
        return 2
    finally:
        if p:
            p.finish()
    LOGGER.info("build 完成：%d 张 -> %s.*", n, a.prefix)
    return 0


def cmd_add(cfg: Config, a: argparse.Namespace) -> int:
    eng = HybridEngine(cfg)
    # 需要 ResNet 精排时走“融合单遍解码”；仅粗筛时也是单阶段
    phases = 1
    p = _mk_progress(a, phases)
    try:
        eng.add(a.prefix, a.img_dir, limit=_val(a, "limit"), progress=p)
    finally:
        if p:
            p.finish()
    return 0


def cmd_build_fine(cfg: Config, a: argparse.Namespace) -> int:
    eng = HybridEngine(cfg)
    eng.open(a.prefix)
    p = _mk_progress(a, 1)
    try:
        n = eng.build_fine(a.prefix, progress=p)
    finally:
        if p:
            p.finish()
    LOGGER.info("build-fine 完成：%d 张 -> %s.fine.npz", n, a.prefix)
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def cmd_search(cfg: Config, a: argparse.Namespace) -> int:
    if not os.path.exists(a.query):
        LOGGER.error("查询图不存在: %s", a.query)
        return 2
    eng = HybridEngine(cfg)
    eng.open(a.prefix)
    out = eng.search(a.query, coarse_k=a.coarse_k, top_k=a.top_k)
    _print_outcome(out)
    if out.hits:
        if a.save:
            save_contact_sheet([h.path for h in out.hits], a.save)
    if a.json:
        _dump_json(out, a.json)
    return 0


def _print_outcome(out: Outcome) -> None:
    print(f"\n查询图: {out.query}")
    print(f"库规模: {out.db_size} 张 | 粗筛候选: {out.coarse_kept} 张"
          + ("（已剔除查询图自身）" if out.self_excluded else ""))
    ms = lambda k: f"{out.times.get(k, 0.0) * 1000:.1f} ms"
    if out.coarse_only:
        print(f"耗时: 合计 {ms('total')}（本次未做 ResNet 精排）")
    else:
        print(f"耗时: 粗筛 {ms('粗筛(特征+扫描)')} | "
              f"精排 {ms('精排(查询特征+打分)')} | 合计 {ms('total')}")
    if not out.hits:
        print("（无结果）")
        return
    print(f"{'#':>3} {'ResNet相似度':>12} {'粗筛分':>7} {'d_hu':>7} {'d_fp':>7}  文件")
    for h in out.hits:
        fine = "        n/a" if h.fine_score != h.fine_score \
            else f"{h.fine_score:12.4f}"
        print(f"{h.rank:>3} {fine} {h.coarse_score:7.3f} {h.d_hu:7.4f} "
              f"{h.d_fp:7.4f}  {h.path}")


def _dump_json(out: Outcome, path: str) -> None:
    payload = {
        "query": out.query,
        "db_size": out.db_size,
        "coarse_kept": out.coarse_kept,
        "coarse_only": out.coarse_only,
        "self_excluded": out.self_excluded,
        "times": {k: round(v, 4) for k, v in out.times.items()},
        "results": [
            {"rank": h.rank, "path": h.path, "fine_score": h.fine_score,
             "coarse_score": h.coarse_score, "d_hu": h.d_hu, "d_fp": h.d_fp}
            for h in out.hits
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    LOGGER.info("结果已写: %s", path)


# ---------------------------------------------------------------------------
# eval：批量召回率（配合 make_test_dataset.py 的分组数据）
# ---------------------------------------------------------------------------
def cmd_eval(cfg: Config, a: argparse.Namespace) -> int:
    """
    数据目录约定（由 make_test_dataset.py 生成）：
      <dataset>/db/      图库，文件名带 g<编号> 分组标记
      <dataset>/queries/ 查询集，文件名带 g<编号>，与 db 中同组为“命中”
    指标：
      粗筛窗口命中率：同组图是否进入粗筛 top coarse_k
      最终 Recall@1/5/10：同组图是否进入最终结果对应名次
    """
    from .io_utils import collect_images

    eng = HybridEngine(cfg)
    eng.open(a.prefix)
    db_group = _groups_of(eng.coarse.paths, a.tag_pattern)
    queries = collect_images(a.queries_dir, cfg.extensions, limit=_val(a, "limit"))
    if not queries:
        LOGGER.error("查询目录无图片: %s", a.queries_dir)
        return 2

    ev = 0
    coarse_window_hit = 0
    recalls = {1: 0, 5: 0, 10: 0}
    fine_used = 0
    t_tot = 0.0
    topk = max(recalls)

    for i, q in enumerate(queries, 1):
        m = re.search(a.tag_pattern, os.path.basename(q))
        if not m:
            LOGGER.warning("跳过无法解析分组名的查询: %s", q)
            continue
        g = int(m.group(1))
        t0 = time.time()
        out = eng.search(q, coarse_k=a.coarse_k, top_k=topk)
        t_tot += time.time() - t0
        ev += 1
        if not out.coarse_only:
            fine_used += 1
        # 粗筛窗口命中：候选路径里是否出现同组图
        cand_paths = [c[1] for c in eng.coarse.coarse_search(q, a.coarse_k)]
        if any(db_group.get(p) == g for p in cand_paths):
            coarse_window_hit += 1
        # 最终结果 Recall@k
        for k in recalls:
            if any(h.rank <= k and db_group.get(h.path) == g for h in out.hits):
                recalls[k] += 1
        if i % 20 == 0:
            LOGGER.info("eval 进度 %d/%d", i, len(queries))

    if ev == 0:
        LOGGER.error("没有可评估的查询")
        return 2
    print("\n===== 召回率评估 =====")
    print(f"查询数: {ev}（ResNet 精排可用: {fine_used}）| 图库: {eng.coarse.size} 张")
    print(f"粗筛窗口命中率(top {a.coarse_k}): "
          f"{coarse_window_hit / ev * 100:.1f}%")
    for k in recalls:
        print(f"最终 Recall@{k:>2}: {recalls[k] / ev * 100:.1f}%")
    print(f"平均单次查询: {t_tot / ev * 1000:.1f} ms")
    return 0


def _groups_of(paths: List[str], pattern: str) -> dict:
    out = {}
    for p in paths:
        m = re.search(pattern, os.path.basename(p))
        if m:
            out[p] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# bench：硬件吞吐基准（展示机器实际能跑多快）
# ---------------------------------------------------------------------------
def cmd_bench(cfg: Config, a: argparse.Namespace) -> int:
    from dataclasses import replace

    import numpy as np

    from .engine import HybridEngine
    from .io_utils import collect_images

    paths = collect_images(a.img_dir, cfg.extensions, limit=_val(a, "limit"))
    if not paths:
        LOGGER.error("目录中没有图片: %s", a.img_dir)
        return 2
    print("\n================ 硬件基准 ================")
    print(f"样本: {len(paths)} 张（{a.img_dir}）")
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else "无 CUDA 设备"
        print(f"torch {torch.__version__} | CPU 线程(推理默认) {torch.get_num_threads()} | {gpu}")
    except Exception:  # noqa: BLE001
        print("torch 未安装：精排相关项不可用")

    import tempfile
    tmp = tempfile.mkdtemp(prefix="hyb_bench_")
    try:
        prefix = os.path.join(tmp, "b")
        cfg_b = replace(cfg, store_fine=False)
        eng = HybridEngine(cfg_b)
        t0 = time.time()
        n = eng.build(prefix, paths=paths)
        t_build = time.time() - t0
        eng.open(prefix)  # build 后重新装载，使本实例可检索
        print(f"\n[粗筛索引] {n} 张 耗时 {t_build:.2f}s → "
              f"{n / t_build:.1f} 张/秒（含 MD5+解码+二值化，线程={cfg_b.workers or '自动'}）")

        # 检索延迟（用库内前 5 张做查询，含“剔除自身”）
        qs = paths[:5]
        lat = []
        for q in qs:
            out = eng.search(q, coarse_k=cfg_b.coarse_k, top_k=cfg_b.top_k)
            lat.append(out.times.get("total", 0) * 1000)
        print(f"[检索延迟] 平均 {np.mean(lat):.1f} ms/查询 "
              f"（库 {n} 张，coarse_k={cfg_b.coarse_k}）")

        # ResNet 精排吞吐（GPU 场景=解码流水线与前向重叠后的真实吞吐）
        try:
            sample = paths[: int(_val(a, "fine_sample", 64))]
            extractor = eng._get_extractor()
            t0 = time.time()
            ok, feats = extractor.extract_batch(sample)
            t_f = time.time() - t0
            if ok:
                print(f"[ResNet精排] 设备={extractor.device} 模型={cfg.model} "
                      f"解码线程={extractor.decode_workers} 批={extractor.batch}")
                print(f"            {len(ok)} 张 耗时 {t_f:.2f}s → "
                      f"{len(ok) / t_f:.1f} 张/秒（{feats.shape[1]} 维）")
        except Exception as e:  # noqa: BLE001
            print(f"[ResNet精排] 不可用: {e}")
        print("==========================================")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cmd_stats(cfg: Config, a: argparse.Namespace) -> int:
    eng = HybridEngine(cfg)
    eng.open(a.prefix)
    s = eng.stats()
    print("\n===== 索引统计 =====")
    print(f"前缀   : {s['prefix']}.*")
    print(f"图片数 : {s['n']} 张")
    print(f"粗筛特征: Hu矩={'开' if s['features']['hu'] else '关'} | "
          f"二值指纹={'开' if s['features']['fingerprint'] else '关'} "
          f"({s['features']['fp_bytes_per_image']} B/张)")
    f = s["fine"]
    if f["exists"]:
        print(f"精排   : 有全库索引（{f['rows']} 张 × {f['dim']} 维）")
    else:
        print("精排   : 无全库索引（查询时对候选实时抽特征）")
    print(f"meta   : {s['files']['meta']}")
    print(f"coarse : {s['files']['coarse_size']}")
    print(f"fine   : {s['files']['fine_size']}")
    return 0


# ---------------------------------------------------------------------------
# ingest：跨进程交接（img_server 按钮 → request JSON → 增量建库）
# ---------------------------------------------------------------------------
def cmd_ingest(cfg: Config, a: argparse.Namespace) -> int:
    from .handoff import process_request_file

    req_path = os.path.abspath(a.request)
    if not os.path.isfile(req_path):
        LOGGER.error("交接文件不存在: %s", req_path)
        return 2

    def prog(done, total, phase):
        LOGGER.info("ingest %s：%d/%d", phase, done, total)

    print(f"[ingest] 处理交接: {req_path}", flush=True)
    result = process_request_file(req_path, progress=prog)
    print(f"[ingest] ok={result.get('ok')} | 新增 {result.get('total_added', 0)} 张"
          f" | 耗时 {result.get('total_secs', 0)}s | 前缀 {result.get('prefix')}",
          flush=True)
    for st in result.get("steps", []):
        flag = "OK " if "error" not in st else "ERR"
        where = st.get("gallery_root") or st.get("root", "")
        locate = ("" if st.get("located")
                  else "（该根自身即图库位置）" if where == st.get("root")
                  else "")
        print(f"    [{flag}] {st.get('root')}  +{st.get('added', 0)} 张 "
              f"({st.get('secs', 0)}s) -> {where}{locate}"
              + (f"  {st['error']}" if "error" in st else ""), flush=True)
    for nt in result.get("notices", []):
        print(f"    [提示] {nt}", flush=True)
    if result.get("errors"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# argparse 组装
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hybrid_search",
        description="二值法粗筛 + ResNet 精排的混合图库检索系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- build
    sp = sub.add_parser("build", help="从零构建索引（粗筛 + 可选精排全库）")
    _add_prefix(sp)
    _add_feature_args(sp)
    sp.add_argument("img_dir", help="图库目录（递归扫描）")
    sp.add_argument("--force", action="store_true", help="索引已存在时覆盖重建")
    sp.add_argument("--limit", type=int, default=None, help="只处理前 N 张（测试）")
    sp.set_defaults(func=cmd_build)

    # ---- add
    sp = sub.add_parser("add", help="增量入库（按路径 + MD5 去重）")
    _add_prefix(sp)
    _add_feature_args(sp)
    sp.add_argument("img_dir", help="新增图片目录")
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_add)

    # ---- build-fine
    sp = sub.add_parser("build-fine", help="为已有粗筛索引补建 ResNet 全库索引")
    _add_prefix(sp)
    _add_feature_args(sp)
    sp.set_defaults(func=cmd_build_fine)

    # ---- search
    sp = sub.add_parser("search", help="以图搜图")
    _add_prefix(sp)
    _add_feature_args(sp)
    sp.add_argument("query", help="查询图片路径")
    sp.add_argument("--top-k", type=int, default=10, help="返回结果数")
    sp.add_argument("--save", default=None, help="把 Top-K 总览图存为 PNG")
    sp.add_argument("--json", default=None, help="把结果存为 JSON")
    sp.set_defaults(func=cmd_search)

    # ---- eval
    sp = sub.add_parser("eval", help="批量召回率评估")
    _add_prefix(sp)
    _add_feature_args(sp)
    sp.add_argument("queries_dir", help="查询图片目录")
    sp.add_argument("--tag-pattern", default=r"g(\d+)",
                    help="从文件名解析分组号的正则（默认 g<数字>）")
    sp.add_argument("--limit", type=int, default=None, help="最多评估 N 个查询")
    sp.set_defaults(func=cmd_eval)

    # ---- ingest（跨进程交接自动增量入库）
    sp = sub.add_parser(
        "ingest", help="处理 img_server 交接 request JSON，自动增量建库")
    sp.add_argument("request", help="交接文件路径（request_*.json / working_*.json）")
    sp.set_defaults(func=cmd_ingest)

    # ---- bench
    sp = sub.add_parser("bench", help="硬件吞吐基准（粗筛/精排/检索延迟实测）")
    _add_feature_args(sp)
    sp.add_argument("img_dir", help="用于基准的图片目录")
    sp.add_argument("--limit", type=int, default=300,
                    help="粗筛基准用前 N 张（默认 300）")
    sp.add_argument("--fine-sample", type=int, default=64,
                    help="精排吞吐抽样张数（默认 64）")
    sp.set_defaults(func=cmd_bench)

    # ---- stats
    sp = sub.add_parser("stats", help="索引统计")
    _add_prefix(sp)
    sp.set_defaults(func=cmd_stats)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    # Windows 终端兜底：强制 UTF-8 输出避免中文乱码
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass

    cfg = Config()
    _apply_feature_args(cfg, args)
    try:
        return int(args.func(cfg, args))
    except KeyboardInterrupt:
        LOGGER.error("已中断")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        LOGGER.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
