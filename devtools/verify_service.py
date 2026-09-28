# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 验证：服务层编排（scan/建库/检索/去重/compact/参数 schema）+ 事件契约
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

"""
验证服务层：`hybrid_search/service.py` 的无 GUI 端到端回归（P0 判据）。

跑的是**真实链路**（make_test_dataset 生成模拟图库 → 服务层扫描/建库/检索/去重），
不对产物做 mock；断言对象是「事件流 + 返回结果 + 索引产物」。

覆盖：
  1. 事件契约：命令带 task_id；事件含 log / progress / phase_boundary(save|done) /
     viz_frame / task_done / task_error；
  2. 扫描：张数与目录一致；
  3. 整图建库（fused）：phase_boundary 以 save→done 收尾、viz_frame 含 coarse+fine；
  4. 检索三模式：full 命中 img_g00000_*（与 README 冒烟同口径）、tiles、hybrid 均有结果；
  5. 去重：完全重复被识别 → 删除（回收站）→ 索引 prune 同步；
  6. compact 幂等（含“npz → 侧车”的真实转换）；
  7. 参数 schema 覆盖 `Config` 全部字段且 JSON 可序列化；等价 CLI 片段可用。

用法: python -E devtools/verify_service.py [--db 200] [--per-group 4] [--keep]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import fields as dc_fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hybrid_search.config import Config  # noqa: E402
from hybrid_search.service import (  # noqa: E402
    EVENT_LOG, EVENT_PROGRESS, EVENT_PHASE_BOUNDARY, EVENT_TASK_DONE,
    EVENT_TASK_ERROR, EVENT_VIZ_FRAME, OP_BUILD, OP_SCAN, OP_SEARCH, OP_TILES,
    PHASE_DONE, PHASE_SAVE, SearchService)

FAILED = []


def check(cond, text: str, detail: str = "") -> bool:
    """断言并打印（不中断：尽量把一个来回的信息都跑出来）。"""
    tag = "✓" if cond else "✗"
    print(f"  {tag} {text}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(text)
    return bool(cond)


class Recorder:
    """事件记录器：`clear()` 只移动观察窗口，`all_*` 看全程。"""

    def __init__(self) -> None:
        self.events = []
        self.view = 0

    def __call__(self, ev: dict) -> None:
        self.events.append(ev)

    def clear(self) -> None:
        self.view = len(self.events)

    def kind(self, kind: str, task_id: str = ""):
        out = [e for e in self.events[self.view:] if e.get("event") == kind]
        return [e for e in out if not task_id or e.get("task_id") == task_id]

    def kind_all(self, kind: str):
        return [e for e in self.events if e.get("event") == kind]

    def boundaries(self, task_id: str):
        return [e.get("phase") for e in self.kind(EVENT_PHASE_BOUNDARY, task_id)]

    def done_ops(self):
        return [e.get("op") for e in self.kind(EVENT_TASK_DONE)]


def _jsonable(v):
    return list(v) if isinstance(v, tuple) else v


def _schema_fields(schema: dict) -> dict:
    out = {}
    for page in schema["pages"]:
        for group in page["groups"]:
            for f in group["fields"]:
                out[f["key"]] = f
    return out


def _tile_keys(prefix: str):
    """瓦片索引 → {(原图路径, 框): (块md5, hu字节, fp字节)}（顺序无关比对用）。"""
    from hybrid_search.store import IndexFiles
    st = IndexFiles(prefix).load_coarse()
    boxes = st.get("boxes")
    if boxes is None:
        return None
    hu, fp = st["hu"], st["fp"]
    keys = {}
    for i, p in enumerate(st["paths"]):
        key = (os.path.normcase(os.path.abspath(p)),
               tuple(int(v) for v in boxes[i]))
        keys[key] = (st["md5s"][i],
                     None if hu is None else bytes(hu[i].tobytes()),
                     None if fp is None else bytes(fp[i].tobytes()))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="服务层无 GUI 端到端回归（P0 判据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--db", type=int, default=200, help="模拟图库图片总数")
    ap.add_argument("--per-group", type=int, default=4, help="每组变体数")
    ap.add_argument("--queries", type=int, default=4, help="查询图数量")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（排查用）")
    a = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work = tempfile.mkdtemp(prefix="svc_verify_")
    data = os.path.join(work, "test_data")
    db_dir = os.path.join(data, "db")
    q_dir = os.path.join(data, "queries")
    prefix = os.path.join(work, "gallery")
    npz_prefix = os.path.join(work, "npz_legacy", "gallery")
    print(f"临时目录: {work}")
    perf_artifacts = []                 # 生成的性能图（结束时清理，不留仓库杂物）
    t_all = time.time()
    try:
        print("\n== 0) 生成模拟图库（make_test_dataset.py，真实链路）==")
        rc = subprocess.run(
            [sys.executable, "-E",
             os.path.join(repo, "make_test_dataset.py"),
             "--out", data, "--db", str(a.db),
             "--per-group", str(a.per_group),
             "--queries", str(a.queries)],
            cwd=repo).returncode
        images = sorted(glob.glob(os.path.join(db_dir, "*.jpg")))
        queries = sorted(glob.glob(os.path.join(q_dir, "*.jpg")))
        check(rc == 0, "生成器退出码 0", f"rc={rc}")
        check(len(images) > 0 and len(queries) > 0,
              "图库/查询集非空（事件契约与检索都要真数据）",
              f"库 {len(images)} 张 / 查询 {len(queries)} 张")
        q = os.path.join(q_dir, "q_g00000_v0.jpg")
        check(os.path.exists(q), "存在按组命名的查询图 q_g00000_v0.jpg")

        cfg = Config()
        cfg.workers = 4
        cfg.decode_workers = 2
        svc = SearchService(cfg=cfg, capture_log=True)
        rec = Recorder()
        svc.subscribe(rec)

        # ---- 1) 参数 schema / 等价 CLI --------------------------------
        print("\n== 1) 参数 schema / 等价 CLI ==")
        schema = svc.get_config_schema()
        json.dumps(schema, ensure_ascii=False)          # 不可序列化会直接抛错
        check(True, "schema JSON 可序列化（Web 前端可直取）")
        fields = _schema_fields(schema)
        cfg_keys = {f.name for f in dc_fields(Config)}
        missing = sorted(cfg_keys - set(fields))
        check(not missing, "schema 覆盖 Config 全部字段",
              f"缺 {missing}" if missing else f"{len(cfg_keys)} 个字段")
        dflt_ok = all(fields[k].get("default") == _jsonable(getattr(Config(), k))
                      for k in cfg_keys)
        check(dflt_ok, "schema default 取自 Config()（默认值只写一次）")
        applied = svc.set_config({"coarse_k": 123, "use_fp": False})
        check(applied == {"coarse_k": 123, "use_fp": False}
              and svc.get_config()["coarse_k"] == 123,
              "set_config 校验并生效", str(applied))
        try:
            svc.set_config({"coarse_size": 4096})
            check(False, "越界值应被拒绝")
        except ValueError as e:
            check(True, "越界值被拒绝", str(e))
        svc.set_config({"coarse_k": Config().coarse_k,
                        "use_fp": Config().use_fp})
        cli = svc.cli_command("build", positionals=[db_dir])
        check(cli.startswith("python main.py build")
              and svc.cli_hint("coarse_blur", 7) == "--blur 7",
              "等价 CLI 提示可生成", cli)

        # ---- 2) 扫描 --------------------------------------------------
        print("\n== 2) 扫描（service.scan）==")
        rec.clear()
        sc = svc.scan("t-scan", db_dir, recursive=True, verify=False)
        check(bool(sc) and sc["count"] == len(images),
              "扫描张数与目录一致", f"{sc and sc['count']} vs {len(images)}")
        check(rec.done_ops() == [OP_SCAN] and bool(rec.kind(EVENT_TASK_DONE)),
              "任务完成事件带 task_id 与 op=scan", str(rec.done_ops()))

        # ---- 3) 整图建库（fused：粗筛 + ResNet 单遍）------------------
        print("\n== 3) 整图建库（fused）==")
        rec.clear()
        t0 = time.time()
        r = svc.build_index("t-build", prefix, img_dir=db_dir, force=True,
                            title="验证·全量建库")
        dt = time.time() - t0
        check(bool(r) and r["n"] == sc["count"] and r["prefix"] == prefix,
              "入库张数 = 扫描张数", f"{r and r['n']} 张 / {dt:.1f}s")
        bounds = rec.boundaries("t-build")
        check(bounds[-2:] == [PHASE_SAVE, PHASE_DONE],
              "阶段边界以 save→done 收尾（铁律）", "→".join(bounds))
        vkinds = sorted({e.get("kind") for e in rec.kind(EVENT_VIZ_FRAME, "t-build")})
        check(vkinds == ["coarse", "fine"],
              "viz_frame 同时透出 coarse / fine 帧", str(vkinds))
        check(bool(rec.kind(EVENT_LOG)) and bool(rec.kind(EVENT_PROGRESS)),
              "log / progress 事件均有透出",
              f"log {len(rec.kind(EVENT_LOG))} / progress {len(rec.kind(EVENT_PROGRESS))}")
        check(rec.done_ops() == [OP_BUILD], "task_done op=build", str(rec.done_ops()))
        check(not rec.kind(EVENT_TASK_ERROR), "本步无 task_error")

        # ---- 4) 检索：full / tiles / hybrid ---------------------------
        print("\n== 4) 检索三模式 ==")
        rec.clear()
        s_full = svc.search("t-full", q, "full", prefix=prefix)
        top1 = (os.path.basename(s_full["hits"][0][1])
                if s_full and s_full["hits"] else "-")
        check(bool(s_full and s_full["hits"]) and top1.startswith("img_g00000"),
              "full 命中 img_g00000_*（README 冒烟口径）",
              f"top1={top1}（{len(s_full['hits']) if s_full else 0} 条）")
        check(bool(s_full and "total" in s_full["times"]),
              "返回 times 含 total", str(sorted((s_full or {}).get("times", {}))))

        tp = svc.tiles_prefix(prefix)
        rec.clear()
        t0 = time.time()
        rt = svc.tiles_index("t-tiles", tp, paths=list(images))
        check(bool(rt) and rt["n"] > 0, "瓦片索引构建",
              f"{rt and rt['n']} 块 / {time.time() - t0:.1f}s")
        check(rec.boundaries("t-tiles")[-2:] == [PHASE_SAVE, PHASE_DONE],
              "瓦片建库同样以 save→done 收尾")
        check(rec.done_ops() == [OP_TILES], "task_done op=tiles", str(rec.done_ops()))
        # 同一批图、**反序**再建一次瓦片索引：落盘顺序 = 线程完成顺序，
        # 因此比对必须按「原图路径 + 框」建键（顺序无关），逐位比对块 md5/指纹
        prefix2 = os.path.join(work, "again", "gallery")
        tp2 = svc.tiles_prefix(prefix2)
        rt2 = svc.tiles_index("t-tiles2", tp2, paths=list(reversed(images)))
        k1, k2 = _tile_keys(tp), _tile_keys(tp2)
        check(bool(k1 and k2) and k1 == k2 and len(k1) == rt["n"] == rt2["n"],
              "瓦片索引两次构建顺序无关且逐位一致",
              f"键 {len(k1 or {})} vs {len(k2 or {})} / 块 {rt['n']} vs {rt2['n']}")
        rec.clear()
        s_tiles = svc.search("t-tiles-q", q, "tiles", prefix=prefix)
        check(bool(s_tiles and s_tiles["hits"]), "瓦片检索有结果",
              f"{len(s_tiles['hits']) if s_tiles else 0} 条")
        s_hyb = svc.search("t-hybrid-q", q, "hybrid", prefix=prefix)
        check(bool(s_hyb and s_hyb["hits"]), "混合检索有结果",
              f"{len(s_hyb['hits']) if s_hyb else 0} 条")
        check(rec.done_ops()[:2] == [OP_SEARCH, OP_SEARCH],
              "检索任务均以 task_done(op=search) 收尾", str(rec.done_ops()))

        # 4d) 性能画像（可选导出）：perf_report 事件 + 报告落盘（搜图不弹窗）
        print("\n== 4d) 性能图导出（perf=True）==")
        rec.clear()
        s_perf = svc.search("t-perf", q, "full", prefix=prefix, perf=True)
        rep_path = (s_perf or {}).get("perf", "")
        evs = rec.kind("perf_report", "t-perf")
        check(bool(evs) and rep_path.endswith(".html") and os.path.exists(rep_path),
              "perf=True 落盘 HTML 并发 perf_report 事件", rep_path or "(未生成)")
        check(bool(evs) and evs[0].get("modal") is False,
              "搜图性能图不弹窗（modal=False）", str(evs[:1]))
        if rep_path:
            js = json.load(open(rep_path[:-5] + ".json", encoding="utf-8"))
            html = open(rep_path, encoding="utf-8").read()
            check(len(js.get("rows") or []) > 0 and "阶段耗时" in html,
                  "报告含采样行与阶段表",
                  f"采样 {len(js.get('rows') or [])} 行 / 阶段 "
                  f"{[p['name'] for p in js.get('phases', [])][:4]}")
            perf_artifacts.extend([rep_path, rep_path[:-5] + ".json"])

        # ---- 5) 去重：扫描 → 删除（回收站）→ 索引同步 -----------------
        print("\n== 5) 去重扫描 / 删除 / 索引同步 ==")
        dup = os.path.join(db_dir, "dup_copy_of_first.jpg")
        shutil.copy2(images[0], dup)
        rec.clear()
        job = svc.dedup_scan("t-dedup", list(images) + [dup], threshold=0.02,
                             prefix=prefix)
        member = {}
        for g in (job.groups if job else []):
            for m in g.members:
                member[os.path.normcase(m.path)] = (g, m)
        md5_dup = member.get(os.path.normcase(dup), (None, None))[1]
        md5_src = member.get(os.path.normcase(images[0]), (None, None))[1]
        same_group = bool(md5_dup and md5_src
                          and member[os.path.normcase(dup)][0]
                          is member[os.path.normcase(images[0])][0])
        check(bool(md5_dup and md5_src and same_group
                   and md5_dup.md5 == md5_src.md5 and md5_dup.md5),
              "字节相同的副本与原件同组且 MD5 一致",
              f"组 {job and len(job.groups)} / 复用索引 {job and job.indexed_used}"
              f" / md5 {bool(md5_dup and md5_dup.md5)}")
        rows0 = len(svc.indexed_paths(prefix))
        out = svc.dedup_delete("t-dedup-del", [images[0]], prefix=prefix,
                               sync=True)
        check(bool(out) and out["removed"] == [images[0]] and not out["failed"],
              "删除成功（移入回收站）", f"removed={len(out['removed']) if out else 0}")
        check(not os.path.exists(images[0]), "原文件已移出图库目录")
        rows1 = len(svc.indexed_paths(prefix))
        check(rows1 == rows0 - 1, "索引同步剔除 1 行", f"{rows0} → {rows1}")
        check(any(res.get("removed") == 1 for res in (out or {}).get("prune", [])),
              "prune 报告剔除 1 条", str((out or {}).get("prune")))
        out2 = svc.dedup_delete("t-dedup-del2", [dup], prefix=prefix, sync=True)
        check(bool(out2) and out2["removed"] == [dup],
              "未入库的副本也能删除", f"prune={len((out2 or {}).get('prune', []))}")

        # ---- 6) compact：真实转换 + 幂等 -----------------------------
        print("\n== 6) compact（npz → 侧车，幂等）==")
        rec.clear()
        npz_cfg = Config()
        npz_cfg.fast_load = False       # 故意写 npz，验证真实转换路径
        npz_cfg.workers = 4
        svc.build_index("t-build-npz", npz_prefix, paths=list(images[1:9]),
                        force=True, cfg=npz_cfg, title="验证·npz 索引")
        c1 = svc.compact("t-compact", [npz_prefix], cfg=npz_cfg, perf=True)
        c2 = svc.compact("t-compact2", [npz_prefix], cfg=npz_cfg)
        conv = (c1 or {}).get("prefixes", [])
        check(bool(conv) and all(not x.get("already") for x in conv)
              and conv[0]["n"] == 8,
              "npz 索引被真正转换成侧车", str([{k: x.get(k) for k in ("already", "n")}
                                               for x in conv]))
        cperf = (c1 or {}).get("perf", "")
        cevs = [e for e in rec.kind_all("perf_report") if e.get("stage") == "index"]
        check(bool(cperf) and bool(cevs) and cevs[-1].get("modal") is True,
              "索引阶段性能图落盘并标记弹窗（modal=True）", cperf or "(未生成)")
        for p in (cperf, cperf[:-5] + ".json" if cperf else ""):
            if p and os.path.exists(p):
                perf_artifacts.append(p)
        again = (c2 or {}).get("prefixes", [])
        check(bool(again) and all(x.get("already") for x in again),
              "再次 compact → already=True（幂等）", str(again))
        check(str(svc.index_status(npz_prefix)["storage"]) == "sidecar",
              "meta 存储格式已标记为 sidecar")

        # ---- 7) 状态 / 统计 / 引擎缓存 --------------------------------
        print("\n== 7) 状态 / 统计 / 引擎缓存与释放 ==")
        status = svc.index_status(prefix)
        check(status["full_exists"] and status["tiles_exists"]
              and status["n"] == rows1 and status["tiles_n"] == rows1
              and status["tiles_n"] == rt["n"] - 1,
              "index_status 与索引实际一致（整图/瓦片同步剔除）",
              f"整图 {status['n']} / 瓦片 {status['tiles_n']}（建库时 {rt['n']}）")
        st = svc.stats("t-stats", prefix)
        check(bool(st) and st["n"] == rows1 and st["fine"]["exists"],
              "stats 可用", f"n={st and st['n']} fine={st and st['fine']['rows']}")
        _eng, cached1 = svc.engine_for(svc.cfg, prefix)
        _eng2, cached2 = svc.engine_for(svc.cfg, prefix)
        check(cached2, "engine_for 二次调用命中缓存", f"首次 {cached1} / 二次 {cached2}")
        freed = svc.release_engines("验证·释放")
        check(not svc._eng_cache, "release_engines 清空引擎缓存",
              f"RSS 回落 {freed:.0f} MB")

        # ---- 8) 全程无 task_error -------------------------------------
        errs = rec.kind_all(EVENT_TASK_ERROR)
        check(not errs, "全程无 task_error",
              str([(e.get("op"), e.get("error")) for e in errs]))
        svc.close()
    finally:
        if a.keep:
            print(f"\n（--keep：保留 {work}）")
        else:
            shutil.rmtree(work, ignore_errors=True)
        n_rm = 0
        for p in perf_artifacts:            # 性能图默认写在仓库 perf_reports/：用完删
            try:
                if os.path.exists(p):
                    os.remove(p)
                    n_rm += 1
            except OSError:
                pass
        if n_rm:
            print(f"已清理本轮生成的性能图 {n_rm} 个文件（perf_reports/）")
    print(f"\n总耗时 {time.time() - t_all:.1f}s")
    if FAILED:
        print(f"结果: 存在失败项 {len(FAILED)} 项 -> {FAILED}")
        return 1
    print("结果: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
