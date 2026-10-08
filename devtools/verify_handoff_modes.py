# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ImageSearchTool · 图库检索管理器 — 验证：交接协议 schema v2（modes：整图 / 子图）
# Copyright (C) 2026 zccored
#
# 本程序是自由软件：你可以再发布和/或修改它，但必须遵守 GNU Affero 通用公共
# 许可证 v3.0（AGPL-3.0-only）的条款；本程序不提供任何担保。完整条款见根目录 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See the LICENSE file for terms.
# ---------------------------------------------------------------------------

r"""验证交接协议 v2 的 modes（整图 / 子图选做或都做）：

  1) 老请求（schema v1、无 modes）→ 按“两个都做”，且 schema v1 仍被接受；
  2) modes 归一化：缺省/空数组/别名/去重/固定顺序（full 在前 tiles 在后）/非法值报错；
  3) modes=["full","tiles"]：整图索引 gallery.* 与子图索引 gallery_tiles.* 都建出来，
     steps[].stages 顺序 = full→tiles，进度阶段里能看到 fused 与 tiles；
  4) modes=["tiles"]：只建子图索引，不建整图索引；
  5) modes=["full"]：只建整图索引，不建子图索引；
  6) 增量：新图入库 +added、瓦片 +tiles_added，再触发一次（同批）为 0（幂等）；
  7) process_request_file 的文件流转：request_ → working_ → result_，working_ 收尾删除；
  8) CLI：python main.py ingest <request> --modes tiles 能覆盖请求里的方案；
  9) 图库根自动定位：locate_gallery_root 的纯函数行为 + 子目录新图并入宿主索引的端到端；
  10) 服务层 handoff（GUI 走的路径）：事件流里阶段只有 fused/tiles、无 phase_boundary、
      task_done op=handoff、日志含「交接方案」；
  11) 过渡进程 handoff_launcher.py 的 cli 模式端到端（img_server 实际调用的入口）：
      rc=0、打印「[launcher] ingest … 方案 full+tiles 新增 N 张 / M 瓦片」、
      result_ 落盘 ok=True、working_ 与 request_ 收尾后不残留。
  12) 过渡进程超时分支：expect_exit 里的进程迟迟不退 → rc=1、错误 result 写进
      **request 所在目录**（img_server 的 handoff\，不是本程序默认目录）、
      request_ 原样保留、未开工、未转 working_。

注：本机 `F:\.gallery_index` 存在一套 2026-09 用 cv2 解码器建的旧索引，而系统 TMP
    就在 `F:\Revit`；若请求里 `prefix` 留空，locate_gallery_root 会沿祖先链上溯到
    `F:\` 命中那套索引，导致「参数与建库参数不一致」而全线失败。故下面凡是要真实
    建库的用例都用 `prefix=<图库根>/.gallery_index/gallery` 把位置钉死；自动定位单独
    用纯函数用例 + 自建宿主索引的端到端用例覆盖。

用法: python devtools/verify_handoff_modes.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
# 工作目录：默认放在「索引器上级目录」（本机 D:\code\新的代码\全栈图库管理器 v3.2bata）
# 而不是系统 TMP —— 本机 TMP=F:\Revit，而 F:\ 根上有一套旧索引，会污染定位用例。
WORK_BASE = os.environ.get("HANDOFF_VERIFY_BASE") or os.path.normpath(
    os.path.join(BASE, "..", "_handoff_verify"))
os.makedirs(WORK_BASE, exist_ok=True)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hybrid_search import handoff as H  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> bool:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}" + (f"  [{extra}]" if extra else ""))
    else:
        FAIL += 1
        print(f"  ✗ {name}" + (f"  [{extra}]" if extra else ""))
    return bool(cond)


def make_images(d: str, n: int, seed: int, side=(1200, 900)) -> None:
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(seed)
    w, h = side
    for i in range(n):
        arr = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(d, f"img_{seed}_{i}.jpg"), quality=90)


def default_prefix(root: str) -> str:
    """图库根无既有索引时，run_ingest 会用的默认前缀。"""
    return os.path.join(root, ".gallery_index", "gallery")


def write_request(d: str, req_id: str, root: str, modes=None, schema=2,
                  prefix: str = "", expect_exit=None, open_mode: str = "cli") -> str:
    os.makedirs(d, exist_ok=True)
    req = {"schema": schema, "kind": "download_batch_complete",
           "request_id": req_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "source": "verify_handoff_modes", "roots": [{"path": root, "note": ""}],
           "prefix": prefix, "open_mode": open_mode,
           "expect_exit": expect_exit if expect_exit is not None
           else {"pids": [], "names": []},
           "note": ""}
    if modes is not None:
        req["modes"] = modes
    fp = os.path.join(d, f"request_{req_id}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False, indent=2)
    return fp


def run(fp: str, d: str, modes=None):
    """执行一次交接，并记录进度阶段名。返回 (result, phases)。"""
    seen = []

    def cb(done, total, phase="?"):
        if not seen or seen[-1] != phase:
            seen.append(phase)
    return H.process_request_file(fp, progress=cb, handoff_dir=d, modes=modes), seen


work = tempfile.mkdtemp(prefix="handoff_v2_", dir=WORK_BASE)
print(f"== 交接协议 v2（modes）验证 ==  工作目录 {work}")
try:
    # 前提：测试目录的祖先链上不能有既有索引，否则 locate_gallery_root 会命中它，
    #       用例会悄悄往别人的索引里增量（本机踩过：F:\.gallery_index）。
    _stray = H.locate_gallery_root(work)
    if not check("测试目录祖先链上无既有索引（避免污染）", _stray is None, str(_stray)):
        print("   ↑ 请换 HANDOFF_VERIFY_BASE 指到干净盘，或用 prefix 钉死位置")

    # ---------------------------------------------------------------- 1) 归一化
    print("\n== 1) modes 归一化 / schema 兼容 ==")
    check("缺省（None）→ 两个都做", H.normalize_modes(None) == ["full", "tiles"],
          str(H.normalize_modes(None)))
    check("空数组 → 两个都做", H.normalize_modes([]) == ["full", "tiles"],
          str(H.normalize_modes([])))
    check("单值字符串 → ['tiles']", H.normalize_modes("tiles") == ["tiles"])
    check("别名 + 去重 + 固定顺序",
          H.normalize_modes(["子图", "整图", "full", "tiles"]) == ["full", "tiles"],
          str(H.normalize_modes(["子图", "整图", "full", "tiles"])))
    check("标签渲染", H.mode_labels(["tiles", "full"]) == "整图增量 + 子图(瓦片)增量",
          H.mode_labels(["tiles", "full"]))
    try:
        H.normalize_modes(["bogus"])
        check("非法取值报错", False)
    except ValueError as e:
        check("非法取值报错", True, str(e))
    v1 = {"schema": 1, "kind": "download_batch_complete",
          "roots": [{"path": work}]}
    try:
        out = H.validate(v1)
        check("schema v1 仍可读，且补出默认 modes",
              out.get("modes") == ["full", "tiles"], str(out.get("modes")))
    except Exception as e:                                    # noqa: BLE001
        check("schema v1 仍可读，且补出默认 modes", False, repr(e))
    try:
        H.validate({"schema": 9, "kind": "download_batch_complete",
                    "roots": [{"path": work}]})
        check("schema 9 被拒绝", False)
    except ValueError as e:
        check("schema 9 被拒绝", True, str(e))

    # --------------------------------------------- 1.5) 图库根自动定位（纯函数）
    print("\n== 1.5) locate_gallery_root 定位规则（不碰引擎）==")
    loc = os.path.join(work, "loc")
    host = os.path.join(loc, "host")
    os.makedirs(os.path.join(host, ".gallery_index"), exist_ok=True)
    open(os.path.join(host, ".gallery_index", "gallery.meta.json"), "w",
         encoding="utf-8").write("{}")
    sub = os.path.join(host, "sub", "deeper")
    os.makedirs(sub, exist_ok=True)
    got = H.locate_gallery_root(sub)
    check("子目录沿祖先链找到图库根",
          bool(got) and os.path.normcase(got["root"]) == os.path.normcase(host),
          str(got))
    check("定位出的 prefix = <根>/.gallery_index/gallery",
          bool(got) and os.path.normcase(got["prefix"]) ==
          os.path.normcase(os.path.join(host, ".gallery_index", "gallery")),
          str(got and got.get("prefix")))
    check("无索引的目录返回 None",
          H.locate_gallery_root(os.path.join(loc, "nothing")) is None,
          str(H.locate_gallery_root(os.path.join(loc, "nothing"))))

    # ------------------------------------------------- 2) 两个都做（首次构建）
    print("\n== 2) modes=[full,tiles] 首次构建 ==")
    gA = os.path.join(work, "A_gallery")
    make_images(gA, 6, seed=11)
    dA = os.path.join(work, "A_handoff")
    fpA = write_request(dA, "batch_A1", gA, modes=["full", "tiles"],
                        prefix=default_prefix(gA))
    rA, phA = run(fpA, dA)
    idxA = os.path.join(gA, ".gallery_index")
    wholeA = os.path.join(idxA, "gallery.meta.json")
    tilesA = os.path.join(idxA, "gallery_tiles.meta.json")
    check("ok=True", rA.get("ok") is True, json.dumps(rA.get("errors"), ensure_ascii=False))
    check("result.modes 回显", rA.get("modes") == ["full", "tiles"], str(rA.get("modes")))
    check("显式 prefix 时不做祖先定位（located=False）",
          (rA.get("steps") or [{}])[0].get("located") is False,
          str((rA.get("steps") or [{}])[0].get("located")))
    check("整图索引已建", os.path.isfile(wholeA), wholeA)
    check("子图索引已建", os.path.isfile(tilesA), tilesA)
    check("整图新增 6 张", rA.get("total_added") == 6, str(rA.get("total_added")))
    check("子图新增 > 0 块", (rA.get("total_tiles_added") or 0) > 0,
          str(rA.get("total_tiles_added")))
    st = (rA.get("steps") or [{}])[0]
    check("stages 顺序 = full→tiles",
          [s.get("mode") for s in st.get("stages", [])] == ["full", "tiles"],
          str([s.get("mode") for s in st.get("stages", [])]))
    check("stages 里整图有 added", st.get("stages", [{}])[0].get("added") == 6)
    check("tiles_prefix 指向 gallery_tiles",
          str(st.get("tiles_prefix", "")).endswith("gallery_tiles"),
          str(st.get("tiles_prefix")))
    check("进度阶段含 fused 与 tiles",
          "fused" in phA and "tiles" in phA, "phase 序列: " + "→".join(map(str, phA)))
    check("进度阶段不含阶段边界 done（不早退）", "done" not in phA and "save" not in phA,
          "phase 序列: " + "→".join(map(str, phA)))
    check("request_ 已消费（转 working_/result_）", not os.path.isfile(fpA))
    check("result 文件已写", os.path.isfile(os.path.join(dA, "result_batch_A1.json")))
    check("working_ 已清理", H.working_files(dA) == [], str(H.working_files(dA)))

    # ------------------------------------------------------- 3) 两个都做（增量）
    print("\n== 3) modes=[full,tiles] 增量（2 张新图）==")
    make_images(gA, 2, seed=22)
    fpA2 = write_request(dA, "batch_A2", gA, modes=["full", "tiles"],
                         prefix=default_prefix(gA))
    rA2, _ = run(fpA2, dA)
    check("整图只新增 2 张", rA2.get("total_added") == 2, str(rA2.get("total_added")))
    check("子图新增 > 0 块", (rA2.get("total_tiles_added") or 0) > 0,
          str(rA2.get("total_tiles_added")))
    st2 = (rA2.get("steps") or [{}])[0]
    check("整图阶段为「增量」",
          st2.get("stages", [{}])[0].get("build_mode") == "增量",
          str(st2.get("stages", [{}])[0].get("build_mode")))
    check("索引总数 = 8", st2.get("total_in_index") == 8, str(st2.get("total_in_index")))

    print("\n== 4) 同批重复触发（幂等）==")
    fpA3 = write_request(dA, "batch_A3", gA, modes=["full", "tiles"],
                         prefix=default_prefix(gA))
    rA3, _ = run(fpA3, dA)
    check("重复触发整图新增 0", rA3.get("total_added") == 0, str(rA3.get("total_added")))
    check("重复触发子图新增 0", (rA3.get("total_tiles_added") or 0) == 0,
          str(rA3.get("total_tiles_added")))
    check("ok=True", rA3.get("ok") is True)

    # ------------------------------------------------------- 5) 只做子图 / 只做整图
    print("\n== 5) modes=[tiles] 只做子图 ==")
    gB = os.path.join(work, "B_gallery")
    make_images(gB, 3, seed=33)
    dB = os.path.join(work, "B_handoff")
    rB, phB = run(write_request(dB, "batch_B1", gB, modes=["tiles"],
                                prefix=default_prefix(gB)), dB)
    idxB = os.path.join(gB, ".gallery_index")
    check("ok=True", rB.get("ok") is True, json.dumps(rB.get("errors"), ensure_ascii=False))
    check("子图索引已建", os.path.isfile(os.path.join(idxB, "gallery_tiles.meta.json")))
    check("整图索引未建", not os.path.isfile(os.path.join(idxB, "gallery.meta.json")))
    check("整图新增为 0", rB.get("total_added") == 0, str(rB.get("total_added")))
    check("stages 只有 tiles",
          [s.get("mode") for s in rB["steps"][0]["stages"]] == ["tiles"])
    check("进度阶段只出现 tiles", set(phB) <= {"tiles"}, "phase: " + "→".join(map(str, phB)))

    print("\n== 6) modes=[full] 只做整图 ==")
    gC = os.path.join(work, "C_gallery")
    make_images(gC, 3, seed=44)
    dC = os.path.join(work, "C_handoff")
    rC, _ = run(write_request(dC, "batch_C1", gC, modes=["full"],
                              prefix=default_prefix(gC)), dC)
    idxC = os.path.join(gC, ".gallery_index")
    check("ok=True", rC.get("ok") is True, json.dumps(rC.get("errors"), ensure_ascii=False))
    check("整图索引已建", os.path.isfile(os.path.join(idxC, "gallery.meta.json")))
    check("子图索引未建", not os.path.isfile(os.path.join(idxC, "gallery_tiles.meta.json")))
    check("子图新增为 0", rC.get("total_tiles_added") == 0)

    # --------------------------------------------------------- 7) schema v1 实跑
    print("\n== 7) schema v1 老请求实跑（等于两个都做）==")
    gD = os.path.join(work, "D_gallery")
    make_images(gD, 3, seed=55)
    dD = os.path.join(work, "D_handoff")
    fpD = write_request(dD, "batch_D1", gD, modes=None, schema=1,
                        prefix=default_prefix(gD))
    rD, _ = run(fpD, dD)
    idxD = os.path.join(gD, ".gallery_index")
    check("ok=True", rD.get("ok") is True, json.dumps(rD.get("errors"), ensure_ascii=False))
    check("v1 请求按两个都做", rD.get("modes") == ["full", "tiles"], str(rD.get("modes")))
    check("整图索引已建", os.path.isfile(os.path.join(idxD, "gallery.meta.json")))
    check("子图索引已建", os.path.isfile(os.path.join(idxD, "gallery_tiles.meta.json")))

    # ------------------------------------------------------------- 8) CLI --modes
    print("\n== 8) CLI: main.py ingest --modes tiles（覆盖请求方案）==")
    gE = os.path.join(work, "E_gallery")
    make_images(gE, 3, seed=66)
    dE = os.path.join(work, "E_handoff")
    fpE = write_request(dE, "batch_E1", gE, modes=["full", "tiles"],
                        prefix=default_prefix(gE))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.time()
    cp = subprocess.run([sys.executable, "-X", "utf8", os.path.join(BASE, "main.py"),
                         "ingest", fpE, "--modes", "tiles"],
                        cwd=BASE, env=env, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    out = (cp.stdout or "") + (cp.stderr or "")
    idxE = os.path.join(gE, ".gallery_index")
    check("CLI 返回码 0", cp.returncode == 0, f"rc={cp.returncode}")
    check("CLI 子图索引已建", os.path.isfile(os.path.join(idxE, "gallery_tiles.meta.json")))
    check("CLI 只做子图（整图索引未建）",
          not os.path.isfile(os.path.join(idxE, "gallery.meta.json")))
    check("CLI 输出含 [ingest] 与子图新增", "[ingest]" in out and "子图新增" in out,
          f"{time.time() - t0:.1f}s")
    resE = json.load(open(os.path.join(dE, "result_batch_E1.json"), encoding="utf-8"))
    check("CLI 结果 modes=['tiles']", resE.get("modes") == ["tiles"], str(resE.get("modes")))

    # ------------------------------------- 9) 图库根自动定位（端到端，子目录并入宿主）
    print("\n== 9) 子目录新图并入宿主索引（prefix 留空 → 自动定位）==")
    host = os.path.join(work, "H_host")
    make_images(host, 3, seed=77)
    dH1 = os.path.join(work, "H_handoff1")
    rH1, _ = run(write_request(dH1, "batch_H1", host, modes=["full", "tiles"],
                               prefix=default_prefix(host)), dH1)
    idxH = os.path.join(host, ".gallery_index")
    check("宿主整图索引先建好", os.path.isfile(os.path.join(idxH, "gallery.meta.json")),
          json.dumps(rH1.get("errors"), ensure_ascii=False))
    check("宿主子图索引先建好", os.path.isfile(os.path.join(idxH, "gallery_tiles.meta.json")))

    subH = os.path.join(host, "sub")
    make_images(subH, 2, seed=88)
    dH2 = os.path.join(work, "H_handoff2")
    rH2, _ = run(write_request(dH2, "batch_H2", subH, modes=["full", "tiles"],
                               prefix=""), dH2)
    stH = (rH2.get("steps") or [{}])[0]
    check("请求根是子目录时 located=True", stH.get("located") is True, str(stH.get("located")))
    check("并入宿主索引（prefix 在 H_host 下）",
          os.path.normcase(str(stH.get("prefix", ""))).startswith(os.path.normcase(idxH)),
          str(stH.get("prefix")))
    check("gallery_root = 宿主根",
          os.path.normcase(str(stH.get("gallery_root", ""))) == os.path.normcase(host),
          str(stH.get("gallery_root")))
    check("只新增子目录里的 2 张", rH2.get("total_added") == 2, str(rH2.get("total_added")))
    check("子图也有新增", (rH2.get("total_tiles_added") or 0) > 0,
          str(rH2.get("total_tiles_added")))
    check("子目录里没有另建索引",
          not os.path.isdir(os.path.join(subH, ".gallery_index")),
          os.path.join(subH, ".gallery_index"))
    check("宿主索引总数 = 5", stH.get("total_in_index") == 5, str(stH.get("total_in_index")))
    # ----------------------- 10) 服务层 handoff（GUI「自动交接」走的正是这条路）
    print("\n== 10) 服务层 handoff（事件流 + 阶段归一）==")
    from hybrid_search.config import Config
    from hybrid_search.service import (EVENT_LOG, EVENT_PROGRESS, EVENT_TASK_DONE,
                                       OP_HANDOFF, SearchService)
    gF = os.path.join(work, "F_gallery")
    make_images(gF, 3, seed=99)
    dF = os.path.join(work, "F_handoff")
    fpF = write_request(dF, "batch_F1", gF, modes=["full", "tiles"],
                        prefix=default_prefix(gF))
    cfgF = Config()
    cfgF.workers = 2
    cfgF.decode_workers = 1
    svcF = SearchService(cfg=cfgF, capture_log=True)
    evs = []
    svcF.subscribe(evs.append)
    rF = svcF.handoff("t-handoff", fpF)
    phases = sorted({str(e.get("phase")) for e in evs
                     if e.get("event") == EVENT_PROGRESS})
    bounds = [e.get("phase") for e in evs if e.get("event") == "phase_boundary"]
    ops = [e.get("op") for e in evs if e.get("event") == EVENT_TASK_DONE]
    texts = " ".join(str(e.get("text", "")) for e in evs if e.get("event") == EVENT_LOG)
    check("服务层交接 ok=True", bool(rF) and rF.get("ok") is True,
          json.dumps((rF or {}).get("errors"), ensure_ascii=False))
    check("整图 3 张 + 子图若干块",
          (rF or {}).get("total_added") == 3
          and (rF or {}).get("total_tiles_added", 0) > 0,
          f"{(rF or {}).get('total_added')} 张 / {(rF or {}).get('total_tiles_added')} 块")
    check("进度阶段只有 fused / tiles（save/done 已归一）", phases == ["fused", "tiles"],
          str(phases))
    check("交接路径不发 phase_boundary（避免整图做完就报完成）", bounds == [], str(bounds))
    check("task_done op=handoff", ops == [OP_HANDOFF], str(ops))
    check("日志含「交接方案」与两种方案标签", "交接方案" in texts and "子图" in texts)

    # ------------- 11) 过渡进程 cli 模式（img_server 写的 request 就是交给它跑的）
    print("\n== 11) handoff_launcher.py cli 模式端到端 ==")
    gG = os.path.join(work, "G_gallery")
    make_images(gG, 2, seed=111)
    dG = os.path.join(work, "G_handoff")
    fpG = write_request(dG, "batch_G1", gG, modes=["full", "tiles"],
                        prefix=default_prefix(gG))
    launcher = os.path.join(BASE, "handoff_launcher.py")
    tG = time.time()
    cpG = subprocess.run([sys.executable, "-X", "utf8", launcher, fpG, "--wait", "1"],
                         cwd=BASE, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=600)
    outG = (cpG.stdout or "") + (cpG.stderr or "")
    check("launcher 返回码 0", cpG.returncode == 0,
          f"rc={cpG.returncode} {outG.strip()[-300:]}")
    check("launcher 打印 ingest 摘要（含方案与瓦片数）",
          "[launcher] ingest" in outG and "方案 full+tiles" in outG and "瓦片" in outG,
          outG.strip().splitlines()[-1] if outG.strip() else "")
    resG = os.path.join(dG, "result_batch_G1.json")
    check("launcher 写出 result_", os.path.isfile(resG))
    jG = json.load(open(resG, encoding="utf-8")) if os.path.isfile(resG) else {}
    check("launcher 结果 ok=True 且整图 2 张 / 子图有块",
          jG.get("ok") is True and jG.get("total_added") == 2
          and (jG.get("total_tiles_added") or 0) > 0,
          f"{jG.get('total_added')} 张 / {jG.get('total_tiles_added')} 块 "
          f"{jG.get('errors')}")
    check("launcher 收尾：request_ / working_ 都不残留",
          not os.path.isfile(fpG)
          and not os.path.isfile(os.path.join(dG, "working_batch_G1.json")),
          f"{time.time() - tG:.1f}s")

    # ------------- 12) 过渡进程：等 img_server 退出超时 → 错误 result 写回请求目录
    print("\n== 12) handoff_launcher.py 超时分支（result 必须落在 request 所在目录）==")
    gH = os.path.join(work, "H_gallery")
    make_images(gH, 1, seed=222)
    dH = os.path.join(work, "H_handoff")
    sleeper = subprocess.Popen([sys.executable, "-X", "utf8", "-c",
                                "import time; time.sleep(30)"])
    try:
        fpH = write_request(dH, "batch_H1", gH, modes=["full", "tiles"],
                            prefix=default_prefix(gH),
                            expect_exit={"pids": [sleeper.pid], "names": []})
        cpH = subprocess.run([sys.executable, "-X", "utf8", launcher, fpH, "--wait", "2"],
                             cwd=BASE, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=180)
        outH = (cpH.stdout or "") + (cpH.stderr or "")
        check("超时返回码 1", cpH.returncode == 1,
              f"rc={cpH.returncode} {outH.strip()[-200:]}")
        resH = os.path.join(dH, "result_batch_H1.json")
        check("错误 result 落在 request 所在目录（img_server 侧可见）", os.path.isfile(resH))
        jH = json.load(open(resH, encoding="utf-8")) if os.path.isfile(resH) else {}
        check("超时 result ok=False 且是「等待超时」文案",
              jH.get("ok") is False and "超时" in str(jH.get("fatal_error", "")),
              str(jH.get("fatal_error")))
        check("错误 result 没有落到本程序默认 handoff\\ 目录",
              not os.path.isfile(os.path.join(BASE, "handoff", "result_batch_H1.json")))
        check("超时后 request_ 原样保留（未开工、未转 working_）",
              os.path.isfile(fpH)
              and not os.path.isfile(os.path.join(dH, "working_batch_H1.json")))
        check("超时后没有开始建库（图库内无索引目录）",
              not os.path.isdir(os.path.join(gH, ".gallery_index")))
        check("打印了超时提示与 result 落点",
              "[launcher] 超时放弃" in outH and "result_batch_H1.json" in outH)
    finally:
        sleeper.kill()
        sleeper.wait(timeout=10)

    if FAIL:
        print("\n---- CLI 输出（调试用）----")
        print(out[-1500:])
finally:
    if FAIL:
        print(f"\n（保留工作目录以便排查：{work}）")
    else:
        shutil.rmtree(work, ignore_errors=True)

print()
if FAIL:
    print(f"FAILED (pass={PASS}, fail={FAIL})")
    sys.exit(1)
print(f"ALL PASS (pass={PASS})")
