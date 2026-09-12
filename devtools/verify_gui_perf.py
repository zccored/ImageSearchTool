# -*- coding: utf-8 -*-
"""验证：引擎缓存/内存释放 + 性能图导出 + add_tiles 增量正确性。

不弹窗口（子类化 App 但跳过 __init__，只借用其方法）。
"""
import os
import queue
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, r"D:\code\新的代码\全栈图库管理器 v3.2bata\image-search")
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from hybrid_search.config import Config  # noqa: E402
import gui as G  # noqa: E402
import perfwatch as PW  # noqa: E402

PROC = psutil.Process()
ROOT = r"F:\视频"
Q = r"F:\靶子\77C93F54F2277365732D6E39B73878E4.png"


def rss():
    return PROC.memory_info().rss / 2 ** 20


class V:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


class Stub(G.App):
    def __init__(self, mode="tiles", perf_search=False, perf_build=False):
        self.prefix = os.path.join(ROOT, ".gallery_index", "gallery")
        self.search_mode_var = V(mode)
        self.perf_search_var = V(perf_search)
        self.perf_build_var = V(perf_build)
        self.q = queue.Queue()
        self._eng_lock = threading.Lock()
        self._eng_cache = {}
        self._perf_ok = set()
        self._last_perf_report = ""
        self.btn_perf_open = None
        self.all_images = []
        self._viz_queue = {"coarse": __import__("collections").deque(maxlen=8),
                           "fine": __import__("collections").deque(maxlen=8)}

    def _log(self, text):
        print("    [log]", text)

    def _perf_report_ready(self, path, modal=False):
        print("    [perf_ready]", path, "modal" if modal else "")


def drain(st, kinds=None):
    got = []
    while True:
        try:
            m = st.q.get_nowait()
        except queue.Empty:
            break
        if kinds is None or m[0] in kinds:
            got.append(m)
    return got


HAS_TILES = os.path.exists(os.path.join(
    ROOT, ".gallery_index", "gallery_tiles.meta.json"))
MODE = "tiles" if HAS_TILES else "full"
if not HAS_TILES:
    print("（未发现瓦片索引 gallery_tiles，本轮回退用整图索引测试）")

print("=========== A) 搜图：引擎缓存 + 内存 ===========")
st = Stub(MODE)
cfg = Config()
cfg.top_k = 5
for i in range(3):
    t0 = time.time()
    G.App._search_worker(st, Q, cfg)
    msgs = drain(st, kinds={"search_done", "error"})
    dt = time.time() - t0
    if not msgs or msgs[0][0] != "search_done":
        print("  search 失败:", msgs)
        break
    d = msgs[0][1]
    top1 = os.path.basename(d["hits"][0][1]) if d["hits"] else "-"
    print(f"  search#{i} 耗时 {dt:5.2f}s  rss={rss():6.0f}MB  "
          f"top1={top1}  total={d['times'].get('total', 0):.2f}s")
print("  引擎缓存:", list(st._eng_cache.keys()))
print("  RSS 释放前:", f"{rss():.0f} MB")
freed = st._drop_engines("验证")
drain(st)
print(f"  _drop_engines 后 rss={rss():.0f}MB (报告回落 {freed:.0f}MB)")

print("=========== B) 搜图性能图导出 ===========")
st2 = Stub(MODE, perf_search=True)
G.App._search_worker(st2, Q, cfg)
msgs = drain(st2, kinds={"search_done", "error"})
perf_path = msgs[0][1].get("perf", "") if msgs and msgs[0][0] == "search_done" else ""
print("  报告:", perf_path or "(未生成)")
if perf_path:
    html = open(perf_path, encoding="utf-8").read()
    js = __import__("json").load(open(perf_path[:-5] + ".json", encoding="utf-8"))
    print(f"  HTML {len(html)} 字节；含时间轴={'时间轴' in html} "
          f"含阶段表={'阶段耗时' in html} 采样点={len(js['rows'])} "
          f"阶段={[p['name'] for p in js['phases']]}")
    print("  RSS 曲线首末:", f"{js['rows'][0].get('rssMB'):.0f} -> "
                           f"{js['rows'][-1].get('rssMB'):.0f} MB")
st2._drop_engines("验证")
drain(st2)

print("=========== C) 索引阶段性能图 + add_tiles 增量 ===========")
work = tempfile.mkdtemp(prefix="perfwatch_gal_")
try:
    import cv2
    rng = np.random.RandomState(0)

    def make_imgs(n, tag):
        out = []
        for i in range(n):
            img = rng.randint(0, 255, (900, 1200, 3), dtype=np.uint8)
            cv2.putText(img, f"{tag}{i}", (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 12)
            p = os.path.join(work, f"{tag}_{i}.jpg")
            cv2.imwrite(p, img)
            out.append(p)
        return out

    paths = make_imgs(8, "a")
    st3 = Stub("tiles", perf_build=True)
    st3.all_images = list(paths)
    tp = os.path.join(work, ".gallery_index", "gallery_tiles")
    cfg3 = Config()
    cfg3.workers = 4
    cfg3.tile_decode_slots = 4
    t0 = time.time()
    G.App._tiles_index_worker(st3, cfg3, tp, False)
    msgs = drain(st3, kinds={"tiles_done", "error"})
    print(f"  建库 {time.time() - t0:.1f}s ->", msgs[0][0] if msgs else "无消息")
    n1 = msgs[0][1]["n"] if msgs and msgs[0][0] == "tiles_done" else 0
    perf1 = msgs[0][1].get("perf", "") if msgs and msgs[0][0] == "tiles_done" else ""
    print(f"  首建瓦片数={n1}（8 张 1200x900 -> 每张 4 块） 报告={bool(perf1)}")

    # 增量：再加 2 张，应只处理 2 张（todo 集合修复验证）
    paths += make_imgs(2, "b")
    st3.all_images = list(paths)
    t0 = time.time()
    G.App._tiles_index_worker(st3, cfg3, tp, True)
    msgs = drain(st3, kinds={"tiles_done", "error"})
    dt = time.time() - t0
    n2 = msgs[0][1]["n"] if msgs and msgs[0][0] == "tiles_done" else 0
    perf2 = msgs[0][1].get("perf", "") if msgs and msgs[0][0] == "tiles_done" else ""
    print(f"  增量 {dt:.1f}s 新增瓦片={n2}（应为 8）报告={bool(perf2)}")
    if msgs and msgs[0][0] == "error":
        print("  错误:", msgs[0][1][:400])

    # 再跑一次增量（无新图）应为 0
    t0 = time.time()
    G.App._tiles_index_worker(st3, cfg3, tp, True)
    msgs = drain(st3, kinds={"tiles_done", "error"})
    n3 = msgs[0][1]["n"] if msgs and msgs[0][0] == "tiles_done" else -1
    print(f"  空增量 {time.time() - t0:.1f}s 新增瓦片={n3}（应为 0）")

    if perf2:
        js = __import__("json").load(open(perf2[:-5] + ".json", encoding="utf-8"))
        print("  增量报告阶段:", [f"{p['name']}({p['t']:.1f}s)"
                                  for p in js["phases"]])
finally:
    shutil.rmtree(work, ignore_errors=True)
print("=========== 完成 ===========")
