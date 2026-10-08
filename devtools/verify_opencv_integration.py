# -*- coding: utf-8 -*-
# ImageSearchTool · 线程策略的交接与 CLI 实链路回归
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from hybrid_search.config import Config
    from hybrid_search.service import SearchService
    from hybrid_search.runtime import opencv_thread_policy
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", type=int)
    args = parser.parse_args()
    base = Path(args.out).resolve()
    gallery = Path(r"F:\视频").resolve()
    if base == gallery or gallery in base.parents:
        raise ValueError("回归禁止写真实图库")
    if args.policy is not None:
        root = base / f"images_{args.policy}"
        svc = SearchService(Config(opencv_threads=args.policy), capture_log=False)
        req_id = f"policy{args.policy}"
        request = {"schema": 2, "kind": "download_batch_complete", "request_id": req_id,
                   "source": "verify_opencv_integration", "roots": [{"path": str(root)}],
                   "prefix": svc.auto_prefix(str(root)), "open_mode": "cli",
                   "expect_exit": {"pids": [], "names": []}, "modes": ["full", "tiles"]}
        path = base / f"request_{req_id}.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        result = svc.handoff(req_id, str(path))
        assert result and result.get("ok"), result
        assert result["total_added"] == 4 and result["total_tiles_added"] > 0, result
        assert opencv_thread_policy() == args.policy
        svc.close()
        print("HANDOFF_OK", args.policy, result["total_added"], result["total_tiles_added"], flush=True)
        return
    import numpy as np
    from PIL import Image
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="opencv_integration_", dir=base))
    rng = np.random.RandomState(817)
    for policy in (0, 2):
        root = work / f"images_{policy}"
        root.mkdir()
        for i in range(4):
            image = Image.fromarray(rng.randint(0, 256, (480, 640, 3), dtype=np.uint8))
            image.save(root / f"{i}.{'jpg' if i % 2 else 'png'}")
        subprocess.run([sys.executable, "-E", "-B", "-X", "utf8", __file__,
                        "--out", str(work), "--policy", str(policy)], check=True)
    # 真 CLI：显式 0 建库，再以默认 1 读取同一索引查询，不改变索引兼容性。
    root = work / "images_0"
    prefix = work / "cli" / "gallery"
    command = [sys.executable, "-E", "-B", "-X", "utf8", str(ROOT / "main.py")]
    subprocess.run(command + ["build", str(root), "--prefix", str(prefix), "--opencv-threads", "0",
                              "--no-prep-cache", "--no-progress"], check=True)
    result_path = work / "query.json"
    subprocess.run(command + ["search", str(root / "0.png"), "--prefix", str(prefix),
                              "--json", str(result_path)], check=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result.get("db_size") == 4 and len(result.get("results", [])) == 3, result
    assert all(Path(hit["path"]).parent == root for hit in result["results"])
    (work / "summary.json").write_text(json.dumps({"handoff_policies": [0, 2],
        "cli_build_policy": 0, "cli_search_policy": Config().opencv_threads, "passed": True}), encoding="utf-8")
    print("INTEGRATION_OK", work, flush=True)


if __name__ == "__main__":
    main()
