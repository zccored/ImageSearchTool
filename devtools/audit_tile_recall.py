# -*- coding: utf-8 -*-
# ImageSearchTool · 只读瓦片召回审计 / Read-only tile recall audit
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""Compare production retrieval with exhaustive cosine ranking, without index writes."""
import argparse
from dataclasses import asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from archive_batch import create_archive, digest
from hybrid_search.config import Config
from hybrid_search.engine import HybridEngine
from hybrid_search.tile_index import (search_tiles_tiled, _lsh_for, _hash_keys,
                                     _decode_to_crops, _tile_md5_of)
from hybrid_search.coarse import extract_binary_features


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--coverage-only', action='store_true',
                        help='Decode region and compare all saved boxes/MD5/fp/Hu; no model inference')
    args = parser.parse_args()
    out = create_archive(args.out, parameters=vars(args))
    print(out, flush=True)
    cfg = Config()
    engine = HybridEngine(cfg)
    engine.open(args.prefix)
    paths = engine.coarse.paths
    region = Path(args.region)
    images = [p for p in region.rglob('*') if p.suffix.lower() in cfg.extensions]
    known = {norm(p) for p in paths}
    missing = [str(p) for p in images if norm(p) not in known]
    report = {'query_sha256': digest(args.query), 'index_meta': engine.meta,
              'index_meta_sha256': digest(args.prefix + '.meta.json'),
              'region_images': len(images), 'missing_paths': missing}
    def save():
        (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    by_name = defaultdict(list)
    for path in set(paths):
        by_name[Path(path).name].append(path)
    duplicates, unexplained = [], []
    for path in missing:
        sha = digest(path)
        same = next((p for p in by_name[Path(path).name] if digest(p) == sha), None)
        if same:
            duplicates.append({'path': path, 'indexed_copy': same, 'sha256': sha})
        else:
            unexplained.append(path)
    report.update(verified_duplicates=duplicates, unexplained_missing=unexplained,
                  region_indexed_images=len({p for p in paths if norm(p).startswith(norm(region) + os.sep)}))
    save()
    if args.coverage_only:
        import cv2
        rows_of = defaultdict(list)
        for row, path in enumerate(paths):
            if norm(path).startswith(norm(region) + os.sep):
                rows_of[path].append(row)
        protocol = engine.meta['tiles']
        def check(path):
            rows = rows_of[path]
            actual = {tuple(map(int, engine.coarse.box_at(r))): r for r in rows}
            crops = _decode_to_crops(path, cfg, protocol['tile'], protocol['overlap'],
                                     protocol['min_side'], protocol['pre_max'])
            expected = {tuple(c[1]) for c in crops or ()}
            mismatches = []
            for crop, box, base, _first in crops or ():
                row = actual.get(tuple(box))
                if row is None:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                _binary, hu, fp = extract_binary_features(gray, cfg)
                if (engine.coarse.md5s[row] != _tile_md5_of(base, box)
                        or not np.array_equal(engine.coarse.fp[row], fp)
                        or not np.array_equal(engine.coarse.hu[row], hu)):
                    mismatches.append(tuple(box))
            return {'path': path, 'rows': len(rows), 'expected': len(expected),
                    'missing': sorted(expected - actual.keys()),
                    'extra': sorted(actual.keys() - expected), 'feature_mismatch': mismatches,
                    'duplicate_boxes': len(rows) - len(actual)}
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            coverage = list(pool.map(check, rows_of))
        report['coverage_wall'] = time.perf_counter() - start
        report['coverage'] = coverage
        report['coverage_failures'] = [r for r in coverage if r['missing'] or r['extra']
                                       or r['feature_mismatch'] or r['duplicate_boxes']]
        save()
        print('coverage', len(coverage), sum(r['rows'] for r in coverage),
              'failures', len(report['coverage_failures']), flush=True)
        if report['coverage_failures'] or unexplained:
            raise RuntimeError('Coverage audit failed; see audit.json')
        return
    start = time.perf_counter()
    result = search_tiles_tiled(engine, args.query, top_k=100)
    report['production'] = asdict(result)
    report['production_wall'] = time.perf_counter() - start
    save()
    print('production', report['production_wall'], [(h.rank, h.path, h.fine_score) for h in result.hits[:3]], flush=True)
    q = engine._query_fine(args.query)
    feats = engine._fine_feats
    scores = np.empty(len(paths), dtype=np.float32)
    start = time.perf_counter()
    for i in range(0, len(paths), 4096):
        block = np.array(feats[i:i+4096], dtype=np.float32, copy=True)
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-8)
        scores[i:i+len(block)] = block @ q
    report['exhaustive_score_wall'] = time.perf_counter() - start
    candidates = set(map(int, _lsh_for(engine, feats, 12, 8).query(q, top_override=12000)))
    ranked, region_ranked, seen = [], [], set()
    for row in np.argsort(-scores, kind='stable'):
        path = paths[row]
        if path in seen:
            continue
        seen.add(path)
        item = {'rank': len(seen), 'path': path, 'row': int(row), 'score': float(scores[row]),
                'box': engine.coarse.box_at(int(row)), 'in_lsh': int(row) in candidates}
        if len(ranked) < 100:
            ranked.append(item)
        if norm(path).startswith(norm(region) + os.sep) and len(region_ranked) < 30:
            region_ranked.append(item)
    report.update(exhaustive_top=ranked, region_top=region_ranked, lsh_candidates=len(candidates))
    lsh = _lsh_for(engine, feats, 12, 8)
    row = ranked[0]['row']
    collisions = []
    for table, projection in zip(lsh._tables, lsh._projs):
        qkey = int(_hash_keys(q.reshape(1, -1), projection)[0])
        target_key = int(_hash_keys(np.asarray(feats[row:row+1], dtype=np.float32), projection)[0])
        ahead = 0
        if qkey == target_key:
            for begin in range(0, row, 4096):
                keys = _hash_keys(np.asarray(feats[begin:min(begin+4096, row)], dtype=np.float32), projection)
                ahead += int(np.count_nonzero(keys == qkey))
        collisions.append({'same_bucket': qkey == target_key, 'earlier_rows_in_bucket': ahead,
                           'stored_bucket_size': len(table.get(qkey, ()))})
    report['top1_old_lsh_collision_audit'] = collisions
    save()
    print('exact', ranked[:5], 'region', region_ranked[:5], flush=True)


if __name__ == '__main__':
    main()
