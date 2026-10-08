# -*- coding: utf-8 -*-
# ImageSearchTool · 真实索引只读召回验收 / Read-only live recall validation
# Copyright (C) 2026 zccored
# 本程序是自由软件：AGPL-3.0-only；完整条款见 LICENSE。
# This program is free software under the GNU Affero General Public License
# v3.0 (AGPL-3.0-only), WITHOUT ANY WARRANTY. See LICENSE for terms.
"""Fresh-process ABBA + CLI/service validation. All generated outputs are archived."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from archive_batch import create_archive, digest


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def child(args):
    from hybrid_search.config import Config
    from hybrid_search.engine import HybridEngine
    from hybrid_search import tile_index as ti
    if args.child == 'baseline':
        spec = importlib.util.spec_from_file_location('hybrid_search._recall_baseline', args.baseline)
        ti = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ti)
    start = time.perf_counter()
    engine = HybridEngine(Config())
    engine.open(args.prefix)
    load = time.perf_counter() - start
    results = []
    process = psutil.Process()
    for i in range(4):
        t0 = time.perf_counter()
        cpu0 = sum(process.cpu_times()[:2])
        out = ti.search_tiles_tiled(engine, args.query, top_k=100,
                                    method='lsh' if args.child == 'baseline' else 'exact')
        results.append({'wall': time.perf_counter() - t0,
                        'cpu': sum(process.cpu_times()[:2]) - cpu0,
                        'rss': process.memory_info().rss,
                        'outcome': asdict(out)})
    write(args.result, {'kind': args.child, 'load': load, 'runs': results})


def no_other_python():
    for p in psutil.process_iter(['pid', 'name']):
        if p.pid != os.getpid() and (p.info['name'] or '').lower() in ('python.exe', 'pythonw.exe'):
            raise RuntimeError(f'Other Python process present: {p.pid}; no concurrent validation')


def main(args):
    no_other_python()
    out = create_archive(args.out, parameters={**vars(args), 'baseline_sha256': digest(args.baseline)})
    print(out, flush=True)
    prefix = Path(args.prefix)
    files = sorted(prefix.parent.glob(prefix.name + '.*'))
    files += sorted(prefix.parent.glob('gallery.*'))
    before = {str(p): digest(p) for p in files if p.is_file()}
    target_before = digest(args.query)
    summary = {'index_before': before, 'query_before': target_before, 'abba': []}
    write(out / 'summary.json', summary)
    def run(name, command):
        no_other_python()
        with (out / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); see archived log')
        print(name, 'PASS', flush=True)
    try:
        for i, kind in enumerate(() if args.skip_abba else ('baseline', 'fixed', 'fixed', 'baseline')):
            name = f'{i}_{kind}'
            result = out / (name + '.json')
            command = [sys.executable, '-E', '-B', '-X', 'utf8', __file__,
                       '--child', kind, '--prefix', args.prefix, '--query', args.query,
                       '--baseline', args.baseline, '--result', str(result)]
            run(name, command)
            data = json.loads(result.read_text(encoding='utf-8'))
            for entry in data['runs']:
                hit = any(h['path'].replace('\\', '/').casefold().startswith(args.region.replace('\\', '/').casefold() + '/')
                          for h in entry['outcome']['hits'])
                if kind == 'fixed' and not hit:
                    raise AssertionError('Fixed retrieval missed expected region')
                entry['expected_region_hit'] = hit
            summary['abba'].append(data)
            write(out / 'summary.json', summary)
        run('cli', [sys.executable, '-E', '-B', '-X', 'utf8', 'main.py', 'search', args.query,
                    '--mode', 'tiles', '--tiles-prefix', args.prefix, '--json', str(out / 'cli.json')])
        base_prefix = str(prefix.parent / 'gallery')
        run('cli_hybrid', [sys.executable, '-E', '-B', '-X', 'utf8', 'main.py', 'search', args.query,
                           '--mode', 'hybrid', '--prefix', base_prefix, '--json', str(out / 'cli_hybrid.json')])
        for name in ('cli', 'cli_hybrid'):
            result = json.loads((out / (name + '.json')).read_text(encoding='utf-8'))
            if not any(args.region.replace('\\', '/').casefold() in h['path'].replace('\\', '/').casefold()
                       for h in result['results']):
                raise AssertionError(f'{name} missed expected region')
        from hybrid_search.config import Config
        from hybrid_search.service import SearchService
        service = SearchService(cfg=Config(), capture_log=False)
        events = []
        service.subscribe(events.append)
        summary['service'] = {}
        for mode in ('tiles', 'hybrid'):
            result = service.search('live-' + mode, args.query, mode, prefix=base_prefix)
            if not result or not any(args.region.replace('\\', '/').casefold() in h[1].replace('\\', '/').casefold()
                                     for h in result['hits']):
                raise AssertionError(f'Service {mode} missed expected region')
            summary['service'][mode] = result
        # Real multi-tile query; this is a read-only self-retrieval check, not a UI setting change.
        from hybrid_search.tile_index import search_tiles_tiled
        source = summary['service']['tiles']['hits'][0][1]
        engine, _cached = service.engine_for(service.cfg, args.prefix)
        saved_exclude = engine.cfg.exclude_self
        try:
            engine.cfg.exclude_self = False
            multi = search_tiles_tiled(engine, source, top_k=100)
        finally:
            engine.cfg.exclude_self = saved_exclude
        own = next((h for h in multi.hits if os.path.normcase(os.path.abspath(h.path)) == os.path.normcase(os.path.abspath(source))), None)
        if own is None or own.fine_score < .9999:
            raise AssertionError('Real multi-tile query failed self retrieval')
        summary['multi_tile_self_check'] = asdict(multi)
        service.release_engines()
        summary['service_events'] = [e for e in events if e['event'] in ('task_done', 'task_error')]
        write(out / 'summary.json', summary)
        for name in ('verify_tile_recall', 'verify_cli_defaults', 'verify_tile_lifecycle', 'verify_fused_rollback'):
            run(name, [sys.executable, '-E', '-B', '-X', 'utf8', f'devtools/{name}.py'])
        # Existing full service regression writes only new synthetic data under this archive.
        temp = out / 'synthetic_service'
        temp.mkdir()
        code = ('import tempfile,runpy,sys; tempfile.tempdir=sys.argv[1]; '
                'sys.argv=["verify_service.py","--db","24","--queries","2","--keep"]; '
                'runpy.run_path("devtools/verify_service.py",run_name="__main__")')
        run('service_regression', [sys.executable, '-E', '-B', '-X', 'utf8', '-c', code, str(temp)])
        summary['index_after'] = {str(p): digest(p) for p in files if p.is_file()}
        summary['query_after'] = digest(args.query)
        if summary['index_after'] != before or summary['query_after'] != target_before:
            raise AssertionError('Read-only inputs changed during validation')
        summary['medians'] = {}
        for kind in ('baseline', 'fixed'):
            rows = [d for d in summary['abba'] if d['kind'] == kind]
            if not rows:
                continue
            summary['medians'][kind] = {
                'first_query': statistics.median(d['runs'][0]['wall'] for d in rows),
                'warm_query': statistics.median(r['wall'] for d in rows for r in d['runs'][1:])}
        summary['status'] = 'PASS'
        write(out / 'summary.json', summary)
        print(summary['medians'], flush=True)
    except BaseException as exc:
        summary['status'] = 'FAIL'
        summary['error'] = repr(exc)
        write(out / 'summary.json', summary)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--region')
    parser.add_argument('--out')
    parser.add_argument('--child', choices=('baseline', 'fixed'))
    parser.add_argument('--result')
    parser.add_argument('--skip-abba', action='store_true', help='Only run correctness checks; do not repeat completed ABBA')
    arguments = parser.parse_args()
    if arguments.child:
        child(arguments)
    else:
        main(arguments)
