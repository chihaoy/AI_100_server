#!/usr/bin/env python3
"""Capture and summarize full-model Qwen3 MoE SDK traces on four AI 100s.

Use identically instrumented (-stats-level=70) QPCs. Trace spans describe
execution inside the full model, not isolated-operator or uninstrumented time.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from moe_qwen3_baseline import REFERENCE


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(argv, log):
    argv = list(map(str, argv))
    dump(log.with_suffix('.command.json'), argv)
    with log.open('x') as output:
        subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, check=True)


def capture(args):
    args.out.mkdir(parents=True, exist_ok=False)
    bindings = []
    for name, path, dims, size, direction in [
        ('input_ids', REFERENCE / 'input_ids_i64.bin', [1, 128], 8, 'in'),
        ('position_ids', REFERENCE / 'position_ids_i64.bin', [1, 128], 8, 'in'),
        ('logits', args.expected / 'logits_f32.bin', [1, 1, 151936], 4, 'out'),
        ('routing_counts', args.expected / 'counts_i32.bin', [48, 128], 4, 'out'),
    ]:
        if path.stat().st_size != int(np.prod(dims)) * size:
            raise ValueError(f'Invalid input/expected buffer: {path}')
        bindings.append({'path': str(path.resolve()), 'dims': dims, 'elem-size': size,
                         'io-direction': direction, 'map-to': name})
    dump(args.out / 'io.json', {'IO-files': [bindings]})
    for directory in ['stats', 'outputs', 'trace']:
        (args.out / directory).mkdir()
    command(['/opt/qti-aic/exec/qaic-runner', '-t', args.qpc, '-D', '0:1:2:3',
             '--aic-batch-json-input', args.out / 'io.json', '-n', args.samples + 2,
             '-S', 1, '-T', 1, '-c', '--aic-profiling-type', 'raw_device_stats',
             '--aic-profiling-start-iter', 2, '--aic-profiling-num-samples', args.samples,
             '--aic-profiling-out-dir', args.out / 'stats',
             '--write-output-start-iter', 2, '--write-output-num-samples', args.samples,
             '--write-output-dir', args.out / 'outputs'], args.out / 'runner.log')
    # Verify every saved sample byte for byte, beyond the SDK's tolerance check.
    expected_by_size = {607744: args.expected / 'logits_f32.bin',
                        24576: args.expected / 'counts_i32.bin'}
    checked = defaultdict(list)
    for path in sorted((args.out / 'outputs').rglob('*')):
        if not path.is_file() or path.stat().st_size not in expected_by_size:
            continue
        expected = expected_by_size[path.stat().st_size]
        if path.read_bytes() != expected.read_bytes():
            raise ValueError(f'Profiled output differs from timing run: {path}')
        checked[expected.name].append(str(path))
    if any(len(checked[name]) != args.samples for name in ['logits_f32.bin', 'counts_i32.bin']):
        raise ValueError(f'Missing profiled output samples: {dict(checked)}')
    dump(args.out / 'output_validation.json', {'bit_exact': True, 'samples': args.samples,
                                             'expected': str(args.expected), 'files': dict(checked)})
    command(['/opt/qti-aic/exec/qaic-opstats', '--qpc', args.qpc / 'programqpc.bin',
             '--input-dir', args.out / 'stats', '--output-dir', args.out / 'trace',
             '--summary', '--trace', '--merge-mq-traces', 'true', '--flow-events', 'none'],
            args.out / 'opstats.log')
    traces = sorted((args.out / 'trace').glob('*merged*.trace.json'))
    if len(traces) != args.samples:
        raise ValueError(f'Expected {args.samples} merged traces, found {len(traces)}')
    dump(args.out / 'capture.json', {'qpc': str(args.qpc), 'samples': args.samples,
                                    'traces': list(map(str, traces)), 'outputs_bit_exact': True})
    print('CAPTURE_OK', args.out, flush=True)


def events(path, metadata_only=False, work_only=False):
    """SDK emits one complete JSON event per line (including merged traces)."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if metadata_only and '"thread_name"' not in line:
                continue
            if work_only and ('"ph": "X"' not in line or
                              '"name": "sync ' in line or '"name": "barrier ' in line):
                continue
            line = line.strip().rstrip(',')
            if line.startswith('{') and '"ph"' in line:
                yield json.loads(line)


def canon(name):
    return re.sub(r'/__\d+$', '', name.split()[-1]) if name else ''


def summarize_trace(path):
    threads = {}
    for event in events(path, metadata_only=True):
        if event.get('ph') == 'M' and event.get('name') == 'thread_name':
            threads[event['pid'], event['tid']] = event['args']['name']
    # Every node retains a wall-clock envelope; engine busy time is separate,
    # because adding the work of 64 cores does not give operator latency.
    nodes = defaultdict(lambda: [float('inf'), 0., 0, 0.])
    execution = []
    for event in events(path, work_only=True):
        if event.get('ph') != 'X':
            continue
        thread = threads.get((event['pid'], event['tid']), '')
        match = re.search(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$', thread)
        if not match:
            continue
        card, core = int(match[1]), int(match[2])
        engine = match[3] or 'execution'
        start, duration = float(event['ts']), float(event.get('dur', 0.))
        if engine == 'execution':
            execution.append((start, start + duration))
            continue
        if event['name'].startswith(('sync ', 'barrier ')):
            continue
        node = canon(event.get('args', {}).get('opName', ''))
        if not node:
            continue
        kind = event.get('args', {}).get('opKind', '').strip()
        key = (node, card, core, engine, kind)
        row = nodes[key]
        row[0] = min(row[0], start)
        row[1] = max(row[1], start + duration)
        row[2] += 1
        row[3] += duration
    if not execution:
        raise ValueError(f'No four-card execution timelines: {path}')
    rows = []
    for key, values in nodes.items():
        node, card, core, engine, kind = key
        rows.append(dict(node=node, card=card, core=core, engine=engine, kind=kind,
                         first_us=values[0], last_us=values[1], events=values[2], busy_us=values[3]))
    return rows, (max(end for _, end in execution) - min(start for start, _ in execution)) / 1000


def analyze(args):
    args.out.mkdir(parents=True, exist_ok=False)
    traces = sorted(args.trace_dir.glob('*merged*.trace.json*'))
    if not traces:
        raise ValueError('No merged traces')
    validation_file = args.trace_dir.parent / 'output_validation.json'
    counts = None
    if validation_file.exists():
        validation = json.loads(validation_file.read_text())
        if not validation['bit_exact']:
            raise ValueError('Unvalidated profiler outputs')
        counts = np.fromfile(Path(validation['expected']) / 'counts_i32.bin', np.int32).reshape(48, 2, 64)
        if np.any(counts < 0) or not np.all(counts.sum((1, 2)) == 1024):
            raise ValueError('Invalid routing counts associated with these traces')
    samples, raw = [], []
    for index, path in enumerate(traces):
        nodes, execution_ms = summarize_trace(path)
        for row in nodes:
            raw.append(dict(sample=index, **row))
        for layer in range(48):
            prefix = f'/model/layers.{layer}/mlp/'
            mlp = [r for r in nodes if r['node'].startswith(prefix) or
                   r['node'].startswith(f'oracle_L{layer}_routing')]
            # The router HMX start is an observable, consistent execution
            # boundary, excluding speculative expert-weight prefetches.
            router = [r for r in mlp if r['node'] == prefix + 'gate/MatMul'
                      and r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32']
            combine = [r for r in mlp if r['node'] == prefix + 'Einsum_4'
                       and r['engine'] in ['HMX', 'HVX'] and r['kind'] != 'aicendcyclestats']
            if not router or not combine or {r['card'] for r in mlp} != {0, 1, 2, 3}:
                raise ValueError(f'Missing MoE boundary/card coverage for layer {layer}, {path}')
            start = min(r['first_us'] for r in router)
            end = max(r['last_us'] for r in combine)
            if end <= start:
                raise ValueError('Invalid MoE interval')
            row = dict(sample=index, layer=layer, execution_ms=execution_ms,
                       router_start_ms=start/1000, combine_end_ms=end/1000,
                       moe_ms=(end-start)/1000,
                       mlp_activity_first_ms=min(r['first_us'] for r in mlp)/1000,
                       mlp_activity_last_ms=max(r['last_us'] for r in mlp)/1000)
            for stage in range(2):
                names = {prefix + ('MatMul' if n == 0 else f'MatMul_{n}')
                         for n in range(stage*3, stage*3+3)}
                gemm = [r for r in mlp if r['node'] in names and r['engine'] == 'HMX'
                        and r['kind'] == 'aicconvolutiond32']
                # A zero-assignment stage can have no recorded GEMMs. Require
                # independently validated counts and absence on every engine;
                # do not silently turn incomplete profiling coverage into zero.
                if (not gemm and counts is not None and counts[layer, stage].sum() == 0
                        and not any(r['node'] in names for r in mlp)):
                    row[f'stage{stage}_gemm_span_ms'] = 0.
                    row[f'stage{stage}_hmx_busy_max_core_ms'] = 0.
                    row[f'stage{stage}_no_gemm_events_zero_assignments'] = True
                    continue
                row[f'stage{stage}_no_gemm_events_zero_assignments'] = False
                if {r['node'] for r in gemm} != names:
                    raise ValueError(f'Missing expert GEMMs: layer {layer}, stage {stage}')
                row[f'stage{stage}_gemm_span_ms'] = (max(r['last_us'] for r in gemm) -
                                                    min(r['first_us'] for r in gemm))/1000
                busy = defaultdict(float)
                for r in gemm:
                    busy[r['card'], r['core']] += r['busy_us']/1000
                row[f'stage{stage}_hmx_busy_max_core_ms'] = max(busy.values())
            samples.append(row)
    for name, rows in [('samples.csv', samples), ('nodes.csv', raw)]:
        with (args.out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    layers = []
    for layer in range(48):
        values = [r for r in samples if r['layer'] == layer]
        record = dict(layer=layer, samples=len(values))
        for name in ['moe_ms', 'stage0_gemm_span_ms', 'stage1_gemm_span_ms',
                     'stage0_hmx_busy_max_core_ms', 'stage1_hmx_busy_max_core_ms']:
            times = [r[name] for r in values]
            record[name] = float(np.median(times))
            record[name + '_min'] = min(times)
            record[name + '_max'] = max(times)
        layers.append(record)
    dump(args.out / 'summary.json', {
        'scope': 'Instrumented full-model traces, four cards, first chunk of 128 tokens',
        'moe_boundary': 'Earliest router HMX start to latest final Einsum_4 compute end across four cards; '
                        'excludes earlier input conversion/prefetch and any following residual operation',
        'caveat': 'Engine spans overlap; do not sum them or sum busy time over cores to estimate latency',
        'traces': list(map(str, traces)), 'layers': layers,
        'zero_assignment_stages_without_gemm_events': sorted({
            (r['layer'], stage) for r in samples for stage in range(2)
            if r[f'stage{stage}_no_gemm_events_zero_assignments']}),
    })
    print('ANALYZE_OK', args.out, 'layers', len(layers), 'samples', len(traces), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['capture', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--qpc', type=Path)
    parser.add_argument('--expected', type=Path)
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    for name in ['out', 'qpc', 'expected', 'trace_dir']:
        if getattr(args, name):
            setattr(args, name, getattr(args, name).resolve())
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.action == 'capture' and (not args.qpc or not args.expected):
        parser.error('capture requires --qpc and --expected')
    if args.action == 'analyze' and not args.trace_dir:
        parser.error('analyze requires --trace-dir')
    {'capture': capture, 'analyze': analyze}[args.action](args)


if __name__ == '__main__':
    main()
