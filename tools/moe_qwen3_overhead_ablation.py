#!/usr/bin/env python3
"""Controlled accumulator/read and reduction-tiling ablations of trained layer 2."""
import argparse
from collections import defaultdict
import copy
from pathlib import Path
import re
import shutil

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
from onnx.reference import ReferenceEvaluator

from moe_qwen3_cold_capacity import (SDK, command, compile_model, dependencies, dump,
                                    load, resources, save_csv, sha, tensor)
from moe_qwen3_profile import canon, events, summarize_trace


STEM = '/model/layers.2/mlp/'
DEFAULT_CASES = ['baseline', 'drop_zero', 'tile_t16', 'tile_h16']
CORE = re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$')


def config(name):
    if name == 'baseline':
        return False, None, 0
    drop = name.startswith('drop_zero')
    if name == 'drop_zero':
        return True, None, 0
    match = re.fullmatch(r'(?:drop_zero_)?tile_([th])(4|8|16)', name)
    if not match:
        raise ValueError(f'Unknown case {name}')
    return drop, {'t': 2, 'h': 3}[match[1]], int(match[2])


def tiled_reduce(model, axis, tiles):
    original = next(n for n in model.graph.node if n.name == STEM+'Einsum_3')
    if original.op_type != 'Einsum' or h.get_attribute_value(original.attribute[0]) != b'dpth->dth':
        raise ValueError('Unexpected local reduction')
    extent = {2: 128, 3: 2048}[axis]
    width = extent//tiles
    initializers = [nh.from_array(np.array([axis], np.int64), 'ablation_slice_axis')]
    replacement, results = [], []
    for index in range(tiles):
        start, end = f'ablation_start_{index}', f'ablation_end_{index}'
        initializers += [nh.from_array(np.array([index*width], np.int64), start),
                         nh.from_array(np.array([(index+1)*width], np.int64), end)]
        sliced, reduced = f'ablation_slice_{index}', f'ablation_reduced_{index}'
        replacement += [h.make_node('Slice', [original.input[0], start, end, 'ablation_slice_axis'],
                                    [sliced], name=STEM+f'reduce_slice_{index}'),
                        h.make_node('Einsum', [sliced], [reduced], equation='dpth->dth',
                                    name=STEM+f'reduce_tile_{index}')]
        results.append(reduced)
    replacement.append(h.make_node('Concat', results, list(original.output), axis=axis-1, name=original.name))
    nodes = []
    for node in model.graph.node:
        nodes.extend(replacement if node.name == original.name else [copy.deepcopy(node)])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    # A standalone CPU check exercises the exact replacement subgraph, preserving
    # each output's expert-axis reduction instead of reassociating partial sums.
    graph = h.make_graph(replacement, 'tile_check', [tensor(original.input[0], TP.FLOAT, [4,16,128,2048])],
                         [tensor(original.output[0], TP.FLOAT, [4,128,2048])], initializers)
    check_model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
    rng = np.random.default_rng(20260922)
    source = rng.standard_normal((4,16,128,2048), dtype=np.float32)
    result = ReferenceEvaluator(check_model).run(None, {original.input[0]: source})[0]
    expected = np.einsum('dpth->dth', source)
    delta = np.abs(result-expected)
    if not np.allclose(result, expected, rtol=1e-6, atol=1e-5):
        raise ValueError('Tiled reduction CPU semantic check failed')
    return dict(axis=axis, tiles=tiles, width=width, input_shape=list(source.shape),
                cpu_max_abs=float(delta.max()), cpu_equal=bool(np.array_equal(result, expected)))


def export(args, name):
    out = args.out/name
    out.mkdir(exist_ok=False)
    source = args.cold/'c2/model.onnx'
    model = onnx.load(source, load_external_data=False)
    baseline = copy.deepcopy(model)
    drop, axis, tiles = config(name)
    changes = []
    if drop:
        by_name = {n.name: n for n in model.graph.node}
        gather, add, zero = [by_name[STEM+n] for n in ['CtxGather3D_2', 'Add', 'ConstantOfShape_1']]
        if (gather.op_type != 'CtxGather3D' or gather.input[0] != zero.output[0]
                or list(add.input) != [gather.output[0], by_name[STEM+'Mul_5'].output[0]]):
            raise ValueError('First update is not a gather of the zero accumulator plus weighted outputs')
        value = next(h.get_attribute_value(a) for a in zero.attribute if a.name == 'value')
        if np.any(nh.to_array(value) != 0):
            raise ValueError('Accumulator is not zero initialized')
        if sum(gather.output[0] in n.input for n in model.graph.node) != 1:
            raise ValueError('Unexpected use of first accumulator gather')
        old, new = add.output[0], add.input[1]
        nodes = []
        for node in model.graph.node:
            if node.name in [gather.name, add.name]:
                continue
            node = copy.deepcopy(node)
            for index, value in enumerate(node.input):
                if value == old:
                    node.input[index] = new
            nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        changes.append(dict(remove=[gather.name, add.name], replace_tensor={old: new},
                            zero_initialization_retained=True))
    if axis:
        changes.append(tiled_reduce(model, axis, tiles))
    before = {n.name: n for n in baseline.graph.node}
    after = {n.name: n for n in model.graph.node}
    for suffix in ['MatMul', 'MatMul_1', 'MatMul_2', 'MatMul_3', 'MatMul_4', 'MatMul_5',
                   'Slice', 'Slice_1', 'Range_1', 'Range_3', 'Einsum_4']:
        if before[STEM+suffix] != after[STEM+suffix]:
            raise ValueError('Unexpected change to protected expert/capacity/final-reduction node')
    if list(model.graph.initializer[:len(baseline.graph.initializer)]) != list(baseline.graph.initializer):
        raise ValueError('Original initializers changed')
    dependencies(out, args.cold/'c2')
    onnx.save(model, out/'model.onnx')
    onnx.checker.check_model(str(out/'model.onnx'))
    if name == 'baseline' and sha(source) != sha(out/'model.onnx'):
        raise ValueError('Baseline serialization changed')
    dump(out/'export.json', dict(case=name, source=str(source), source_sha256=sha(source),
         graph_sha256=sha(out/'model.onnx'), input_sha256=sha(args.cold/'input_f16.bin'),
         trained_weight_references=load(args.cold/'c2/export.json')['weights'],
         hot_capacity=128, cold_capacity=2, expert_nodes_unchanged=True, changes=changes))
    print('EXPORT_OK', name, flush=True)


def qpc(args, name):
    if args.stats_level == 0:
        return args.scratch/f'{name}_stats0_compile/qpc'
    return (args.base_scratch/'c2_compile' if name == 'baseline' else args.scratch/f'{name}_compile')/'qpc'


def compile_case(args, name):
    out = args.out/name
    if name == 'baseline' and args.stats_level == 70:
        if not (qpc(args, name)/'programqpc.bin').exists():
            raise ValueError('The original C2 control QPC is unavailable')
        dump(out/'compile_reuse.json', dict(qpc=str(qpc(args, name)),
             graph_sha256=sha(args.cold/'c2/model.onnx'), qpc_sha256=sha(qpc(args, name)/'programqpc.bin')))
        return
    target = qpc(args, name).parent
    try:
        compile_model(out/'model.onnx', target, stats_level=args.stats_level)
    finally:
        for filename in ['compile.log', 'compile.command.json', 'result.json']:
            if (target/filename).exists():
                copied_name = filename if args.stats_level == 70 else 'stats0_'+filename
                shutil.copy2(target/filename, out/copied_name)


def validate(args, out):
    reference = args.cold/'c2/timing_r0/y.bin'
    y = np.fromfile(out/'y.bin', np.float16)
    ref = np.fromfile(reference, np.float16)
    counts = np.fromfile(out/'counts.bin', np.int32)
    expected = np.fromfile(args.cold/'expected_counts_i32.bin', np.int32)
    if (y.shape != (128*2048,) or not np.isfinite(y).all() or not np.array_equal(counts, expected)
            or counts.shape != (128,) or counts.sum() != 1024
            or counts.min() < 0 or counts[:64].max() > 128 or counts[64:].max() > 2):
        raise ValueError('Invalid output or routing mismatch')
    delta = y.astype(np.float64)-ref.astype(np.float64)
    result = dict(reference=str(reference), bit_exact=(out/'y.bin').read_bytes()==reference.read_bytes(),
                  counts_exact=True, no_overflow=True, max_abs=float(np.abs(delta).max()),
                  relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(ref)))
    dump(out/'validation.json', result)
    if result['relative_l2'] > 0.001:
        raise ValueError(f'Material numerical drift: {result}')
    return result


def timing(args, qpc_fn=qpc):
    host = args.base_scratch/'moe_single_host'
    if not host.is_file():
        raise ValueError('Build the existing single-QPC benchmark host first')
    for index in range(args.rounds):
        order = args.cases if index % 2 == 0 else list(reversed(args.cases))
        for name in order:
            case = args.out/name
            out = case/f'{args.timing_prefix}_r{index}'
            resources(case/f'{args.timing_prefix}_before_r{index}.txt')
            command([host, qpc_fn(args, name), args.cold/'input_f16.bin', out, args.iterations, 10],
                    case/f'{args.timing_prefix}_r{index}.log')
            resources(case/f'{args.timing_prefix}_after_r{index}.txt')
            result = validate(args, out)
            reference = case/f'{args.timing_prefix}_r0/y.bin'
            if (out/'y.bin').read_bytes() != reference.read_bytes():
                raise ValueError('Nonrepeatable output across rounds')
            print('TIMING_OK', name, index, float(np.median(load(out/'timing.json')['samples_ms'])),
                  'bit_exact', result['bit_exact'], flush=True)


def profile(args, name, qpc_fn=qpc):
    case = args.out/name
    out = case/'profile'
    out.mkdir(exist_ok=False)
    expected = case/f'{args.timing_prefix}_r0'
    bindings = []
    for io_name, path, dims, size, direction in [
        ('probe_input', args.cold/'input_f16.bin', [128,2176], 2, 'in'),
        ('y', expected/'y.bin', [128,2048], 2, 'out'),
        ('counts', expected/'counts.bin', [128], 4, 'out')]:
        if path.stat().st_size != int(np.prod(dims))*size:
            raise ValueError('Unexpected IO size')
        bindings.append({'path': str(path), 'dims': dims, 'elem-size': size,
                         'io-direction': direction, 'map-to': io_name})
    dump(out/'io.json', {'IO-files': [bindings]})
    for directory in ['stats', 'outputs', 'trace']:
        (out/directory).mkdir()
    resources(out/'resources_before.txt')
    command([SDK/'exec/qaic-runner', '-t', qpc_fn(args, name), '-D', '0:1:2:3',
             '--aic-batch-json-input', out/'io.json', '-n', 5, '-S', 1, '-T', 1, '-c',
             '--aic-profiling-type', 'raw_device_stats', '--aic-profiling-start-iter', 2,
             '--aic-profiling-num-samples', 3, '--aic-profiling-out-dir', out/'stats',
             '--write-output-start-iter', 2, '--write-output-num-samples', 3,
             '--write-output-dir', out/'outputs'], out/'runner.log')
    resources(out/'resources_after.txt')
    checked = defaultdict(list)
    for path in (out/'outputs').rglob('*'):
        if not path.is_file():
            continue
        key = {524288: 'y', 512: 'counts'}.get(path.stat().st_size)
        if key:
            if path.read_bytes() != (expected/f'{key}.bin').read_bytes():
                raise ValueError('Profile output differs from timing output')
            checked[key].append(str(path))
    if any(len(checked[k]) != 3 for k in ['y', 'counts']):
        raise ValueError('Missing profile output')
    command([SDK/'exec/qaic-opstats', '--qpc', qpc_fn(args, name)/'programqpc.bin',
             '--input-dir', out/'stats', '--output-dir', out/'trace', '--summary', '--trace',
             '--merge-mq-traces', 'true', '--flow-events', 'full'], out/'opstats.log')
    dump(out/'validation.json', dict(bit_exact_to_own_timing=True, samples=3, files=dict(checked)))
    print('PROFILE_OK', name, flush=True)


def trace_metrics(path, require_full_expert_coverage=True, allow_card0_reduction=False):
    nodes, device_ms = summarize_trace(path)
    threads = {(e['pid'], e['tid']): e['args']['name'] for e in events(path, metadata_only=True)
               if e['name']=='thread_name'}
    work, execution, p2p = [], [], []
    for event in events(path, work_only=True):
        args = event.get('args', {})
        if args.get('PortType')=='P2P' and '->' in args.get('PortDescription', ''):
            p2p.append(dict(node=canon(args['opName']), port=args['PortDescription'], bytes=int(args['opOutputSize'])))
        match = CORE.search(threads.get((event['pid'], event['tid']), ''))
        if not match:
            continue
        engine = match[3] or 'execution'
        start, dur = float(event['ts']), float(event.get('dur', 0))
        if engine=='execution':
            execution.append((start, start+dur))
        else:
            work.append(dict(node=canon(args.get('opName', '')), card=int(match[1]), core=int(match[2]),
                engine=engine, kind=args.get('opKind', '').strip(), memory=args.get('opMemory', ''),
                start_us=start, end_us=start+dur, duration_us=dur))
    result = dict(device_ms=device_ms, device_start_ms=min(a for a, _ in execution)/1000,
                  p2p_bytes=sum(r['bytes'] for r in p2p))
    for stage, names in [('hot', ['MatMul','MatMul_1','MatMul_2']), ('cold', ['MatMul_3','MatMul_4','MatMul_5'])]:
        selected = [r for r in work if r['node'] in [STEM+n for n in names] and r['engine']=='HMX'
                    and r['kind']=='aicconvolutiond32']
        result[stage+'_start_ms'] = min(r['start_us'] for r in selected)/1000
        result[stage+'_end_ms'] = max(r['end_us'] for r in selected)/1000
        result[stage+'_span_ms'] = result[stage+'_end_ms']-result[stage+'_start_ms']
        result[stage+'_cores'] = len({(r['card'], r['core']) for r in selected})
        if require_full_expert_coverage and result[stage+'_cores'] != 64:
            raise ValueError('Expert core coverage changed')
    local = [r for r in work if (r['node']==STEM+'Einsum_3' or r['node'].startswith(STEM+'reduce_tile_'))
             and r['engine']=='HVX' and r['kind']=='aicbatchedreduceadd']
    final = [r for r in work if r['node']==STEM+'Einsum_4' and r['engine'] in ['HMX','HVX']
             and r['kind']!='aicendcyclestats']
    if not local or not final:
        raise ValueError('Missing reduction events; inspect compiler lowering')
    result['local_start_ms'] = min(r['start_us'] for r in local)/1000
    result['local_end_ms'] = max(r['end_us'] for r in local)/1000
    result['final_end_ms'] = max(r['end_us'] for r in final)/1000
    result['combine_span_ms'] = result['final_end_ms']-result['local_start_ms']
    result['post_gemm_tail_ms'] = result['final_end_ms']-result['cold_end_ms']
    for memory in ['all', 'DDR', 'TCM']:
        selected = [r for r in local if memory=='all' or r['memory']==memory]
        busy = defaultdict(float)
        for r in selected:
            busy[r['card'], r['core']] += r['duration_us']/1000
        result[f'local_{memory}_cores'] = len(busy)
        result[f'local_{memory}_max_core_ms'] = max(busy.values(), default=0)
        result[f'local_{memory}_sum_core_ms'] = sum(busy.values())
    # Each local reduction lowers to one partial kernel per core and a second
    # merge kernel on core 0. Tiling preserves this placement but changes the
    # merge's memory path; do not call the gain a 16-way parallel merge.
    merge_busy = defaultdict(float)
    merge_memory = set()
    compact = all({(r['card'],r['core']) for r in local if r['node']==node} == {(0,c) for c in range(4)}
                  and sum(r['node']==node for r in local)==4 for node in {r['node'] for r in local})
    if compact:
        if not allow_card0_reduction:
            raise ValueError('Reduction moved to four cores of card 0; inspect lowering')
        # Observed with cold-path compaction: one kernel per tile on card-0
        # cores 0..3, with no second local merge kernel. Do not mislabel it.
        result['local_reduction_layout'] = 'card0_cores0_3'
        result['core0_merge_max_ms'] = None
        result['core0_merge_memory'] = None
        return result, nodes, work, p2p
    for card in range(4):
        for node in {r['node'] for r in local}:
            selected = [r for r in local if r['card']==card and r['node']==node]
            per_core = {core: sorted([r for r in selected if r['core']==core],
                                    key=lambda r: r['start_us']) for core in range(16)}
            if len(per_core[0]) != 2 or any(len(per_core[c]) != 1 for c in range(1,16)):
                raise ValueError('Changed reduction lowering; inspect core-0 merge classification')
            merge = per_core[0][1]
            merge_busy[card] += merge['duration_us']/1000
            merge_memory.add(merge['memory'])
    result['core0_merge_max_ms'] = max(merge_busy.values())
    result['core0_merge_memory'] = ','.join(sorted(merge_memory))
    result['local_reduction_layout'] = 'all_cards_16cores_then_core0_merge'
    return result, nodes, work, p2p


def analyze(args):
    summaries = []
    for name in args.cases:
        case = args.out/name
        out = case/'analysis'
        out.mkdir(exist_ok=True)
        rounds = [load(case/f'{args.timing_prefix}_r{i}/timing.json') for i in range(args.rounds)]
        host = [s for r in rounds for s in r['samples_ms']]
        samples, all_nodes, all_work, all_p2p = [], [], [], []
        paths = sorted((case/'profile/trace').glob('*merged*.trace.json'))
        if len(paths) != 3 or not load(case/'profile/validation.json')['bit_exact_to_own_timing']:
            raise ValueError('Need three validated traces')
        for index, path in enumerate(paths):
            metrics, nodes, work, p2p = trace_metrics(path)
            samples.append(dict(case=name, sample=index, trace=str(path), **metrics))
            all_nodes.extend(dict(sample=index, **r) for r in nodes)
            all_work.extend(dict(sample=index, **r) for r in work)
            all_p2p.extend(dict(sample=index, **r) for r in p2p)
        for key, values in [('samples', samples), ('nodes', all_nodes), ('work', all_work), ('p2p', all_p2p)]:
            save_csv(out/f'{key}.csv', values)
        chosen = sorted(samples, key=lambda r: r['device_ms'])[1]
        result = dict(chosen, host_median_ms=float(np.median(host)), host_p10_ms=float(np.percentile(host,10)),
            host_p90_ms=float(np.percentile(host,90)), host_round_medians_ms=[float(np.median(r['samples_ms'])) for r in rounds],
            host_samples=len(host), samples=samples,
            validation=load(case/f'{args.timing_prefix}_r0/validation.json'), export=load(case/'export.json'))
        dump(out/'summary.json', result)
        summaries.append(result)
        print('ANALYZE_OK', name, 'host', result['host_median_ms'], 'device', result['device_ms'],
              'local', result['local_all_max_core_ms'], 'DDR cores', result['local_DDR_cores'], flush=True)
    dump(args.out/f'{args.summary_name}.json', dict(scope='FP16 layer 2, hot128/cold2, fixed real input/routing/weights',
         timing_prefix=args.timing_prefix, rounds=args.rounds, iterations=args.iterations, cases=summaries))


def timing_summary(args):
    cases = []
    for name in args.cases:
        directory = args.out/name
        rounds = [load(directory/f'{args.timing_prefix}_r{i}/timing.json') for i in range(args.rounds)]
        samples = [s for r in rounds for s in r['samples_ms']]
        validation = [load(directory/f'{args.timing_prefix}_r{i}/validation.json') for i in range(args.rounds)]
        if len(samples) != args.rounds*args.iterations or not all(v['bit_exact'] for v in validation):
            raise ValueError('Incomplete or non-exact confirmation')
        cases.append(dict(case=name, host_median_ms=float(np.median(samples)),
            host_p10_ms=float(np.percentile(samples, 10)), host_p90_ms=float(np.percentile(samples, 90)),
            host_round_medians_ms=[float(np.median(r['samples_ms'])) for r in rounds],
            host_samples=len(samples), all_saved_outputs_bit_exact=True))
    dump(args.out/f'{args.summary_name}.json', dict(stats_level=args.stats_level,
         timing_prefix=args.timing_prefix, rounds=args.rounds, iterations=args.iterations, cases=cases))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['export','compile','timing','profile','analyze','timing-summary'])
    p.add_argument('--cold', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--base-scratch', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--timing-prefix', default='timing')
    p.add_argument('--summary-name', default='summary')
    p.add_argument('--stats-level', type=int, choices=[0,70], default=70)
    args = p.parse_args()
    if args.rounds < 1 or args.iterations < 1 or len(set(args.cases)) != len(args.cases):
        p.error('Need positive timing counts and distinct cases')
    if args.stats_level == 0 and args.action in ['profile','analyze']:
        p.error('Stats-level 0 has no device trace; use timing-summary')
    for key in ['cold','out','scratch','base_scratch']:
        setattr(args, key, getattr(args,key).resolve())
    for name in args.cases:
        config(name)
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    if args.action=='timing':
        timing(args)
    elif args.action=='analyze':
        analyze(args)
    elif args.action=='timing-summary':
        timing_summary(args)
    else:
        for name in args.cases:
            {'export': export, 'compile': compile_case, 'profile': profile}[args.action](args, name)


if __name__=='__main__':
    main()
