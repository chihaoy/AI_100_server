#!/usr/bin/env python3
"""Retune capacities and integer routing scans after the layer-2 overhead rewrites."""
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

import moe_qwen3_overhead_ablation as overhead
from moe_qwen3_cold_capacity import compile_model, dependencies, dump, load, save_csv, sha, tensor

STEM = overhead.STEM


def config(name):
    match = re.fullmatch(r'c(2|4|8|16|32|64|128)(?:_(rows4|rows16|rows64|tree))?', name)
    if not match:
        raise ValueError(f'Unknown case {name}')
    return int(match[1]), match[2]


def scan_nodes(original, kind):
    """Return an exact int32 inclusive scan on [64 experts,128 tokens]."""
    stem = original.name + '_retune_'
    nodes, initializers = [], []

    def constant(name, data):
        name = stem + name
        initializers.append(nh.from_array(np.asarray(data, dtype=np.int64), name))
        return name

    if kind.startswith('rows'):
        tiles = int(kind[4:])
        width = 64 // tiles
        axis = constant('expert_axis', [0])
        results = []
        for index in range(tiles):
            sliced, scanned = stem+f'slice_{index}', stem+f'scan_{index}'
            nodes += [h.make_node('Slice', [original.input[0], constant(f'start_{index}', [index*width]),
                       constant(f'end_{index}', [(index+1)*width]), axis], [sliced], name=sliced),
                      h.make_node('CumSum', [sliced, original.input[1]], [scanned], name=scanned)]
            results.append(scanned)
        nodes.append(h.make_node('Concat', results, list(original.output), axis=0, name=original.name))
    elif kind == 'tree':
        # Hillis-Steele inclusive scan: stage d adds the previous stage shifted
        # by d=1,2,...,64. All values remain integer, with no change of precision.
        axis, start = constant('token_axis', [1]), constant('start', [0])
        current = original.input[0]
        for shift in [1, 2, 4, 8, 16, 32, 64]:
            zero, sliced, shifted, result = [stem+f'{part}_{shift}' for part in ['zero','slice','shift','add']]
            initializers.append(nh.from_array(np.zeros((64,shift), np.int32), zero))
            nodes += [h.make_node('Slice', [current, start, constant(f'end_{shift}', [128-shift]), axis],
                                  [sliced], name=sliced),
                      h.make_node('Concat', [zero, sliced], [shifted], axis=1, name=shifted),
                      h.make_node('Add', [current, shifted], [result], name=result)]
            current = result
        nodes.append(h.make_node('Identity', [current], list(original.output), name=original.name))
    else:
        raise ValueError(kind)
    return nodes, initializers


def rewrite_scans(model, kind, cold):
    checks, nodes = [], []
    for original in model.graph.node:
        if original.name not in [STEM+'CumSum', STEM+'CumSum_1']:
            nodes.append(copy.deepcopy(original))
            continue
        if original.op_type != 'CumSum' or original.attribute:
            raise ValueError('Expected default inclusive forward CumSum')
        axis_node = next(n for n in model.graph.node if original.input[1] in n.output)
        axis = nh.to_array(h.get_attribute_value(axis_node.attribute[0]))
        if axis != 1:
            raise ValueError('Unexpected scan dimension')
        replacement, initializers = scan_nodes(original, kind)
        check_graph = h.make_graph([copy.deepcopy(axis_node)] + replacement, 'routing_scan_check',
             [tensor(original.input[0], TP.INT32, [64,128])],
             [tensor(original.output[0], TP.INT32, [64,128])], initializers)
        check = ReferenceEvaluator(h.make_model(check_graph, opset_imports=[h.make_opsetid('',17)]))
        rng = np.random.default_rng(20260922)
        routes = np.fromfile(cold/'input_f16.bin', np.float16).reshape(128,2176)[:,2048:].T
        group = 0 if original.name == STEM+'CumSum' else 1
        inputs = [np.zeros((64,128), np.int32), np.ones((64,128), np.int32),
                  rng.integers(0,2,(64,128),dtype=np.int32),
                  (routes[group*64:(group+1)*64]>0).astype(np.int32)]
        for source in inputs:
            actual = check.run(None, {original.input[0]: source})[0]
            if not np.array_equal(actual, np.cumsum(source,axis=1,dtype=np.int32)):
                raise ValueError('Routing scan semantic check failed')
        checks.append(dict(node=original.name, kind=kind, exact_cpu_cases=len(inputs)))
        nodes.extend(replacement)
        model.graph.initializer.extend(initializers)
    if len(checks) != 2:
        raise ValueError('Expected two routing scans')
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return checks


def export(args, name):
    capacity, scan = config(name)
    out = args.out/name
    out.mkdir(exist_ok=False)
    source = args.overhead/'drop_zero_tile_t16/model.onnx'
    model = onnx.load(source, load_external_data=False)
    before = copy.deepcopy(model)
    for value in model.graph.initializer:
        if value.name in ['probe_stage1_end', 'probe_stage1_stop']:
            old = nh.to_array(value)
            if not np.all(old == 2):
                raise ValueError('Expected cold capacity 2 source')
            value.CopyFrom(nh.from_array(np.full_like(old,capacity),value.name))
    checks = rewrite_scans(model,scan,args.cold) if scan else []
    # All original nodes except the two scans must remain byte-identical.
    after_nodes = {n.name:n for n in model.graph.node}
    for node in before.graph.node:
        if scan and node.name in [STEM+'CumSum', STEM+'CumSum_1']:
            continue
        if node != after_nodes[node.name]:
            raise ValueError(f'Uncontrolled node change: {node.name}')
    after_init = {v.name:v for v in model.graph.initializer}
    for value in before.graph.initializer:
        if value.name not in ['probe_stage1_end', 'probe_stage1_stop'] and value != after_init[value.name]:
            raise ValueError(f'Uncontrolled initializer change: {value.name}')
    dependencies(out, source.parent)
    onnx.save(model,out/'model.onnx')
    onnx.checker.check_model(str(out/'model.onnx'))
    if name == 'c2' and sha(source) != sha(out/'model.onnx'):
        raise ValueError('Control graph changed')
    dump(out/'export.json',dict(case=name,source=str(source),source_sha256=sha(source),
        graph_sha256=sha(out/'model.onnx'),input_sha256=sha(args.cold/'input_f16.bin'),
        hot_capacity=128,cold_capacity=capacity,scan=scan,checks=checks,
        unchanged_trained_weights=True,unchanged_expert_and_combination_nodes=True))
    print('EXPORT_OK',name,flush=True)


def qpc(args,name):
    if name == 'c2':
        suffix = '_stats0' if args.stats_level == 0 else ''
        return args.overhead_scratch/f'drop_zero_tile_t16{suffix}_compile/qpc'
    return args.scratch/f'{name}_stats{args.stats_level}_compile/qpc'


def compile_case(args,name):
    case = args.out/name
    if name == 'c2':
        path = qpc(args,name)/'programqpc.bin'
        if not path.is_file():
            raise ValueError('Missing previous optimized control binary')
        dump(case/f'stats{args.stats_level}_reuse.json',dict(qpc=str(path),sha256=sha(path),
             identical_graph_sha256=sha(case/'model.onnx')))
        return
    target = qpc(args,name).parent
    try:
        compile_model(case/'model.onnx',target,stats_level=args.stats_level)
    finally:
        for filename in ['compile.log','compile.command.json','result.json']:
            if (target/filename).exists():
                shutil.copy2(target/filename,case/f'stats{args.stats_level}_{filename}')


def analyze(args):
    results = []
    for name in args.cases:
        case = args.out/name
        directory = case/'analysis'
        directory.mkdir(exist_ok=True)
        traces = sorted((case/'profile/trace').glob('*merged*.trace.json'))
        if len(traces) != 3 or not load(case/'profile/validation.json')['bit_exact_to_own_timing']:
            raise ValueError('Expected three validated captures')
        samples, work_all, p2p_all = [], [], []
        for index,path in enumerate(traces):
            metrics, _, work, p2p = overhead.trace_metrics(path)
            metrics['between_groups_ms'] = metrics['cold_start_ms']-metrics['hot_end_ms']
            for label, node in [('hot','CumSum'),('cold','CumSum_1')]:
                selected = [r for r in work if (r['node']==STEM+node or r['node'].startswith(STEM+node+'_retune_'))
                            and r['engine']=='HVX' and r['duration_us']>0 and r['kind']!='aicendcyclestats']
                if not selected:
                    raise ValueError('No routing scan events')
                busy = defaultdict(float)
                intervals = defaultdict(list)
                for r in selected:
                    busy[r['card'],r['core']] += r['duration_us']/1000
                    intervals[r['card'],r['core']].append((r['start_us'],r['end_us']))
                metrics[label+'_scan_cores'] = len(busy)
                metrics[label+'_scan_max_core_ms'] = max(busy.values())
                metrics[label+'_scan_max_core_span_ms'] = max(
                    (max(b for _,b in spans)-min(a for a,_ in spans))/1000 for spans in intervals.values())
                metrics[label+'_scan_events'] = len(selected)
                metrics[label+'_scan_start_ms'] = min(r['start_us'] for r in selected)/1000
                metrics[label+'_scan_end_ms'] = max(r['end_us'] for r in selected)/1000
            samples.append(dict(case=name,sample=index,trace=str(path),**metrics))
            work_all.extend(dict(sample=index,**r) for r in work)
            p2p_all.extend(dict(sample=index,**r) for r in p2p)
        save_csv(directory/'samples.csv',samples)
        save_csv(directory/'work.csv',work_all)
        save_csv(directory/'p2p.csv',p2p_all)
        chosen = sorted(samples,key=lambda s:s['device_ms'])[1]
        dump(directory/'summary.json',dict(chosen,samples=samples))
        results.append(dict(chosen,samples=samples))
        print('ANALYZE_OK',name,'device',chosen['device_ms'],'scan',chosen['hot_scan_max_core_ms'],
              chosen['cold_scan_max_core_ms'],'cores',chosen['hot_scan_cores'],flush=True)
    dump(args.out/f'{args.summary_name}.json',dict(cases=results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','compile','timing','timing-summary','profile','analyze'])
    for key in ['cold','overhead','out','scratch','base-scratch','overhead-scratch']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--stats-level',type=int,choices=[0,70],default=0)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--timing-prefix',default='timing')
    p.add_argument('--summary-name',default='timing_summary')
    args = p.parse_args()
    if args.rounds<1 or args.iterations<1 or len(set(args.cases))!=len(args.cases):
        p.error('Need positive timing counts and distinct cases')
    if args.action in ['profile','analyze'] and args.stats_level != 70:
        p.error('Profiling requires stats-level 70')
    for key in ['cold','overhead','out','scratch','base_scratch','overhead_scratch']:
        setattr(args,key,getattr(args,key).resolve())
    for name in args.cases:
        config(name)
    args.out.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    if args.action == 'timing':
        overhead.timing(args,qpc_fn=qpc)
    elif args.action == 'timing-summary':
        overhead.timing_summary(args)
    elif args.action == 'analyze':
        analyze(args)
    else:
        for name in args.cases:
            if args.action == 'profile':
                overhead.profile(args,name,qpc_fn=qpc)
            else:
                {'export':export,'compile':compile_case}[args.action](args,name)


if __name__ == '__main__':
    main()
