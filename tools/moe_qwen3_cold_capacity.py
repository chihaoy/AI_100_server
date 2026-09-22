#!/usr/bin/env python3
"""Controlled FP16 cold-capacity sweep using one trained Qwen3 MoE layer."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
from onnx.utils import Extractor

from moe_qwen3_baseline import EXPORT, REFERENCE
from moe_qwen3_oracle import CONFIG
from moe_qwen3_profile import events, canon, summarize_trace


SDK = Path('/opt/qti-aic')
CAPACITIES = [128, 64, 32, 16, 8, 4, 2]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, log):
    argv = list(map(str, argv))
    dump(log.with_suffix('.command.json'), argv)
    with log.open('x') as stream:
        subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)


def dependencies(directory, source):
    for name in ['weights', 'regrouped']:
        if (source / name).exists():
            (directory / name).symlink_to((source / name).resolve(), target_is_directory=True)


def tensor(name, dtype, shape):
    return h.make_tensor_value_info(name, dtype, shape)


def compile_model(graph, out, prefix=False, stats_level=70):
    out.mkdir(parents=True, exist_ok=False)
    cmd = [SDK / 'exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', f'-m={graph}',
           '-convert-to-fp16', '-aic-num-cores=16', '-mos=1', '-aic-enable-depth-first',
           f'-mdp-load-partition-config={CONFIG / "mdp_ts_4.json"}', '-compile-only',
           f'-aic-binary-dir={out / "qpc"}']
    if prefix:
        cmd += ['-retained-state', f'-network-specialization-config={CONFIG / "specializations_flat.json"}',
                f'-custom-IO-list-file={graph.parent / "custom_io.yaml"}']
    else:
        cmd += [f'-stats-level={stats_level}']
    started = time.monotonic()
    command(cmd, out / 'compile.log')
    if not (out / 'qpc/programqpc.bin').is_file():
        raise RuntimeError('Compiler produced no QPC')
    dump(out / 'result.json', dict(seconds=time.monotonic()-started, graph=str(graph),
                                  qpc_bytes=(out / 'qpc/programqpc.bin').stat().st_size,
                                  stats_level=0 if prefix else stats_level))
    print('COMPILE_OK', out, flush=True)


def prefix(args):
    out = args.out / 'prefix'
    out.mkdir(parents=True, exist_ok=False)
    source = args.oracle / 'min_fp16_graph'
    model = onnx.load(source / 'diagnostic.onnx', load_external_data=False)
    stem = f'/model/layers.{args.layer}/mlp/'
    x_name, route_name = stem + 'Reshape_output_0', f'oracle_L{args.layer}_routing'
    model.graph.node.extend([
        h.make_node('Identity', [x_name], ['probe_x'], name='probe_x'),
        h.make_node('Identity', [route_name], ['probe_routes'], name='probe_routes'),
    ])
    model.graph.value_info.extend([tensor('probe_x', TP.FLOAT16, [128, 2048]),
                                    tensor('probe_routes', TP.FLOAT16, [128, 128])])
    past = [f'past_{kind}.{layer}' for layer in range(args.layer+1) for kind in ['key', 'value']]
    subset = Extractor(model).extract_model(['input_ids', 'position_ids'] + past,
                    ['probe_x', 'probe_routes'] + [name + '_RetainedState' for name in past])
    dependencies(out, source)
    onnx.save(subset, out / 'model.onnx')
    onnx.checker.check_model(str(out / 'model.onnx'))
    names = past + [name+'_RetainedState' for name in past] + ['probe_x', 'probe_routes']
    (out / 'custom_io.yaml').write_text(''.join(f' - IOName: {name}\n   Precision: float16\n\n' for name in names))
    dump(out / 'export.json', dict(layer=args.layer, source=str(source / 'diagnostic.onnx'),
         source_sha256=sha(source / 'diagnostic.onnx'), nodes=len(subset.graph.node),
         inputs=[v.name for v in subset.graph.input], outputs=[v.name for v in subset.graph.output]))
    build = args.scratch / 'prefix_compile'
    compile_model(out / 'model.onnx', build, prefix=True)
    for name in ['compile.log', 'compile.command.json', 'result.json']:
        shutil.copy2(build / name, out / name)
    bindings = [dict(path=str(REFERENCE / f'{name}_i64.bin'), dims=[1,128],
                     **{'elem-size':8, 'io-direction':'in', 'map-to':name}) for name in ['input_ids','position_ids']]
    dump(out / 'io.json', {'IO-files':[bindings]})
    command([SDK / 'exec/qaic-runner', '-t', build / 'qpc', '-D', '0:1:2:3',
             '--aic-batch-json-input', out / 'io.json', '-n', 3, '-S', 1, '-T', 1,
             '--write-output-start-iter', 1, '--write-output-num-samples', 2,
             '--write-output-dir', out / 'outputs'], out / 'run.log')
    arrays = {}
    for name, shape in [('probe_x',(128,2048)), ('probe_routes',(128,128))]:
        files = [p for p in (out / 'outputs').rglob('*') if p.is_file() and p.stat().st_size == np.prod(shape)*2]
        if len(files)!=2 or files[0].read_bytes()!=files[1].read_bytes():
            raise ValueError(f'Missing or nonrepeatable capture {name}')
        arrays[name] = np.fromfile(files[0], np.float16).reshape(shape)
        if not np.isfinite(arrays[name]).all():
            raise ValueError('Nonfinite prefix outputs')
        arrays[name].tofile(out / f'{name}.bin')
    routes=arrays['probe_routes']
    if np.any(routes<0) or not np.all((routes>0).sum(1)==8):
        raise ValueError('Invalid captured top-8 routing')
    counts=(routes>0).sum(0).astype(np.int32)
    expected=np.fromfile(args.oracle / 'sorted128_fp16_run/counts_i32.bin',np.int32).reshape(48,128)[args.layer]
    data=np.concatenate([arrays['probe_x'],routes],axis=1)
    data.tofile(args.out/'input_f16.bin')
    counts.tofile(args.out/'expected_counts_i32.bin')
    report=dict(layer=args.layer, counts=counts.tolist(), required_capacities=counts.reshape(2,64).max(1).tolist(),
                matches_full_model_counts=bool(np.array_equal(counts,expected)),
                repeatable=True, input_sha256=sha(args.out/'input_f16.bin'),
                source_input_ids_sha256=sha(REFERENCE/'input_ids_i64.bin'))
    dump(out/'capture.json',report)
    if counts[64:].max()>min(args.capacities):
        raise ValueError('Captured cold routing exceeds the smallest planned capacity')
    print('PREFIX_CAPTURE_OK',json.dumps(report),flush=True)


def export_case(args, capacity):
    out=args.out/f'c{capacity}'
    out.mkdir(parents=True,exist_ok=False)
    source=args.oracle/'min_fp16_graph'
    model=onnx.load(source/'diagnostic.onnx',load_external_data=False)
    stem=f'/model/layers.{args.layer}/mlp/'
    x_name,route_name=stem+'Reshape_output_0',f'oracle_L{args.layer}_routing'
    output_name=stem+'Einsum_4_output_0'
    model.graph.value_info.extend([tensor(x_name,TP.FLOAT16,[128,2048]),
        tensor(route_name,TP.FLOAT16,[128,128]),tensor(output_name,TP.FLOAT16,[128,2048]),
        tensor(stem+'Einsum_1_output_0',TP.INT32,[64]),tensor(stem+'Einsum_2_output_0',TP.INT32,[64])])
    subset=Extractor(model).extract_model([x_name,route_name],
                [output_name,stem+'Einsum_1_output_0',stem+'Einsum_2_output_0'])
    # A shared single-input ABI allows the existing checked-size four-card host.
    # Only the cold Slice end and mask Range limit vary across capacity cases.
    inits=[nh.from_array(np.array(value,np.int64),name) for name,value in
           [('probe_axis',[1]),('probe_zero',[0]),('probe_hidden',[2048]),('probe_end',[2176])]]
    nodes=[h.make_node('Slice',['probe_input','probe_zero','probe_hidden','probe_axis'],[x_name],name='probe_input_x'),
           h.make_node('Slice',['probe_input','probe_hidden','probe_end','probe_axis'],[route_name],name='probe_input_routes')]
    nodes.extend(copy.deepcopy(n) for n in subset.graph.node)
    nodes.extend([h.make_node('Identity',[output_name],['y'],name='probe_output'),
                  h.make_node('Concat',[stem+'Einsum_1_output_0',stem+'Einsum_2_output_0'],['counts'],axis=0,name='probe_counts')])
    by_name={n.name:n for n in nodes}
    for stage,cap in enumerate([128,capacity]):
        slice_name=stem+('Slice' if stage==0 else 'Slice_1')
        range_name=stem+('Range_1' if stage==0 else 'Range_3')
        end_name=f'probe_stage{stage}_end';stop_name=f'probe_stage{stage}_stop'
        inits.extend([nh.from_array(np.array([cap],np.int64),end_name),nh.from_array(np.array(cap,np.int32),stop_name)])
        by_name[slice_name].input[2]=end_name
        by_name[range_name].input[1]=stop_name
    # Remove unused old capacity constants so a structural comparison has only
    # the two intended scalar differences between cases.
    inits.extend(copy.deepcopy(v) for v in subset.graph.initializer)
    used={v for n in nodes for v in n.input}
    inits=[v for v in inits if v.name in used]
    graph=h.make_graph(nodes,'cold_capacity_control', [tensor('probe_input',TP.FLOAT16,[128,2176])],
                       [tensor('y',TP.FLOAT16,[128,2048]),tensor('counts',TP.INT32,[128])],inits)
    result=h.make_model(graph,opset_imports=list(subset.opset_import),functions=list(subset.functions),ir_version=subset.ir_version)
    dependencies(out,source)
    onnx.save(result,out/'model.onnx')
    onnx.checker.check_model(str(out/'model.onnx'))
    weights={v.name:dict(shape=list(v.dims),external={e.key:e.value for e in v.external_data})
             for v in inits if v.external_data}
    if len(weights)!=6:
        raise ValueError(f'Expected six trained expert banks, got {len(weights)}')
    dump(out/'export.json',dict(layer=args.layer,hot_capacity=128,cold_capacity=capacity,
         input_sha256=sha(args.out/'input_f16.bin'),weights=weights,nodes=len(nodes)))
    return out


def verify_graphs(args):
    """Reject any graph difference beyond the two cold-capacity constants."""
    normalized, weights = [], []
    for capacity in args.capacities:
        directory = args.out / f'c{capacity}'
        model = onnx.load(directory / 'model.onnx', load_external_data=False)
        for value in model.graph.initializer:
            if value.name in ['probe_stage1_end', 'probe_stage1_stop']:
                array = nh.to_array(value)
                if not np.all(array == capacity):
                    raise ValueError('Incorrect capacity constant')
                value.CopyFrom(nh.from_array(np.full_like(array, 128), value.name))
        normalized.append(hashlib.sha256(model.SerializeToString()).hexdigest())
        metadata = load(directory / 'export.json')
        weights.append(metadata['weights'])
        if metadata['input_sha256'] != sha(args.out / 'input_f16.bin'):
            raise ValueError('Input changed after export')
    if len(set(normalized)) != 1 or any(w != weights[0] for w in weights):
        raise ValueError('Uncontrolled graph/weight difference')
    dump(args.out / 'graph_validation.json', dict(capacities=args.capacities,
        identical_except_cold_constants=True, normalized_graph_sha256=normalized[0],
        identical_weight_references=True, input_sha256=sha(args.out / 'input_f16.bin'),
        partition_sha256=sha(CONFIG / 'mdp_ts_4.json')))


def resources(path):
    result = subprocess.run([str(SDK / 'tools/qaic-util'), '-q'],
                            check=True, text=True, stdout=subprocess.PIPE).stdout
    path.write_text(result)
    blocks = re.split(r'(?m)^QID ', result)[1:]
    if len(blocks) != 4:
        raise ValueError('Expected four devices')
    for block in blocks:
        for key, value in [('Status', 'Ready'), ('Nsp Free', '16'),
                           ('Networks Loaded', '0'), ('Networks Active', '0')]:
            if not re.search(rf'(?m)^\s*{key}:\s*{value}\s*$', block):
                raise ValueError(f'Device unavailable or leaked resources: {key}')


def validate_output(args, directory, capacity):
    y = np.fromfile(directory / 'y.bin', np.float16)
    counts = np.fromfile(directory / 'counts.bin', np.int32)
    expected = np.fromfile(args.out / 'expected_counts_i32.bin', np.int32)
    if y.size != 128*2048 or not np.isfinite(y).all():
        raise ValueError('Invalid MoE output')
    if not np.array_equal(counts, expected) or counts.sum() != 1024:
        raise ValueError('Routing mismatch')
    if counts[:64].max() > 128 or counts[64:].max() > capacity:
        raise ValueError('Capacity overflow')
    reference = args.out / 'c128/timing_r0/y.bin'
    if not reference.exists():
        raise ValueError('Run C128 first')
    ref = np.fromfile(reference, np.float16).astype(np.float64)
    delta = y.astype(np.float64) - ref
    report = dict(counts_exact=True, no_overflow=True,
                  reference=str(reference), bit_exact=bool(np.array_equal(y, ref)),
                  max_abs=float(np.abs(delta).max()),
                  relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-30)))
    dump(directory / 'validation.json', report)
    # Shape changes may alter FP16 reduction order. Reject material drift;
    # report even small differences rather than silently treating them as exact.
    if report['relative_l2'] > 0.001:
        raise ValueError(f'Unexpected output drift: {report}')


def run_timings(args):
    verify_graphs(args)
    host = args.scratch / 'moe_single_host'
    if not host.exists():
        command(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                 f'-I{SDK / "dev/inc"}', Path(__file__).with_name('moe_single_qpc_host.cpp'),
                 '-o', host, f'-L{SDK / "dev/lib/x86_64"}', '-lQAic',
                 f'-Wl,-rpath,{SDK / "dev/lib/x86_64"}'], args.out / 'host_build.log')
    for round_index in range(args.rounds):
        order = args.capacities if round_index % 2 == 0 else list(reversed(args.capacities))
        for capacity in order:
            case = args.out / f'c{capacity}'
            out = case / f'timing_r{round_index}'
            if out.exists():
                raise FileExistsError(out)
            resources(case / f'resources_before_r{round_index}.txt')
            command([host, args.scratch / f'c{capacity}_compile/qpc',
                     args.out / 'input_f16.bin', out, args.iterations, 10],
                    case / f'timing_r{round_index}.log')
            resources(case / f'resources_after_r{round_index}.txt')
            validate_output(args, out, capacity)
            if round_index and (out / 'y.bin').read_bytes() != (case / 'timing_r0/y.bin').read_bytes():
                raise ValueError('Output changed between timing rounds')
            timing = load(out / 'timing.json')
            print('TIMING_OK', capacity, round_index,
                  float(np.median(timing['samples_ms'])), flush=True)


def profile_case(args, capacity):
    case = args.out / f'c{capacity}'
    out = case / 'profile'
    out.mkdir(exist_ok=False)
    expected = case / 'timing_r0'
    qpc = args.scratch / f'c{capacity}_compile/qpc'
    bindings = []
    for name, path, dims, size, direction in [
        ('probe_input', args.out / 'input_f16.bin', [128,2176], 2, 'in'),
        ('y', expected / 'y.bin', [128,2048], 2, 'out'),
        ('counts', expected / 'counts.bin', [128], 4, 'out')]:
        if path.stat().st_size != int(np.prod(dims)) * size:
            raise ValueError(f'Invalid buffer size: {path}')
        bindings.append({'path': str(path), 'dims': dims, 'elem-size': size,
                         'io-direction': direction, 'map-to': name})
    dump(out / 'io.json', {'IO-files': [bindings]})
    for name in ['stats','outputs','trace']:
        (out / name).mkdir()
    resources(out / 'resources_before.txt')
    command([SDK / 'exec/qaic-runner', '-t', qpc, '-D', '0:1:2:3',
             '--aic-batch-json-input', out / 'io.json', '-n', 5, '-S', 1, '-T', 1, '-c',
             '--aic-profiling-type', 'raw_device_stats', '--aic-profiling-start-iter', 2,
             '--aic-profiling-num-samples', 3, '--aic-profiling-out-dir', out / 'stats',
             '--write-output-start-iter', 2, '--write-output-num-samples', 3,
             '--write-output-dir', out / 'outputs'], out / 'runner.log')
    resources(out / 'resources_after.txt')
    checked = defaultdict(list)
    for path in (out / 'outputs').rglob('*'):
        if not path.is_file():
            continue
        name = {524288:'y', 512:'counts'}.get(path.stat().st_size)
        if name:
            if path.read_bytes() != (expected / f'{name}.bin').read_bytes():
                raise ValueError(f'Profile output mismatch: {path}')
            checked[name].append(str(path))
    if any(len(checked[name]) != 3 for name in ['y','counts']):
        raise ValueError('Missing profile outputs')
    command([SDK / 'exec/qaic-opstats', '--qpc', qpc / 'programqpc.bin',
             '--input-dir', out / 'stats', '--output-dir', out / 'trace', '--summary',
             '--trace', '--merge-mq-traces', 'true', '--flow-events', 'none'], out / 'opstats.log')
    traces = sorted((out / 'trace').glob('*merged*.trace.json'))
    if len(traces) != 3:
        raise ValueError('Missing merged traces')
    dump(out / 'validation.json', dict(bit_exact=True, samples=3, files=dict(checked)))
    print('PROFILE_OK', capacity, flush=True)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_trace(path, layer):
    rows, execution_ms = summarize_trace(path)
    stem = f'/model/layers.{layer}/mlp/'
    names = {stage: {stem + name for name in banks} for stage, banks in
             [('hot', ['MatMul','MatMul_1','MatMul_2']),
              ('cold', ['MatMul_3','MatMul_4','MatMul_5'])]}
    result = dict(device_ms=execution_ms)
    bounds, coverage = {}, {}
    for stage in names:
        selected = [r for r in rows if r['node'] in names[stage] and
                    r['engine']=='HMX' and r['kind']=='aicconvolutiond32']
        if {r['node'] for r in selected} != names[stage]:
            raise ValueError(f'Missing {stage} GEMMs')
        bounds[stage] = (min(r['first_us'] for r in selected),
                         max(r['last_us'] for r in selected))
        busy, counts = defaultdict(float), defaultdict(int)
        for row in selected:
            key = row['card'], row['core']
            busy[key] += row['busy_us']
            counts[key] += row['events']
        coverage[stage] = sorted(busy)
        result.update({f'{stage}_span_ms': (bounds[stage][1]-bounds[stage][0])/1000,
                       f'{stage}_hmx_busy_max_core_ms': max(busy.values())/1000,
                       f'{stage}_hmx_busy_median_core_ms': float(np.median(list(busy.values())))/1000,
                       f'{stage}_hmx_events_total': sum(counts.values()),
                       f'{stage}_hmx_events_min_core': min(counts.values()),
                       f'{stage}_hmx_events_max_core': max(counts.values()),
                       f'{stage}_cores': len(busy)})
    reduction = [r for r in rows if r['node']==stem+'Einsum_3' and
                 r['engine']=='HVX' and r['kind']=='aicbatchedreduceadd']
    combine = [r for r in rows if r['node']==stem+'Einsum_4' and
               r['engine'] in ['HMX','HVX'] and r['kind']!='aicendcyclestats']
    if not reduction or not combine:
        raise ValueError('Missing final reduction')
    result['between_groups_ms'] = (bounds['cold'][0]-bounds['hot'][1])/1000
    result['last_unpack_ms'] = (min(r['first_us'] for r in reduction)-bounds['cold'][1])/1000
    result['combine_ms'] = (max(r['last_us'] for r in combine)-min(r['first_us'] for r in reduction))/1000
    if any(result[k] < 0 for k in ['between_groups_ms','last_unpack_ms','combine_ms']):
        raise ValueError('Overlapping phase boundaries')
    threads = {}
    for event in events(path, metadata_only=True):
        if event.get('name')=='thread_name':
            threads[event['pid'],event['tid']] = event['args']['name']
    syncs, detail, kernels = defaultdict(list), [], defaultdict(lambda: [0,0.,0.,set()])
    pending_hmx, wait_matches, max_wait_boundary_error = {}, 0, 0.
    for event in events(path):
        if event.get('ph')!='X':
            continue
        thread = threads.get((event['pid'],event['tid']), '')
        match = re.search(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$', thread)
        if not match:
            continue
        card, core, engine = int(match[1]), int(match[2]), match[3] or 'execution'
        arg = event.get('args', {})
        node = canon(arg.get('opName',''))
        stage = next((s for s in names if node in names[s]), None)
        if stage is None:
            continue
        start = float(event['ts']); duration = float(event.get('dur',0))
        sync = float(arg.get('opSyncDurUs',duration)) if event['name'].startswith('sync ') else 0.
        kind, memory = arg.get('opKind',''), arg.get('opMemory','')
        key = stage, card, core, engine, node, event['name'], kind, memory
        row = kernels[key]
        row[0] += 1; row[1] += duration; row[2] += sync
        row[3].add(arg.get('opName',''))
        if card==0 and core==0:
            detail.append(dict(stage=stage,engine=engine,node=node,name=event['name'],
                               kind=kind,memory=memory,start_us=start,duration_us=duration,
                               sync_us=sync,lowered_op=arg.get('opName','')))
        if engine=='HMX' and kind=='aicconvolutiond32':
            op_key = card,core,arg.get('oc'),arg.get('opName')
            if event['name']=='sync HMX':
                pending_hmx[op_key] = start+sync
                lo,hi=bounds[stage]
                syncs[stage,card,core].append((max(start,lo),min(start+sync,hi)))
            elif event['name']=='aicconvolutiond32':
                if op_key in pending_hmx:
                    error = abs(pending_hmx.pop(op_key)-start)
                    max_wait_boundary_error = max(max_wait_boundary_error,error)
                    # Allow sub-microsecond timestamp quantization/dispatch gaps,
                    # and retain the actual maximum discrepancy in the report.
                    if error>1.0:
                        raise ValueError('HMX sync timestamp does not lead to its compute event')
                    wait_matches += 1
    for stage in names:
        sums = []
        for card,core in coverage[stage]:
            intervals = sorted((a,b) for a,b in syncs[stage,card,core] if b>a)
            if any(b>c+0.01 for (_,b),(c,_) in zip(intervals,intervals[1:])):
                raise ValueError('Overlapping waits on the same HMX thread')
            sums.append(sum(b-a for a,b in intervals))
        result[f'{stage}_hmx_wait_clipped_median_core_ms'] = float(np.median(sums))/1000
        result[f'{stage}_hmx_wait_clipped_max_core_ms'] = max(sums)/1000
    if wait_matches==0 or pending_hmx:
        raise ValueError('Missing paired HMX wait/compute events')
    result['paired_hmx_wait_events'] = wait_matches
    result['max_wait_boundary_error_us'] = max_wait_boundary_error
    kernel_rows=[]
    for key, values in kernels.items():
        stage,card,core,engine,node,name,kind,memory = key
        kernel_rows.append(dict(stage=stage,card=card,core=core,engine=engine,node=node,
            name=name,kind=kind,memory=memory,events=values[0],duration_us=values[1],
            sync_us=values[2],lowered_ops=len(values[3])))
    return result, rows, kernel_rows, detail, coverage


def analyze(args):
    verify_graphs(args)
    all_samples, summaries = [], []
    for capacity in args.capacities:
        case = args.out/f'c{capacity}'
        out = case/'analysis'
        out.mkdir(exist_ok=False)
        rounds = [load(case/f'timing_r{r}/timing.json') for r in range(args.rounds)]
        host = [s for run in rounds for s in run['samples_ms']]
        if len(host) != args.rounds*args.iterations:
            raise ValueError('Unexpected timing sample count')
        current, all_nodes, all_kernels, coverages = [], [], [], []
        traces = sorted((case/'profile/trace').glob('*merged*.trace.json*'))
        if len(traces)!=3 or not load(case/'profile/validation.json')['bit_exact']:
            raise ValueError('Incomplete profile validation')
        for sample,path in enumerate(traces):
            metrics,nodes,kernels,detail,coverage = analyze_trace(path,args.layer)
            current.append(dict(capacity=capacity,sample=sample,**metrics))
            all_nodes.extend(dict(sample=sample,**r) for r in nodes)
            all_kernels.extend(dict(sample=sample,**r) for r in kernels)
            save_csv(out/f'card0_core0_sample{sample}.csv',detail)
            coverages.append(coverage)
        if any(c!=coverages[0] for c in coverages):
            raise ValueError('GEMM core coverage changed between samples')
        chosen = sorted(current,key=lambda r:r['device_ms'])[1]
        summary = dict(chosen, host_median_ms=float(np.median(host)),
            host_p10_ms=float(np.percentile(host,10)),host_p90_ms=float(np.percentile(host,90)),
            host_round_medians_ms=[float(np.median(r['samples_ms'])) for r in rounds],
            host_samples=len(host), profile_samples=3,
            profile_device_min_ms=min(r['device_ms'] for r in current),
            profile_device_max_ms=max(r['device_ms'] for r in current),
            output_validation=load(case/'timing_r0/validation.json'),coverage=coverages[0])
        dump(out/'summary.json',summary)
        save_csv(out/'samples.csv',current)
        save_csv(out/'nodes.csv',all_nodes)
        save_csv(out/'kernels.csv',all_kernels)
        all_samples.extend(current);summaries.append(summary)
        for name in ['compile.log','compile.command.json','result.json']:
            shutil.copy2(args.scratch/f'c{capacity}_compile'/name,case/name)
        print('ANALYSIS_OK',capacity,json.dumps(chosen),flush=True)
    save_csv(args.out/'samples.csv',all_samples)
    dump(args.out/'summary.json',dict(layer=args.layer,precision='fp16',hot_capacity=128,
        stats_level=70,capture=load(args.out/'prefix/capture.json'),cases=summaries))


def package_audit(args):
    result = {}
    for capacity in args.capacities:
        dest = args.scratch/f'c{capacity}_segments'
        if not dest.exists():
            command([SDK/'tools/qaic-qpc','extract','--qpc',
                     args.scratch/f'c{capacity}_compile/qpc/programqpc.bin',
                     '--output-dir',dest,'-s','*StaticConstants.constants.bin'],
                    args.out/f'c{capacity}'/'package_extract.log')
        constants = sorted(dest.rglob('StaticConstants.constants.bin'))
        if len(constants)!=4:
            raise ValueError('Expected four card constant segments')
        result[str(capacity)] = {str(p.relative_to(dest)):
                                dict(bytes=p.stat().st_size,sha256=sha(p)) for p in constants}
    dump(args.out/'package_audit.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prefix','export','verify','compile','timing','profile','analyze','package'])
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--layer',type=int,default=2)
    parser.add_argument('--capacities',type=int,nargs='+',default=CAPACITIES)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    if (not 0<=args.layer<48 or args.rounds<1 or args.iterations<1 or
            len(set(args.capacities))!=len(args.capacities) or
            any(c<1 or c>128 for c in args.capacities)):
        parser.error('Require layer 0–47, positive rounds/iterations, distinct capacities 1–128')
    args.oracle,args.out,args.scratch=args.oracle.resolve(),args.out.resolve(),args.scratch.resolve()
    args.out.mkdir(parents=True,exist_ok=True);args.scratch.mkdir(parents=True,exist_ok=True)
    if args.action=='prefix':prefix(args)
    elif args.action=='export':
        for capacity in args.capacities:export_case(args,capacity)
    elif args.action=='verify':verify_graphs(args)
    elif args.action=='compile':
        verify_graphs(args)
        for capacity in args.capacities:
            compile_model(args.out/f'c{capacity}/model.onnx',args.scratch/f'c{capacity}_compile')
    elif args.action=='timing':run_timings(args)
    elif args.action=='profile':
        for capacity in args.capacities:profile_case(args,capacity)
    elif args.action=='analyze':analyze(args)
    elif args.action=='package':package_audit(args)


if __name__=='__main__':
    main()
