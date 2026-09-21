#!/usr/bin/env python3
"""Select MoE group topology using a shape-resolved ONNX If in one resident QPC.

Branches reuse the validated width sweep's graphs and share one source bank.
Tag values are zero; tag length selects one of six compiled widths. This is
ahead-of-time shape specialization, not a data-dependent runtime branch.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from onnx.utils import Extractor
from onnx.reference import ReferenceEvaluator
from moe_capacity_audit import CtxGather3D, CtxScatter3DInt
from moe_resident_selection import CASES, command, compare, dump
from moe_resident_audit import trace_summary

SDK = Path('/opt/qti-aic')
WIDTHS = [1, 2, 4, 8, 16, 32]
COMPILE_WIDTHS = {'multi': WIDTHS, **{f'single_w{w}': [w] for w in WIDTHS},
                 'pair_w4_w8': [4,8], 'triple_w2_w4_w8': [2,4,8]}


def vi(name, dtype, shape):
    return h.make_tensor_value_info(name, dtype, shape)


def frontend(a):
    """Reproduce the branch-condition and branch-output restrictions cheaply."""
    base = a.out/'frontend'
    base.mkdir(parents=True, exist_ok=False)
    report = {}
    for kind in ['shape_one_output', 'shape_two_outputs', 'value_one_output']:
        dest = base/kind
        dest.mkdir()
        then_nodes = [h.make_node('MatMul', ['z','w'], ['a'])]
        else_nodes = [h.make_node('MatMul', ['z','w'], ['b']), h.make_node('MatMul', ['b','w'], ['c'])]
        then_outputs = [vi('a',T.FLOAT16,[4,4])]
        else_outputs = [vi('c',T.FLOAT16,[4,4])]
        output_names = ['y']
        if kind == 'shape_two_outputs':
            then_nodes.append(h.make_node('Identity',['a'],['a_copy']))
            else_nodes.append(h.make_node('Identity',['c'],['c_copy']))
            then_outputs.append(vi('a_copy',T.FLOAT16,[4,4]))
            else_outputs.append(vi('c_copy',T.FLOAT16,[4,4]))
            output_names.append('y_copy')
        nodes = [h.make_node('Shape',['tag'],['shape']), h.make_node('Gather',['shape','zero'],['length'],axis=0),
                 h.make_node('ReduceSum',['tag'],['sum'],keepdims=0), h.make_node('Cast',['sum'],['offset'],to=T.FLOAT16),
                 h.make_node('Add',['x','offset'],['z'])]
        if kind == 'value_one_output':
            nodes += [h.make_node('Gather',['tag','zero'],['value'],axis=0), h.make_node('Cast',['value'],['selector'],to=T.INT64)]
        else:
            nodes.append(h.make_node('Identity',['length'],['selector']))
        nodes += [h.make_node('Equal',['selector','one'],['condition']),
                  h.make_node('If',['condition'],output_names,
                              then_branch=h.make_graph(then_nodes,'once',[],then_outputs),
                              else_branch=h.make_graph(else_nodes,'twice',[],else_outputs))]
        graph = h.make_graph(nodes,kind,[vi('x',T.FLOAT16,[4,4]),vi('tag',T.INT32,['W'])],
            [vi(n,T.FLOAT16,[4,4]) for n in output_names],
            [nh.from_array(np.eye(4,dtype=np.float16)*2,'w'),nh.from_array(np.array(0,np.int64),'zero'),
             nh.from_array(np.array(1,np.int64),'one')])
        model = h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=9)
        onnx.checker.check_model(model)
        onnx.save(model,dest/'model.onnx')
        dump(dest/'specs.json',{'specializations':[{'W':'1'},{'W':'2'}]})
        status = command([SDK/'exec/qaic-compile','-aic-hw','-aic-hw-version=ai100',f'-m={dest/"model.onnx"}',
                          '-convert-to-fp16','-aic-num-cores=16','-mos=1','-aic-enable-depth-first','-compile-only',
                          f'-network-specialization-config={dest/"specs.json"}',f'-aic-binary-dir={dest/"qpc"}'],dest/'compile.log')
        report[kind] = {'exit_code':status,'qpc_exists':(dest/'qpc/programqpc.bin').exists(),
                        'log_tail':(dest/'compile.log').read_text().splitlines()[-5:]}
        dump(base/'result.json',report)
    print(json.dumps(report,indent=2),flush=True)


def export(a):
    source = a.source.resolve()
    original = json.loads((source/'info.json').read_text())
    if set(original['widths']) != set(WIDTHS) or original['capacity'] != 32:
        raise RuntimeError('Expected the six-width, capacity-32 sweep')
    root = a.out
    root.mkdir(parents=True, exist_ok=False)
    for name in ['orders', 'reference']:
        shutil.copytree(source/name, root/name)
    shutil.copyfile(source/'x.bin', root/'x.bin')
    branches, bank, hashes, functions, graph_audit = {}, {}, {}, {}, {}
    router_nodes = []
    H = original['H']
    for width in WIDTHS:
        model = onnx.load(source/f'w{width}/model.onnx')
        cut = '/ScatterElements_output_0'
        mapping = {'x': 'x', 'expert_ids': 'selected_ids', cut: 'route_weights'}
        for init in model.graph.initializer:
            name = init.name if init.name in ['Wg', 'Wu', 'Wd'] else 'router_weight'
            values = nh.to_array(init)
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            if name in hashes and hashes[name] != digest:
                raise RuntimeError('Width controls do not share identical source weights')
            hashes[name] = digest
            mapping[init.name] = name
            if name not in bank:
                bank[name] = copy.deepcopy(init)
                bank[name].name = name
        model.graph.value_info.append(vi(cut, T.FLOAT16, [128,128]))
        extractor = Extractor(model)
        if not router_nodes:
            router = extractor.extract_model(['x'], [cut, 'counts'])
            rm = dict(mapping)
            for node in router.graph.node:
                for out in node.output:
                    rm[out] = 'route_weights' if out == cut else ('counts' if out == 'counts' else 'router/'+out)
            for node in router.graph.node:
                new = copy.deepcopy(node)
                new.name = 'shared_router'+node.name
                new.input[:] = [rm.get(n,n) for n in new.input]
                new.output[:] = [rm[n] for n in new.output]
                router_nodes.append(new)
        model = extractor.extract_model(['x','expert_ids',cut], ['y'])
        for node in model.graph.node:
            for out in node.output:
                mapping[out] = f'w{width}/{out}'
        nodes = []
        weight_gathers, matmuls = [], []
        for node in model.graph.node:
            new = copy.deepcopy(node)
            new.name = f'topology_w{width}'+node.name
            for i, name in enumerate(new.input):
                new.input[i] = mapping.get(name, name)
            for i, name in enumerate(new.output):
                new.output[i] = mapping[name]
            if node.op_type == 'Gather' and new.input[0] in ['Wg', 'Wu', 'Wd']:
                weight_gathers.append(new.name)
            if node.op_type == 'MatMul' and node.name != '/MatMul':
                matmuls.append(new.name)
            nodes.append(new)
        assert len(weight_gathers) == len(matmuls) == 3*32//width
        branches[width] = h.make_graph(nodes, f'width_{width}', [],
            [vi(mapping['y'], T.FLOAT16, [128, H])])
        graph_audit[str(width)] = {'groups': 32//width, 'weight_gathers': weight_gathers,
                                  'expert_matmuls': matmuls, 'packed_shape': [width, 32, H]}
        for f in model.functions:
            key = (f.domain, f.name)
            if key in functions and functions[key].SerializeToString() != f.SerializeToString():
                raise RuntimeError('Different custom function definitions across widths')
            functions[key] = copy.deepcopy(f)
    exported_hashes = dict(zip(['gate', 'up', 'down'], [hashes[n] for n in ['Wg', 'Wu', 'Wd']]))
    assert exported_hashes == original['bank_sha256']
    nodes = [h.make_node('ReduceSum', ['layout_tag'], ['tag_sum'], name='tag_sum', keepdims=0),
             h.make_node('Add', ['expert_ids', 'tag_sum'], ['selected_ids'], name='selected_ids'),
             h.make_node('Shape', ['layout_tag'], ['layout_shape'], name='layout_shape'),
             h.make_node('Gather', ['layout_shape', 'zero'], ['layout_width'], name='layout_width', axis=0)]+router_nodes
    constants = list(bank.values())+[nh.from_array(np.array(0, np.int64), 'zero')]
    for width in WIDTHS[:-1]:
        constants.append(nh.from_array(np.array(width, np.int64), f'width_{width}'))
        nodes.append(h.make_node('Equal', ['layout_width', f'width_{width}'], [f'is_w{width}'], name=f'is_w{width}'))
    def branch(index):
        width = WIDTHS[index]
        if index == len(WIDTHS)-1:
            return branches[width]
        return h.make_graph([h.make_node('If', [f'is_w{width}'], [f'dispatch_{width}_y'],
                            name=f'dispatch_w{width}', then_branch=branches[width], else_branch=branch(index+1))],
                            f'dispatch_from_{width}', [], [vi(f'dispatch_{width}_y', T.FLOAT16, [128,H])])
    dispatch = branch(0).node[0]
    dispatch.output[:] = ['y']
    nodes.append(dispatch)
    graph = h.make_graph(nodes, 'topology_specialization',
        [vi('x', T.FLOAT16, [128,H]), vi('expert_ids', T.INT32, [32]), vi('layout_tag', T.INT32, ['W'])],
        [vi('y', T.FLOAT16, [128,H]), vi('counts', T.INT32, [32])], constants)
    model = h.make_model(graph, opset_imports=list(model.opset_import), functions=list(functions.values()), ir_version=9)
    onnx.checker.check_model(model)
    onnx.save_model(model, root/'model.onnx', save_as_external_data=True,
                    all_tensors_to_one_file=True, location='weights.bin', size_threshold=1024)
    onnx.checker.check_model(str(root/'model.onnx'))
    for variant, widths in COMPILE_WIDTHS.items():
        dump(root/f'{variant}_specializations.json', {'specializations': [{'W': str(w)} for w in widths]})
    dump(root/'info.json', {**original, 'source': str(source), 'widths': WIDTHS,
         'tag_values': 'zero', 'scope': 'Single-card shape-specialized topology; synthetic 32/128 partial MoE; no KV',
         'source_initializer_sha256': hashes})
    dump(root/'graph_audit.json', graph_audit)
    print('EXPORT_OK', root, flush=True)


def compile_graph(a):
    base = a.out/a.precision
    base.mkdir(exist_ok=True)
    info = json.loads((a.out/'info.json').read_text())
    result = json.loads((base/'compilation.json').read_text()) if (base/'compilation.json').exists() else {}
    for variant in a.variants:
        dest = base/variant
        if dest.exists():
            raise RuntimeError(f'Will not overwrite {dest}')
        spec = a.out/f'{variant}_specializations.json'
        if not spec.exists():
            dump(spec, {'specializations': [{'W': str(w)} for w in COMPILE_WIDTHS[variant]]})
        args = [SDK/'exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', f'-m={a.out/"model.onnx"}',
                '-convert-to-fp16', '-aic-num-cores=16', '-mos=1', '-aic-enable-depth-first', '-stats-level=70',
                '-compile-only', f'-network-specialization-config={a.out/f"{variant}_specializations.json"}',
                f'-aic-binary-dir={dest}']
        if a.precision == 'mxfp6':
            args.append('-mxfp6-matmul')
        status = command(args, base/f'{variant}_compile.log')
        result[variant] = {'exit_code': status, 'qpc_exists': (dest/'programqpc.bin').exists()}
        dump(base/'compilation.json', result)
    for w in WIDTHS:
        link = base/f'fixed_w{w}'
        if not link.exists():
            link.symlink_to(Path(info['source'])/a.precision/f'w{w}', target_is_directory=True)
    if not all(v['exit_code'] == 0 and v['qpc_exists'] for v in result.values()):
        raise RuntimeError('Compilation failed; inspect logs')


def run(a):
    base = a.out/a.precision/a.run_name
    base.mkdir(exist_ok=False)
    host = base/'host'
    if command(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror', f'-I{SDK}/dev/inc',
                Path(__file__).with_name('moe_topology_host.cpp'), '-o', host,
                f'-L{SDK}/dev/lib/x86_64', '-lQAic', f'-Wl,-rpath,{SDK}/dev/lib/x86_64'], base/'host_build.log'):
        raise RuntimeError('Host build failed')
    info = json.loads((a.out/'info.json').read_text())
    for variant in a.run_variants:
        if command([host, a.out/a.precision/variant, a.out, base/variant, a.device, info['H'], a.iterations, variant],
                   base/f'{variant}_host.log'):
            raise RuntimeError(f'Host failed: {variant}')


def analyze(a):
    info = json.loads((a.out/'info.json').read_text())
    base = a.out/a.precision/a.run_name
    counts = np.load(a.out/'reference/counts.npy')
    result = {'scope': info['scope'], 'precision': a.precision, 'variants': {}}
    for variant in a.run_variants:
        dest = base/variant
        with (dest/'samples.csv').open() as f:
            rows = list(csv.DictReader(f))
        with (dest/'resources.csv').open() as f:
            resources = list(csv.DictReader(f))
        vr = {'metadata': json.loads((dest/'metadata.json').read_text()), 'widths': {}, 'resources': resources,
              'activation_dram_delta_mib': (int(resources[0]['dram_free_kib'])-int(resources[1]['dram_free_kib']))/1024,
              'resources_stable': len({(r['dram_free_kib'], r['nsp_free']) for r in resources[1:]}) == 1}
        widths = WIDTHS if variant == 'multi' else [int(variant.split('_w')[1])]
        for width in widths:
            wr = {'cases': {}, 'latency': {}}
            for case in CASES:
                y = np.fromfile(dest/f'w{width}_{case}_y.bin', np.float16)
                actual_counts = np.fromfile(dest/f'w{width}_{case}_counts.bin', np.int32)
                control = base/f'fixed_w{width}'/f'w{width}_{case}_y.bin'
                if not control.exists():
                    raise RuntimeError(f'Missing same-width control: {control}')
                wr['cases'][case] = {'vs_cpu': compare(y, np.load(a.out/'reference'/f'{case}.npy').reshape(-1)),
                    'vs_fixed': compare(y, np.fromfile(control, np.float16)), 'counts_exact': bool(np.array_equal(counts, actual_counts))}
            for mode in sorted(set(r['mode'] for r in rows)):
                selected = [r for r in rows if int(r['width']) == width and r['mode'] == mode]
                wr['latency'][mode] = {'samples': len(selected)} | {
                    k: {'median_ms': float(np.median([float(r[k]) for r in selected])),
                        'p10_ms': float(np.percentile([float(r[k]) for r in selected], 10)),
                        'p90_ms': float(np.percentile([float(r[k]) for r in selected], 90))}
                    for k in ['total_ms', 'bind_ms', 'execute_ms']}
            identity = np.fromfile(dest/f'w{width}_identity_y.bin', np.float16)
            duplicate = np.fromfile(dest/f'w{width}_duplicate_control_y.bin', np.float16)
            wr['duplicate_changes_output'] = compare(duplicate, identity)['relative_l2'] > 0.1
            vr['widths'][str(width)] = wr
        result['variants'][variant] = vr
    all_widths = [w for v in result['variants'].values() for w in v['widths'].values()]
    result['mechanism_pass'] = bool(all(w['duplicate_changes_output'] and all(c['counts_exact'] and c['vs_fixed']['relative_l2'] < 0.005
                                    for c in w['cases'].values()) for w in all_widths))
    result['reference_accuracy_pass'] = bool(all(c['vs_cpu']['relative_l2'] < 0.05 for w in all_widths for c in w['cases'].values()))
    dump(base/'result.json', result)
    for variant, vr in result['variants'].items():
        print(variant, {w: {mode: round(v['total_ms']['median_ms'],4) for mode,v in wr['latency'].items()}
                        for w,wr in vr['widths'].items()}, flush=True)
    print('MECHANISM', result['mechanism_pass'], 'ACCURACY', result['reference_accuracy_pass'], flush=True)
    if not result['mechanism_pass'] or (not a.equivalence_only and not result['reference_accuracy_pass']):
        raise RuntimeError('Validation failed; see separate gates')


def audit(a):
    base = a.out/a.precision
    info = json.loads((a.out/'info.json').read_text())
    nodes = json.loads((a.out/'graph_audit.json').read_text())
    report = {'packages': {}, 'traces': {}}
    model = onnx.load(a.out/'model.onnx')
    x = np.fromfile(a.out/'x.bin', np.float16).reshape(128, info['H'])
    ids = np.fromfile(a.out/'orders/identity.bin', np.int32)
    report['logical_packing'] = {}
    for width in WIDTHS:
        selected = model.graph.node[-1]
        for w in WIDTHS[:-1]:
            attrs = {att.name: att.g for att in selected.attribute}
            if width == w:
                branch = attrs['then_branch']
                break
            branch = attrs['else_branch']
            if w != WIDTHS[-2]:
                selected = branch.node[0]
        outputs = [n.input[0] for n in branch.node if n.name in nodes[str(width)]['expert_matmuls'][::3]]
        assert len(outputs) == 32//width
        flat = h.make_model(h.make_graph(list(model.graph.node[:-1])+list(branch.node), 'selected_packing',
            list(model.graph.input), [vi(n,T.FLOAT16,[width,32,info['H']]) for n in outputs], list(model.graph.initializer)),
            opset_imports=list(model.opset_import), functions=list(model.functions), ir_version=9)
        sub = Extractor(flat).extract_model([v.name for v in flat.graph.input], outputs)
        evaluator = ReferenceEvaluator(sub, new_ops=[CtxGather3D, CtxScatter3DInt])
        shapes = [list(v.shape) for v in evaluator.run(None, {'x':x, 'expert_ids':ids,'layout_tag':np.zeros(width,np.int32)})]
        assert shapes == [[width,32,info['H']]]*(32//width)
        report['logical_packing'][str(width)] = shapes
    for variant in list(COMPILE_WIDTHS)+[f'fixed_w{w}' for w in WIDTHS]:
        if variant in report['packages']:
            continue
        qpc = base/variant/'programqpc.bin'
        if not qpc.exists():
            continue
        dest = base/'audit'/variant
        if not dest.exists():
            dest.mkdir(parents=True)
            if command([SDK/'tools/qaic-qpc', 'extract', '--qpc', qpc, '--output-dir', dest,
                        '-s', '*StaticConstants.constants.bin', '-s', '*networkdesc.bin_dir/networkdesc.json'], dest/'extract.log'):
                raise RuntimeError('QPC extraction failed')
        size = sum(p.stat().st_size for p in dest.rglob('StaticConstants.constants.bin'))
        assert size > 0
        descriptions = [json.loads(p.read_text()) for p in dest.rglob('networkdesc.json')]
        report['packages'][variant] = {'static_constants_bytes': size, 'qpc_bytes': qpc.stat().st_size,
                                      'network_descriptions': descriptions}
        if variant in COMPILE_WIDTHS:
            expected = set(COMPILE_WIDTHS[variant])
            for d in descriptions:
                index = [v['name'] for v in d['inputs']].index('layout_tag')
                actual = ({v['shapes'][index]['dims'][0] for v in d['allowed_shapes']} if d['allowed_shapes']
                          else {d['inputs'][index]['io_initial']['dims'][0]})
                assert actual == expected
    if a.profile:
        dest = base/'profiling'
        dest.mkdir(exist_ok=False)
        for width in WIDTHS:
            sub = dest/f'w{width}'
            sub.mkdir()
            tag = sub/'layout_tag.bin'
            np.zeros(width, np.int32).tofile(tag)
            ios = [{'path': str(a.out/'x.bin'), 'dims': [128,info['H']], 'elem-size': 2, 'io-direction': 'in', 'map-to': 'x'},
                   {'path': str(a.out/'orders/identity.bin'), 'dims': [32], 'elem-size': 4, 'io-direction': 'in', 'map-to': 'expert_ids'},
                   {'path': str(tag), 'dims': [width], 'elem-size': 4, 'io-direction': 'in', 'map-to': 'layout_tag'}]
            for key, shape, size in [('y', [128,info['H']], 2), ('counts', [32], 4)]:
                ios.append({'path': str(base/a.run_name/'multi'/f'w{width}_identity_{key}.bin'), 'dims': shape,
                            'elem-size': size, 'io-direction': 'out', 'map-to': key})
            config = sub/'io.json'
            dump(config, {'IO-files': [ios]})
            stats = sub/'stats'
            stats.mkdir()
            if command([SDK/'exec/qaic-runner', '-t', base/'multi', '-d', a.device, '--aic-batch-json-input', config,
                        '-n', '3', '-S', '1', '-T', '1', '-c', '--aic-profiling-type', 'raw_device_stats',
                        '--aic-profiling-num-samples', '1', '--aic-profiling-out-dir', stats], sub/'runner.log'):
                raise RuntimeError('Device profiling failed')
            traces = sub/'trace'
            traces.mkdir()
            if command([SDK/'exec/qaic-opstats', '--qpc', base/'multi/programqpc.bin', '--input-dir', stats,
                        '--output-dir', traces, '--summary', '--trace', '--merge-mq-traces', 'true', '--flow-events', 'none'], sub/'opstats.log'):
                raise RuntimeError('Trace conversion failed')
            print('PROFILE_OK', width, flush=True)
    for width in WIDTHS:
        directory = base/'profiling'/f'w{width}'/'trace'
        if not directory.exists():
            continue
        summary = trace_summary(directory, nodes[str(width)]['weight_gathers'])
        observed, matmuls, dequant_matmuls, cores = set(), set(), set(), set()
        for path in directory.glob('*trace*.json'):
            events = json.loads(path.read_text()).get('traceEvents', [])
            threads = {(e['pid'],e['tid']):e['args']['name'] for e in events if e.get('name') == 'thread_name'}
            for e in events:
                if e.get('ph') == 'X' and e.get('name') == e.get('args', {}).get('opKind'):
                    op = e.get('args', {}).get('opName', '')
                    observed.update(map(int, re.findall(r'topology_w(\d+)', op)))
                    names = re.findall(r'topology_w\d+/MatMul_\d+', op)
                    if e['name'] == 'aicconvolutiond32' and names:
                        matmuls.update(names)
                        match = re.search(r'_Core_(\d+)_HMX$', threads[(e['pid'],e['tid'])])
                        assert match
                        cores.add(int(match[1]))
                    if 'dequant' in e['name'].lower():
                        dequant_matmuls.update(names)
        summary['observed_topology_widths'] = sorted(observed)
        summary['expert_matmul_nodes'] = sorted(matmuls)
        summary['dequantized_expert_matmul_nodes'] = sorted(dequant_matmuls)
        summary['expert_hmx_cores'] = sorted(cores)
        assert observed == {width}
        assert matmuls == set(nodes[str(width)]['expert_matmuls'])
        if a.precision == 'mxfp6':
            assert dequant_matmuls == matmuls
        report['traces'][str(width)] = summary
    dump(base/'package_audit.json', report)
    print('STATIC_CONSTANTS_MIB', {k: round(v['static_constants_bytes']/2**20,4) for k,v in report['packages'].items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--stage', choices=['frontend', 'export', 'compile', 'run', 'analyze', 'audit'], required=True)
    p.add_argument('--precision', choices=['fp16', 'mxfp6'], default='fp16')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--run-name', default='run')
    p.add_argument('--variants', nargs='+', default=['multi', 'single_w4'], choices=list(COMPILE_WIDTHS))
    p.add_argument('--run-variants', nargs='+', default=['multi', 'single_w4']+[f'fixed_w{w}' for w in WIDTHS],
                   choices=['multi']+[f'{prefix}_w{w}' for prefix in ['single','fixed'] for w in WIDTHS])
    p.add_argument('--equivalence-only', action='store_true')
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.device != 0 or a.iterations < 1:
        p.error('This experiment uses card 0 and positive iterations')
    if a.stage == 'export' and a.source is None:
        p.error('--source group-width artifact root is required for export')
    {'frontend': frontend, 'export': export, 'compile': compile_graph, 'run': run, 'analyze': analyze, 'audit': audit}[a.stage](a)


if __name__ == '__main__':
    main()
