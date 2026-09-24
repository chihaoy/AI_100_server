#!/usr/bin/env python3
"""Measure a current-count routing boundary on one AI 100 card.

Cut the already validated adaptive graph at its dense global routing weights.
This preserves global top-8 normalization and the same 32 resident experts.
"""
import argparse
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx.utils import Extractor
from onnx.reference import ReferenceEvaluator
import torch
import torch.nn.functional as F

from moe_resident_selection import ResidentMoE, command, compare, dump

SDK = Path('/opt/qti-aic')


def export(a):
    source = a.source.resolve()
    info = json.loads((source/'info.json').read_text())
    root = a.out
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    (root/'reference').mkdir()
    shutil.copyfile(source/'profiles.txt', root/'profiles.txt')
    shutil.copyfile(source/'multi_specializations.json', root/'specializations.json')
    graph = onnx.load(source/'model.onnx')
    cut = '/ScatterElements_output_0'
    assert any(n.op_type == 'ScatterElements' and cut in n.output for n in graph.graph.node)
    for n in graph.graph.node:
        for names in [n.input, n.output]:
            for i, name in enumerate(names):
                if name == cut:
                    names[i] = 'route_weights'
    graph.graph.value_info.append(onnx.helper.make_tensor_value_info('route_weights', onnx.TensorProto.FLOAT16, [128, 128]))
    extractor = Extractor(graph)
    router = extractor.extract_model(['x'], ['route_weights', 'counts'])
    expert = extractor.extract_model([v.name for v in graph.graph.input]+['route_weights'], ['y', 'counts'])
    assert not any(n.name == '/MatMul' for n in expert.graph.node)
    for name, model in [('router', router), ('expert', expert)]:
        dest = root/name
        dest.mkdir()
        onnx.save_model(model, dest/'model.onnx', save_as_external_data=True,
                        all_tensors_to_one_file=True, location='weights.bin', size_threshold=1024)
        onnx.checker.check_model(str(dest/'model.onnx'))
    evaluator = ReferenceEvaluator(onnx.load(root/'router/model.onnx'))
    x = np.fromfile(source/'x.bin', np.float16).reshape(128, info['H'])
    rw, counts = evaluator.run(None, {'x': x})
    participation = (rw[:, :32] > 0).sum(1)
    hot = int(np.flatnonzero(participation == 2)[0])
    stress = int(participation.argmax())
    inputs = {'normal': x, 'warm': np.concatenate([np.repeat(x[stress:stress+1], 16, axis=0), x[16:]]),
              'hot': np.concatenate([np.repeat(x[hot:hot+1], 48, axis=0), x[48:]]),
              'stress': np.repeat(x[stress:stress+1], 128, axis=0)}
    torch.set_num_threads(8)
    model = ResidentMoE(info['H'], info['I']).eval()
    hashes = {n: hashlib.sha256(w.detach().numpy().tobytes()).hexdigest()
              for n, w in [('gate', model.Wg), ('up', model.Wu), ('down', model.Wd)]}
    assert hashes == info['bank_sha256']
    workloads = []
    for name, z in inputs.items():
        weights, counts = evaluator.run(None, {'x': z})
        assert (weights > 0).sum() == 128*8
        order = np.lexsort((np.arange(32), -counts)).astype(np.int32)
        valid = [p for p in info['profiles'] if np.all(counts[order] <= np.repeat(p['capacities'], 4))]
        selected = min(valid, key=lambda p: (p['padded_expert_rows'], p['tag']))
        z.tofile(root/'inputs'/f'{name}_x.bin')
        weights.tofile(root/'reference'/f'{name}_route_weights.bin')
        counts.tofile(root/'reference'/f'{name}_counts.bin')
        order.tofile(root/'reference'/f'{name}_ids.bin')
        with torch.no_grad():
            tx = torch.from_numpy(z).float()
            reference = torch.zeros_like(tx)
            for e in range(32):
                if counts[e]:
                    reference += ((F.silu(tx@model.Wg[e].float())*(tx@model.Wu[e].float()))@model.Wd[e].float())*torch.from_numpy(weights[:, e, None]).float()
        np.save(root/'reference'/f'{name}_y.npy', reference.numpy())
        workloads.append({'name': name, 'max_count': int(counts.max()), 'assignments': int(counts.sum()),
                          'expected_profile': selected['name'], 'expected_tag': selected['tag'],
                          'sorted_counts': counts[order].tolist()})
    (root/'workloads.txt').write_text(''.join(w['name']+'\n' for w in workloads))
    dump(root/'info.json', {**info, 'source': str(source), 'workloads': workloads,
         'scope': 'Single-card current-count dispatch; synthetic partial 32/128 MoE, fixed T=128, no KV',
         'routing_interface_bytes': 128*128*2+32*4})
    print('EXPORT_OK', workloads, flush=True)


def compile_graph(a):
    base = a.out/a.precision
    base.mkdir(exist_ok=True)
    info = json.loads((a.out/'info.json').read_text())
    variants = {'router1': ('router', 1), 'expert15': ('expert', 15),
                'expert16': ('expert', 16), 'fused15': ('fused', 15)}
    result = {}
    for variant in a.variants or variants:
        kind, cores = variants[variant]
        graph = Path(info['source'])/'model.onnx' if kind == 'fused' else a.out/kind/'model.onnx'
        dest = base/variant
        if dest.exists():
            raise RuntimeError(f'Will not overwrite {dest}')
        args = [SDK/'exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', f'-m={graph}',
                '-convert-to-fp16', f'-aic-num-cores={cores}', '-mos=1', '-aic-enable-depth-first',
                '-stats-level=70', '-compile-only', f'-aic-binary-dir={dest}']
        if kind != 'router':
            args.append(f'-network-specialization-config={a.out/"specializations.json"}')
        if a.precision == 'mxfp6':
            args.append('-mxfp6-matmul')
        status = command(args, base/f'{variant}_compile.log')
        result[variant] = {'exit_code': status, 'qpc_exists': (dest/'programqpc.bin').exists(), 'cores': cores}
        dump(base/'compilation.json', result)
    if not all(v['exit_code'] == 0 and v['qpc_exists'] for v in result.values()):
        raise RuntimeError('Some configurations failed to compile; inspect compilation.json')
    fused16 = base/'fused16'
    if not fused16.exists():
        fused16.symlink_to(Path(info['source'])/a.precision/'multi', target_is_directory=True)


def run(a):
    base = a.out/a.precision/a.run_name
    base.mkdir(exist_ok=False)
    host = base/'host'
    if command(['g++', '-O2', '-std=c++17', '-Wall', '-Wextra', '-Werror', f'-I{SDK}/dev/inc',
                Path(__file__).with_name('moe_current_count_host.cpp'), '-o', host,
                f'-L{SDK}/dev/lib/x86_64', '-lQAic', f'-Wl,-rpath,{SDK}/dev/lib/x86_64'], base/'host_build.log'):
        raise RuntimeError('Host build failed')
    info = json.loads((a.out/'info.json').read_text())
    modes = a.modes or ['split', 'pg', 'fused15', 'fused16']
    for mode in modes:
        if command([host, a.out, a.precision, base/mode, a.device, info['H'], a.iterations, mode], base/f'{mode}_host.log'):
            raise RuntimeError(f'Host failed: {mode}')


def analyze(a):
    info = json.loads((a.out/'info.json').read_text())
    base = a.out/a.precision/a.run_name
    result = {'scope': info['scope'], 'precision': a.precision, 'modes': {}}
    for mode in a.modes or ['split', 'pg', 'fused15', 'fused16']:
        dest = base/mode
        with (dest/'samples.csv').open() as f:
            rows = list(csv.DictReader(f))
        mr = {'workloads': {}, 'metadata': json.loads((dest/'metadata.json').read_text())}
        with (dest/'resources.csv').open() as f:
            resources = list(csv.DictReader(f))
        mr['resources'] = resources
        ready = next(i for i, r in enumerate(resources) if r['stage'] == 'ready')
        mr['resource_stability_after_ready'] = len({(r['dram_free_kib'], r['nsp_free']) for r in resources[ready:]}) == 1
        mr['activation_dram_delta_mib'] = (int(resources[0]['dram_free_kib'])-int(resources[ready]['dram_free_kib']))/1024
        with (dest/'full_samples.csv').open() as f:
            full_rows = list(csv.DictReader(f))
        with (dest/'solo_samples.csv').open() as f:
            solo_rows = list(csv.DictReader(f))
        mr['solo_normal'] = {program: {'samples': len(selected), 'median_ms': float(np.median([float(r['total_ms']) for r in selected]))}
                            for program in ['router', 'expert'] if (selected := [r for r in solo_rows if r['program'] == program])}
        for w in info['workloads']:
            name = w['name']
            y = np.fromfile(dest/f'{name}_y.bin', np.float16)
            counts = np.fromfile(dest/f'{name}_counts.bin', np.int32)
            ids = np.fromfile(dest/f'{name}_ids.bin', np.int32)
            val = {'vs_cpu': compare(y, np.load(a.out/'reference'/f'{name}_y.npy').reshape(-1)),
                   'counts_exact': bool(np.array_equal(counts, np.fromfile(a.out/'reference'/f'{name}_counts.bin', np.int32))),
                   'ids_exact': bool(np.array_equal(ids, np.fromfile(a.out/'reference'/f'{name}_ids.bin', np.int32))),
                   'latency': {}}
            if (dest/f'{name}_route_weights.bin').exists():
                val['routes_exact'] = (dest/f'{name}_route_weights.bin').read_bytes() == (a.out/'reference'/f'{name}_route_weights.bin').read_bytes()
            for schedule in ['held', 'cycle']:
                selected = [r for r in rows if r['workload'] == name and r['schedule'] == schedule]
                val['latency'][schedule] = {'samples': len(selected)} | {
                    k: {'median_ms': float(np.median([float(r[k]) for r in selected])),
                        'p10_ms': float(np.percentile([float(r[k]) for r in selected], 10)),
                        'p90_ms': float(np.percentile([float(r[k]) for r in selected], 90))}
                    for k in ['total_ms', 'router_ms', 'decision_ms', 'handoff_ms', 'expert_ms']}
            val['selected_profiles'] = sorted(set(int(r['tag']) for r in rows if r['workload'] == name))
            val['profile_matches_policy'] = val['selected_profiles'] == [w['expected_tag']]
            if mode.startswith('fused'):
                val['vs_conservative_same_ids'] = compare(y, np.fromfile(dest/f'{name}_full_y.bin', np.float16))
                selected = [r for r in full_rows if r['workload'] == name]
                val['conservative_held'] = {'samples': len(selected), 'median_ms': float(np.median([float(r['total_ms']) for r in selected]))}
            fused = base/'fused16'/f'{name}_y.bin'
            if not fused.exists():
                fused = a.out/a.precision/a.control_run/'fused16'/f'{name}_y.bin'
            if fused.exists():
                val['vs_fused16'] = compare(y, np.fromfile(fused, np.float16))
                val['fused16_control_path'] = str(fused)
            else:
                raise RuntimeError(f'Missing fused control: {fused}')
            mr['workloads'][name] = val
        result['modes'][mode] = mr
    values = [v for m in result['modes'].values() for v in m['workloads'].values()]
    result['mechanism_pass'] = all(v['counts_exact'] and v['ids_exact'] and v['profile_matches_policy'] and v.get('routes_exact', True)
        and v.get('vs_fused16', {'relative_l2': 0})['relative_l2'] < 0.005
        and v.get('vs_conservative_same_ids', {'relative_l2': 0})['relative_l2'] < 0.005 for v in values)
    result['reference_accuracy_pass'] = all(v['vs_cpu']['relative_l2'] < 0.05 for v in values)
    dump(base/'result.json', result)
    for mode, mr in result['modes'].items():
        print(mode, {n: round(v['latency']['cycle']['total_ms']['median_ms'], 4) for n, v in mr['workloads'].items()}, flush=True)
    print('MECHANISM', result['mechanism_pass'], 'ACCURACY', result['reference_accuracy_pass'], flush=True)
    if not result['mechanism_pass'] or (not a.equivalence_only and not result['reference_accuracy_pass']):
        raise RuntimeError('Validation failed; see separate mechanism and accuracy gates')


def audit(a):
    # These structures match the installed SDK 1.21.6 QAicApi.h definitions.
    class Constant(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char*64), ('index', ctypes.c_uint32), ('size', ctypes.c_uint32)]
    class Info(ctypes.Structure):
        _fields_ = [('numPrograms', ctypes.c_uint32), ('programInfo', ctypes.c_void_p),
                    ('numConstants', ctypes.c_uint32), ('constantsInfo', ctypes.POINTER(Constant))]
    lib = ctypes.CDLL(str(SDK/'dev/lib/x86_64/libQAic.so'))
    lib.qaicOpenQpcFile.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.qaicQpcGetInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(Info))]
    lib.qaicCloseQpc.argtypes = [ctypes.c_void_p]
    report = {'scope': 'Packaged constants.bin includes compiler constant metadata; not static packed weights alone', 'packages': {}}
    base = a.out/a.precision
    manifest = json.loads((a.out/'info.json').read_text())
    expected_profiles = {(p['tag'], tuple(p['capacities'])) for p in manifest['profiles']}
    for variant in ['router1', 'expert15', 'expert16', 'fused15', 'fused16']:
        qpc = base/variant/'programqpc.bin'
        handle = ctypes.c_void_p()
        if lib.qaicOpenQpcFile(ctypes.byref(handle), str(qpc).encode()) != 0:
            raise RuntimeError(f'QPC open failed: {qpc}')
        try:
            info = ctypes.POINTER(Info)()
            if lib.qaicQpcGetInfo(handle, ctypes.byref(info)) != 0:
                raise RuntimeError(f'QPC info failed: {qpc}')
            constants = {c.name.decode(): c.size for c in info.contents.constantsInfo[:info.contents.numConstants]}
            report['packages'][variant] = {'num_programs': info.contents.numPrograms,
                                          'constant_segments_bytes': constants, 'qpc_bytes': qpc.stat().st_size}
        finally:
            if lib.qaicCloseQpc(handle) != 0:
                raise RuntimeError('QPC close failed')
        dest = base/'audit'/variant
        if not dest.exists():
            dest.mkdir(parents=True)
            if command([SDK/'tools/qaic-qpc', 'extract', '--qpc', qpc, '--output-dir', dest,
                        '-s', '*networkdesc.bin_dir/networkdesc.json'], dest/'extract.log'):
                raise RuntimeError('Network descriptor extraction failed')
        descriptions = [json.loads(p.read_text()) for p in dest.rglob('networkdesc.json')]
        if not descriptions:
            raise RuntimeError('No network descriptor found')
        report['packages'][variant]['network_shapes'] = [
            {'network_name': d['network_name'], 'allowed_shapes': d.get('allowed_shapes'),
             'inputs': [v['name'] for v in d['inputs']]} for d in descriptions]
        if variant != 'router1':
            for d in report['packages'][variant]['network_shapes']:
                actual = set()
                for s in d['allowed_shapes']:
                    shapes = dict(zip(d['inputs'], [v['dims'] for v in s['shapes']]))
                    assert shapes['x'] == [128, manifest['H']]
                    if variant.startswith('expert'):
                        assert shapes['route_weights'] == [128, 128]
                    actual.add((shapes['tag'][0], tuple(shapes[f'rows{g}'][0] for g in range(8))))
                if actual != expected_profiles:
                    raise RuntimeError(f'Compiled profiles differ from manifest: {variant}')
        report['packages'][variant]['shape_interface_verified'] = True
    dump(base/'package_audit.json', report)
    print('PACKAGED_CONSTANTS_MIB', {k: round(v['constant_segments_bytes']['constants.bin']/2**20, 4)
                                   for k, v in report['packages'].items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--stage', choices=['export', 'compile', 'run', 'analyze', 'audit'], required=True)
    p.add_argument('--precision', choices=['fp16', 'mxfp6'], default='fp16')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--run-name', default='run')
    p.add_argument('--control-run', default='run')
    p.add_argument('--variants', nargs='+', choices=['router1', 'expert15', 'expert16', 'fused15'])
    p.add_argument('--modes', nargs='+', choices=['split', 'pg', 'pgdata', 'fused15', 'fused16'])
    p.add_argument('--equivalence-only', action='store_true')
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.device != 0 or a.iterations < 1:
        p.error('This single-card exploration uses card 0 and positive iterations')
    if a.stage == 'export' and a.source is None:
        p.error('--source adaptive artifact root is required for export')
    {'export': export, 'compile': compile_graph, 'run': run, 'analyze': analyze, 'audit': audit}[a.stage](a)


if __name__ == '__main__':
    main()
