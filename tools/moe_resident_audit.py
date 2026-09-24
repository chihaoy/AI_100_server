#!/usr/bin/env python3
"""Audit resident-selection QPC storage and collect runtime kernel traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

VARIANTS = ['dynamic', 'static_identity', 'static_swap', 'static_shuffle']


def command(args, log):
    with log.open('w') as f:
        f.write(repr(list(map(str, args)))+'\n'); f.flush()
        subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True)


def trace_summary(directory, weight_nodes):
    events = []
    for path in directory.glob('*trace*.json'):
        value = json.loads(path.read_text())
        events.extend(value.get('traceEvents', []))
    kernels = [e for e in events if e.get('ph') == 'X' and
               e.get('name') == e.get('args', {}).get('opKind')]
    if not kernels: raise RuntimeError(f'No kernel events found in {directory}')
    kinds = Counter(e['name'] for e in kernels)
    dequant = [e for e in kernels if 'dequant' in e['name'].lower()]
    conv = [e for e in kernels if e['name'] == 'aicconvolutiond32']
    gather = [e for e in kernels if 'gather' in e['name'].lower()]
    weight_gather = [e for e in gather if any(e.get('args', {}).get('opName', '').startswith(n+'/')
                                            for n in weight_nodes)]
    def rows(items):
        result = {}
        for e in items:
            name = e.get('args', {}).get('opName', '<unknown>')
            row = result.setdefault(name, {'events': 0, 'sum_core_duration_us': 0.0})
            row['events'] += 1; row['sum_core_duration_us'] += e.get('dur', 0)
        return result
    return {'kernel_kinds': dict(kinds), 'weight_gather_events': len(weight_gather),
            'weight_gather_sum_core_duration_us': sum(e.get('dur', 0) for e in weight_gather),
            'dequantization': rows(dequant),
            'convolution': rows(conv), 'gather': rows(gather),
            'note': 'Durations sum kernel events across cores; not wall-clock latency.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--precision', choices=['fp16', 'mxfp6'], default='fp16')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--run-name', default='run')
    a = p.parse_args(); root = a.out.resolve(); base = root/a.precision
    info = json.loads((root/'info.json').read_text())
    graph_audit = json.loads((root/'graph_audit.json').read_text())
    report = {'scope': 'Packed constants and kernel traces; not PCIe DMA accounting.', 'variants': {}}
    for variant in VARIANTS:
        qpc = base/variant/'programqpc.bin'; dest = base/f'{variant}_segments'
        if not dest.exists():
            command(['/opt/qti-aic/tools/qaic-qpc', 'extract', '--qpc', qpc,
                     '--output-dir', dest, '-s', '*StaticConstants.constants.bin'],
                    base/f'{variant}_extract.log')
        constants = sum(f.stat().st_size for f in dest.rglob('StaticConstants.constants.bin'))
        if not constants: raise RuntimeError(f'No static constants found for {variant}')
        report['variants'][variant] = {'static_constants_bytes': constants, 'qpc_bytes': qpc.stat().st_size}
    if a.profile:
        dest = base/'profiling'; dest.mkdir(exist_ok=False)
        for variant in VARIANTS:
            case = 'identity' if variant == 'dynamic' else variant.removeprefix('static_')
            sub = dest/variant; sub.mkdir()
            ios = [{'path': str(root/'x.bin'), 'dims': [128, info['H']], 'elem-size': 2,
                    'io-direction': 'in', 'map-to': 'x'}]
            if variant == 'dynamic':
                ios.append({'path': str(root/'orders/identity.bin'), 'dims': [32], 'elem-size': 4,
                            'io-direction': 'in', 'map-to': 'expert_ids'})
            for name, shape, size in [('y', [128, info['H']], 2), ('counts', [32], 4)]:
                ios.append({'path': str(base/a.run_name/variant/f'{case}_{name}.bin'),
                            'dims': shape, 'elem-size': size, 'io-direction': 'out', 'map-to': name})
            config = sub/'io.json'; config.write_text(json.dumps({'IO-files': [ios]}, indent=2)+'\n')
            stats = sub/'stats'; stats.mkdir()
            command(['/opt/qti-aic/exec/qaic-runner', '-t', base/variant, '-d', a.device,
                     '--aic-batch-json-input', config, '-n', '3', '-S', '1', '-T', '1', '-c',
                     '--aic-profiling-type', 'raw_device_stats', '--aic-profiling-num-samples', '1',
                     '--aic-profiling-out-dir', stats], sub/'runner.log')
            traces = sub/'trace'; traces.mkdir()
            command(['/opt/qti-aic/exec/qaic-opstats', '--qpc', base/variant/'programqpc.bin',
                     '--input-dir', stats, '--output-dir', traces, '--summary', '--trace',
                     '--merge-mq-traces', 'true', '--flow-events', 'none'], sub/'opstats.log')
            print('PROFILE_OK', variant, flush=True)
    for variant in VARIANTS:
        traces = base/'profiling'/variant/'trace'
        if traces.exists():
            report['variants'][variant]['trace'] = trace_summary(traces,
                [n['name'] for n in graph_audit[variant]['weight_gathers']])
    (base/'package_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
