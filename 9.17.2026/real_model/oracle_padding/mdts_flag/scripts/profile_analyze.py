#!/usr/bin/env python3
"""Profile instrumented QPCs (5 iters, 3 samples), decode, and analyze with the repo's cold-capacity metrics.
Usage: profile_analyze.py <spec.json> <out_dir>
spec: {"input": path, "ref_y": path, "ref_counts": path, "cases": [{"label":..., "qpc": qpc_dir}, ...]}"""
import subprocess, json, sys, re, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); sys.path.insert(0, '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad')
from tolerant_metrics import analyze as analyze_trace_tolerant
spec = json.load(open(sys.argv[1])); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
S = Path('/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad')
def devices_ok():
    out = subprocess.run(['/opt/qti-aic/tools/qaic-util', '-q'], text=True, capture_output=True, timeout=60).stdout
    blocks = re.split(r'(?m)^QID ', out)[1:]
    return len(blocks) == 4 and all(re.search(r'Status:\s*Ready', b) and re.search(r'Nsp Free:\s*16', b) and re.search(r'Networks Loaded:\s*0', b) for b in blocks)
REF_Y = Path(spec['ref_y']).read_bytes(); REF_C = Path(spec['ref_counts']).read_bytes()
results = {}
for case in spec['cases']:
    d = OUT / case['label']
    have = len(list((d / 'trace').glob('*merged*.trace.json'))) == 3 and len(list((d / 'outputs').glob('y-*.bin'))) == 3
    if not have:
        shutil.rmtree(d, ignore_errors=True); (d / 'stats').mkdir(parents=True); (d / 'outputs').mkdir(); (d / 'trace').mkdir()
    io = {'IO-files': [[{'path': spec['input'], 'dims': [128, 2176], 'elem-size': 2, 'io-direction': 'in', 'map-to': 'probe_input'},
                        {'path': spec['ref_y'], 'dims': [128, 2048], 'elem-size': 2, 'io-direction': 'out', 'map-to': 'y'},
                        {'path': spec['ref_counts'], 'dims': [128], 'elem-size': 4, 'io-direction': 'out', 'map-to': 'counts'}]]}
    json.dump(io, open(d / 'io.json', 'w'))
    if have:
        r = subprocess.CompletedProcess([], 0, stdout=(d / 'runner.log').read_text() if (d / 'runner.log').exists() else '', stderr='')
    else:
      assert devices_ok(), 'devices not idle'
      r = subprocess.run(['/opt/qti-aic/exec/qaic-runner', '-t', case['qpc'], '-D', '0:1:2:3', '--aic-batch-json-input', str(d / 'io.json'), '-n', '5', '-S', '1', '-T', '1', '-c',
                          '--aic-profiling-type', 'raw_device_stats', '--aic-profiling-start-iter', '2', '--aic-profiling-num-samples', '3', '--aic-profiling-out-dir', str(d / 'stats'),
                          '--write-output-start-iter', '2', '--write-output-num-samples', '3', '--write-output-dir', str(d / 'outputs')], text=True, capture_output=True)
    if not have: (d / 'runner.log').write_text(r.stdout + r.stderr)
    exec_ms = {m[1]: float(m[2]) for m in re.finditer(r'ExecTimeUs_Dev_(\d)_Func_0,\s+([\d.]+)', r.stdout)}
    outs = sorted((d / 'outputs').glob('y-*.bin')); cnts = sorted((d / 'outputs').glob('counts-*.bin'))
    bit_exact = all(p.read_bytes() == REF_Y for p in outs) and all(p.read_bytes() == REF_C for p in cnts) and len(outs) == 3
    if not have: r2 = subprocess.run(['/opt/qti-aic/exec/qaic-opstats', '--qpc', case['qpc'] + '/programqpc.bin', '--input-dir', str(d / 'stats'), '--output-dir', str(d / 'trace'),
                         '--summary', '--trace', '--merge-mq-traces', 'true', '--flow-events', 'none'], text=True, capture_output=True)
    traces = sorted((d / 'trace').glob('*merged*.trace.json'))
    samples = []
    for i, t in enumerate(traces):
        metrics = analyze_trace_tolerant(Path(t))
        samples.append(dict(sample=i, **metrics))
    chosen = sorted(samples, key=lambda x: x['device_ms'])[1] if len(samples) == 3 else None
    results[case['label']] = dict(qpc=case['qpc'], exec_ms_per_device=exec_ms, bit_exact=bit_exact, samples=samples, chosen=chosen)
    json.dump(results, open(OUT / 'profiles.json', 'w'), indent=1)
    c = chosen or {}
    print(f"{case['label']:>18s}: device {c.get('device_ms', float('nan')):7.3f} ms | hot span {c.get('hot_span_ms') or 0:.3f} between {c.get('between_groups_ms') or 0:.3f} cold span {c.get('cold_span_ms') or 0:.3f} unpack {c.get('last_unpack_ms') or 0:.3f} combine {c.get('combine_ms') or 0:.3f} | cold cores {c.get('cold_cores')} per card {c.get('cold_cores_per_card')} | hot card spans {c.get('hot_card_spans_ms')} | cold card spans {c.get('cold_card_spans_ms')} | bit-exact {bit_exact} | runner dev ms {exec_ms}", flush=True)
    for junk in ('stats',):
        shutil.rmtree(d / junk, ignore_errors=True)
print('PROFILE_DONE')
