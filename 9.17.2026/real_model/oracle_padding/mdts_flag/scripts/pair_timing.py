#!/usr/bin/env python3
"""Matched host timing of QPC pairs (single-input host). Usage: pair_timing.py <spec.json> <out.json> [rounds]
spec: {"input": path, "ref_y": path, "ref_counts": path, "cases": [{"label":..., "base": qpc_dir, "flag": qpc_dir}, ...]}"""
import subprocess, json, sys, re, statistics as st
from pathlib import Path
import numpy as np
spec = json.load(open(sys.argv[1])); outp = Path(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
HOST = '/dev/shm/qwen3_cold_wentao_20260922/moe_single_host'
REF_Y = np.fromfile(spec['ref_y'], np.float16); REF_C = np.fromfile(spec['ref_counts'], np.int32)
def devices_ok():
    out = subprocess.run(['/opt/qti-aic/tools/qaic-util', '-q'], text=True, capture_output=True, timeout=60).stdout
    blocks = re.split(r'(?m)^QID ', out)[1:]
    return len(blocks) == 4 and all(re.search(r'Status:\s*Ready', b) and re.search(r'Nsp Free:\s*16', b) and re.search(r'Networks Loaded:\s*0', b) for b in blocks)
assert devices_ok(), 'devices not idle'
rows = []; work = outp.parent / (outp.stem + '_runs'); work.mkdir(parents=True, exist_ok=True)
for r in range(rounds):
    cases = spec['cases'] if r % 2 == 0 else list(reversed(spec['cases']))
    for case in cases:
        for var in (('base', 'flag') if r % 2 == 0 else ('flag', 'base')):
            if var not in case: continue
            out = work / f"{case['label']}_{var}_r{r}"; subprocess.run(['rm', '-rf', str(out)])
            res = subprocess.run([HOST, case[var], spec['input'], str(out), '100', '10'], text=True, capture_output=True)
            m = re.search(r'HOST_LATENCY_MS median=([\d.]+) p10=([\d.]+) p90=([\d.]+)', res.stdout + res.stderr)
            if not m: print('FAILED', case['label'], var, r, (res.stdout + res.stderr)[-400:]); sys.exit(1)
            y = np.fromfile(out / 'y.bin', np.float16); c = np.fromfile(out / 'counts.bin', np.int32)
            ok = bool(np.array_equal(y, REF_Y) and np.array_equal(c, REF_C))
            samples = json.load(open(out / 'timing.json'))['samples_ms']
            rows.append(dict(label=case['label'], variant=var, round=r, median=float(m[1]), p10=float(m[2]), p90=float(m[3]), bit_exact=ok, samples=samples))
            print(f"round {r} {case['label']:>16s} {var:4s} median {m[1]} p10 {m[2]} p90 {m[3]} bit-exact: {ok}", flush=True)
    assert devices_ok(), 'devices not idle after round'
json.dump(rows, open(outp, 'w'))
print('\n=== pooled medians (ms) ===')
print(f'{"case":>16s} {"baseline":>10s} {"-mdts-mos=1":>12s} {"ratio":>7s}   round medians base | flag')
for case in spec['cases']:
    pool = {v: [s for row in rows if row['label'] == case['label'] and row['variant'] == v for s in row['samples']] for v in ('base', 'flag')}
    rm = {v: [row['median'] for row in rows if row['label'] == case['label'] and row['variant'] == v] for v in ('base', 'flag')}
    b = st.median(pool['base']) if pool['base'] else float('nan'); f = st.median(pool['flag']) if pool['flag'] else float('nan')
    print(f"{case['label']:>16s} {b:10.3f} {f:12.3f} {f/b:7.3f}   {' '.join(f'{x:.3f}' for x in rm['base'])} | {' '.join(f'{x:.3f}' for x in rm['flag'])}")
