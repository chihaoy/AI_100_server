#!/usr/bin/env python3
"""Matched host timing of QPCs that each have their own input/expected counts. Usage: perm_timing.py <spec.json> <out.json> [rounds]
spec: {"ref_y": path, "cases": [{"label":..., "qpc":..., "input":..., "counts":...}, ...]}"""
import subprocess, json, sys, re, statistics as st
from pathlib import Path
import numpy as np
spec = json.load(open(sys.argv[1])); outp = Path(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
HOST = '/dev/shm/qwen3_cold_wentao_20260922/moe_single_host'; import time
_refs = {}
def ref_of(case):
    p = case.get('ref_y', spec.get('ref_y'))
    if p not in _refs: _refs[p] = np.fromfile(p, np.float16).astype(np.float64)
    return _refs[p]
def devices_ok():
    out = subprocess.run(['/opt/qti-aic/tools/qaic-util', '-q'], text=True, capture_output=True, timeout=60).stdout
    blocks = re.split(r'(?m)^QID ', out)[1:]
    return len(blocks) == 4 and all(re.search(r'Status:\s*Ready', b) and re.search(r'Nsp Free:\s*16', b) and re.search(r'Networks Loaded:\s*0', b) for b in blocks)
for _ in range(90):
    if devices_ok(): break
    print('devices busy, waiting', flush=True); time.sleep(20)
assert devices_ok(), 'devices not idle after 30 min'
rows = []; work = outp.parent / (outp.stem + '_runs'); work.mkdir(parents=True, exist_ok=True)
for r in range(rounds):
    cases = spec['cases'] if r % 2 == 0 else list(reversed(spec['cases']))
    for case in cases:
        out = work / f"{case['label']}_r{r}"; subprocess.run(['rm', '-rf', str(out)])
        res = subprocess.run([HOST, case['qpc'], case['input'], str(out), '100', '10'], text=True, capture_output=True)
        m = re.search(r'HOST_LATENCY_MS median=([\d.]+) p10=([\d.]+) p90=([\d.]+)', res.stdout + res.stderr)
        if not m: print('FAILED', case['label'], r, (res.stdout + res.stderr)[-400:]); sys.exit(1)
        y = np.fromfile(out / 'y.bin', np.float16).astype(np.float64); c = np.fromfile(out / 'counts.bin', np.int32)
        counts_ok = bool(np.array_equal(c, np.fromfile(case['counts'], np.int32))); REF = ref_of(case); rel = float(np.linalg.norm(y - REF) / np.linalg.norm(REF)) if y.size == REF.size else float('nan')
        assert np.isfinite(y).all(), 'non-finite output'
        samples = json.load(open(out / 'timing.json'))['samples_ms']
        rows.append(dict(label=case['label'], round=r, median=float(m[1]), p10=float(m[2]), p90=float(m[3]), counts_ok=counts_ok, rel_l2_vs_ref=rel, samples=samples))
        print(f"round {r} {case['label']:>22s} median {m[1]} p10 {m[2]} p90 {m[3]}  counts exact {counts_ok}  y rel L2 vs ref {rel:.2e}", flush=True)
    if not devices_ok(): print('WARNING devices not idle after round', r, flush=True)
json.dump(rows, open(outp, 'w'))
print('\n=== pooled medians (ms) ===')
for case in spec['cases']:
    pool = [s for row in rows if row['label'] == case['label'] for s in row['samples']]; rm = [row['median'] for row in rows if row['label'] == case['label']]
    print(f"{case['label']:>22s} pooled {st.median(pool):7.3f}  p10 {sorted(pool)[len(pool)//10]:7.3f}  p90 {sorted(pool)[len(pool)*9//10]:7.3f}  rounds {' '.join(f'{x:.3f}' for x in rm)}")
