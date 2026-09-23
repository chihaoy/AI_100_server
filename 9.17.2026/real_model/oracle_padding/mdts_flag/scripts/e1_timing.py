#!/usr/bin/env python3
"""Matched host timing: baseline (default compile) vs -mdts-mos=1 for each cold capacity; same session, alternating."""
import subprocess, json, sys, re, statistics as st
from pathlib import Path
import numpy as np
S = Path('/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad')
B = Path('/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/cold_capacity_control')
HOST = '/dev/shm/qwen3_cold_wentao_20260922/moe_single_host'
INPUT = B / 'input_f16.bin'
REF_Y = np.fromfile(B / 'c128/timing_r0/y.bin', np.float16)
REF_C = np.fromfile(B / 'expected_counts_i32.bin', np.int32)
caps = [int(c) for c in sys.argv[1].split(',')] if len(sys.argv) > 1 else [128, 64, 32, 16, 8, 4, 2]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
variants = {'base': lambda c: f'/dev/shm/qwen3_cold_wentao_20260922/c{c}_compile/qpc',
            'flag': lambda c: str(S / f'e1/c{c}_flag_s70/qpc')}
def devices_ok():
    out = subprocess.run(['/opt/qti-aic/tools/qaic-util', '-q'], text=True, capture_output=True, timeout=60).stdout
    blocks = re.split(r'(?m)^QID ', out)[1:]
    return len(blocks) == 4 and all(re.search(r'Status:\s*Ready', b) and re.search(r'Nsp Free:\s*16', b) and re.search(r'Networks Loaded:\s*0', b) for b in blocks)
assert devices_ok(), 'devices not idle'
rows = []
for r in range(rounds):
    order = caps if r % 2 == 0 else list(reversed(caps))
    for cap in order:
        for var in (('base', 'flag') if r % 2 == 0 else ('flag', 'base')):
            out = S / f'e1/timing/{var}_c{cap}_r{r}'
            if out.exists(): subprocess.run(['rm', '-rf', str(out)])
            out.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run([HOST, variants[var](cap), str(INPUT), str(out), '100', '10'], text=True, capture_output=True)
            m = re.search(r'HOST_LATENCY_MS median=([\d.]+) p10=([\d.]+) p90=([\d.]+)', res.stdout + res.stderr)
            if not m: print('FAILED', var, cap, r, res.stdout[-300:], res.stderr[-300:]); sys.exit(1)
            y = np.fromfile(out / 'y.bin', np.float16); c = np.fromfile(out / 'counts.bin', np.int32)
            ok = np.array_equal(y, REF_Y) and np.array_equal(c, REF_C)
            samples = json.load(open(out / 'timing.json'))['samples_ms']
            rows.append(dict(variant=var, capacity=cap, round=r, median=float(m[1]), p10=float(m[2]), p90=float(m[3]), bit_exact=ok, samples=samples))
            print(f'round {r} {var:4s} c{cap:<3d} median {m[1]} ms  p10 {m[2]}  p90 {m[3]}  bit-exact vs C128 ref: {ok}', flush=True)
    assert devices_ok(), 'devices not idle after round'
json.dump(rows, open(S / 'e1/timing.json', 'w'))
print('\n=== pooled medians over all rounds (ms) ===')
print(f'{"cold cap":>8s} {"baseline":>10s} {"-mdts-mos=1":>12s} {"ratio":>7s}   per-round medians base | flag')
for cap in caps:
    pool = {v: [s for row in rows if row['variant'] == v and row['capacity'] == cap for s in row['samples']] for v in variants}
    rm = {v: [row['median'] for row in rows if row['variant'] == v and row['capacity'] == cap] for v in variants}
    b, f = st.median(pool['base']), st.median(pool['flag'])
    print(f'{cap:>8d} {b:10.3f} {f:12.3f} {f/b:7.3f}   {" ".join(f"{x:.3f}" for x in rm["base"])} | {" ".join(f"{x:.3f}" for x in rm["flag"])}')
