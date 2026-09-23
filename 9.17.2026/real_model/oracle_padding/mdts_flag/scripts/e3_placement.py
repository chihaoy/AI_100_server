#!/usr/bin/env python3
"""All-active placement/capacity plans: compile each with -mdts-mos=1 (stats0), time flag vs existing baseline QPC with the
multi-input host (inner rounds reverse input order), validate outputs, delete the flag QPC. Usage: e3_placement.py <plans,...> [outer_passes]"""
import subprocess, json, sys, re, shutil, statistics as st
from pathlib import Path
import numpy as np
S = Path('/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad')
A = Path('/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/all_active')
HOST = '/dev/shm/qwen3_all_active_wentao_20260922/multi_input_host'
BASE = '/dev/shm/qwen3_all_active_wentao_20260922/{plan}_stats0_compile/qpc'
plans = sys.argv[1].split(','); outer_passes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = S / 'e3'; OUT.mkdir(exist_ok=True)
def devices_ok():
    out = subprocess.run(['/opt/qti-aic/tools/qaic-util', '-q'], text=True, capture_output=True, timeout=60).stdout
    blocks = re.split(r'(?m)^QID ', out)[1:]
    return len(blocks) == 4 and all(re.search(r'Status:\s*Ready', b) and re.search(r'Nsp Free:\s*16', b) and re.search(r'Networks Loaded:\s*0', b) for b in blocks)
def compile_flag(plan):
    out = OUT / plan; shutil.rmtree(out / 'qpc', ignore_errors=True); out.mkdir(parents=True, exist_ok=True)
    cmd = ['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', f'-m={A / plan / "model.onnx"}', '-convert-to-fp16', '-aic-num-cores=16', '-mos=1',
           '-aic-enable-depth-first', f'-mdp-load-partition-config={S / "mdp_ts_4.json"}', '-compile-only', f'-aic-binary-dir={out / "qpc"}', '-stats-level=0', '-mdts-mos=1']
    r = subprocess.run(cmd, text=True, capture_output=True); (out / 'compile.log').write_text(r.stdout + r.stderr)
    assert (out / 'qpc/programqpc.bin').exists(), f'compile failed for {plan}: {r.stdout[-500:]}'
    return str(out / 'qpc')
def run_host(qpc, plan, tag):
    target = OUT / plan / f'runs_{tag}'; shutil.rmtree(target, ignore_errors=True)
    info = json.load(open(A / 'experiment.json')); eligible = info['plans'][plan]['eligible']
    manifest = OUT / plan / 'inputs_manifest.txt'; manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(''.join(f'{w} {A / plan / "inputs" / (w + ".bin")}\n' for w in eligible))
    for w in eligible: assert (A / plan / 'inputs' / (w + '.bin')).exists(), f'missing input {plan} {w}'
    assert devices_ok(), 'devices not idle'
    r = subprocess.run([HOST, qpc, str(manifest), str(target), '100', '10', '3'], text=True, capture_output=True)
    assert 'COMPLETE' in r.stdout, f'host failed {plan} {tag}: {r.stdout[-300:]} {r.stderr[-300:]}'
    res = {}
    for line in manifest.read_text().split('\n'):
        if not line.strip(): continue
        w = line.split()[0]; samples, ys = [], []
        for rnd in range(3):
            p = target / w / f'r{rnd}'; samples += json.load(open(p / 'timing.json'))['samples_ms']; ys.append((p / 'y.bin').read_bytes())
        res[w] = dict(samples=samples, y=ys[0], counts=(target / w / 'r0' / 'counts.bin').read_bytes(), stable=all(y == ys[0] for y in ys))
    return res
results = json.load(open(OUT / 'results.json')) if (OUT / 'results.json').exists() else {}
for i, plan in enumerate(plans):
    flag_qpc = compile_flag(plan); base_qpc = BASE.format(plan=plan)
    entry = dict(base={}, flag={}, bit_exact={}, stable=True)
    for outer in range(outer_passes):
        order = ('base', 'flag') if (i + outer) % 2 == 0 else ('flag', 'base')
        for var in order:
            res = run_host(flag_qpc if var == 'flag' else base_qpc, plan, f'{var}_o{outer}')
            for w, v in res.items():
                entry[var].setdefault(w, []).extend(v['samples']); entry['stable'] &= v['stable']
                if var == 'flag' and outer == 0: entry['_flag_y_' + w] = v['y']; entry['_flag_c_' + w] = v['counts']
                if var == 'base' and outer == 0: entry['_base_y_' + w] = v['y']; entry['_base_c_' + w] = v['counts']
    for w in entry['base']:
        entry['bit_exact'][w] = bool(entry['_flag_y_' + w] == entry['_base_y_' + w] and entry['_flag_c_' + w] == entry['_base_c_' + w])
    for k in [k for k in entry if k.startswith('_')]: del entry[k]
    results[plan] = entry; json.dump(results, open(OUT / 'results.json', 'w'))
    shutil.rmtree(OUT / plan / 'qpc', ignore_errors=True)
    for w in entry['base']:
        b, f = st.median(entry['base'][w]), st.median(entry['flag'][w])
        print(f"{plan:>18s} {w:8s} base {b:7.3f}  flag {f:7.3f}  ratio {f/b:.3f}  n={len(entry['base'][w])}/{len(entry['flag'][w])}  bit-exact {entry['bit_exact'][w]}  stable {entry['stable']}", flush=True)
print('E3_DONE')
