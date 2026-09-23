#!/usr/bin/env python3
"""All-active plans, FP16 (flag) vs MXFP6 (flag), both compiled here (stats 0), timed with the multi-input host in alternating
outer passes. Usage: e8_allactive.py <plans,...> [outer_passes]"""
import subprocess, json, sys, re, shutil, statistics as st, time
from pathlib import Path
import numpy as np
S = Path('/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad')
A = Path('/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/all_active')
HOST = '/dev/shm/qwen3_all_active_wentao_20260922/multi_input_host'
plans = sys.argv[1].split(','); outer_passes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
OUT = S / 'e8/allactive'; OUT.mkdir(parents=True, exist_ok=True)
def devices_ok():
    out = subprocess.run(['/opt/qti-aic/tools/qaic-util', '-q'], text=True, capture_output=True, timeout=60).stdout
    blocks = re.split(r'(?m)^QID ', out)[1:]
    return len(blocks) == 4 and all(re.search(r'Status:\s*Ready', b) and re.search(r'Nsp Free:\s*16', b) and re.search(r'Networks Loaded:\s*0', b) for b in blocks)
def wait_idle():
    for _ in range(90):
        if devices_ok(): return
        print('devices busy, waiting', flush=True); time.sleep(20)
    raise SystemExit('devices not idle after 30 min')
def compile_plan(plan, prec):
    out = OUT / plan / f'qpc_{prec}'; shutil.rmtree(out, ignore_errors=True); out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', f'-m={A / plan / "model.onnx"}', '-convert-to-fp16', '-aic-num-cores=16', '-mos=1',
           '-aic-enable-depth-first', f'-mdp-load-partition-config={S / "mdp_ts_4.json"}', '-compile-only', f'-aic-binary-dir={out}', '-stats-level=0', '-mdts-mos=1'] + (['-mxfp6-matmul'] if prec == 'mx' else [])
    t0 = time.time(); r = subprocess.run(cmd, text=True, capture_output=True); (OUT / plan / f'compile_{prec}.log').write_text(r.stdout + r.stderr)
    assert (out / 'programqpc.bin').exists(), f'compile failed for {plan} {prec}: {r.stdout[-500:]}'
    print(f'### {plan} {prec} compiled in {time.time()-t0:.0f} s, {(out / "programqpc.bin").stat().st_size} bytes', flush=True); return str(out)
def run_host(qpc, plan, tag):
    target = OUT / plan / f'runs_{tag}'; shutil.rmtree(target, ignore_errors=True)
    info = json.load(open(A / 'experiment.json')); eligible = info['plans'][plan]['eligible']
    manifest = OUT / plan / 'inputs_manifest.txt'
    manifest.write_text(''.join(f'{w} {A / plan / "inputs" / (w + ".bin")}\n' for w in eligible))
    for w in eligible: assert (A / plan / 'inputs' / (w + '.bin')).exists(), f'missing input {plan} {w}'
    wait_idle()
    r = subprocess.run([HOST, qpc, str(manifest), str(target), '100', '10', '3'], text=True, capture_output=True)
    assert 'COMPLETE' in r.stdout, f'host failed {plan} {tag}: {r.stdout[-300:]} {r.stderr[-300:]}'
    res = {}
    for w in eligible:
        samples, ys = [], []
        for rnd in range(3):
            p = target / w / f'r{rnd}'; samples += json.load(open(p / 'timing.json'))['samples_ms']; ys.append((p / 'y.bin').read_bytes())
        res[w] = dict(samples=samples, y=ys[0], counts=(target / w / 'r0' / 'counts.bin').read_bytes(), stable=all(y == ys[0] for y in ys))
    return res
results = json.load(open(OUT / 'results.json')) if (OUT / 'results.json').exists() else {}
for i, plan in enumerate(plans):
    qpcs = {prec: compile_plan(plan, prec) for prec in ('fp16', 'mx')}
    entry = dict(fp16={}, mx={}, rel_l2={}, counts_equal={}, stable={'fp16': True, 'mx': True}); ys = {}
    for outer in range(outer_passes):
        order = ('fp16', 'mx') if (i + outer) % 2 == 0 else ('mx', 'fp16')
        for prec in order:
            res = run_host(qpcs[prec], plan, f'{prec}_o{outer}')
            for w, v in res.items():
                entry[prec].setdefault(w, []).extend(v['samples']); entry['stable'][prec] &= v['stable']
                if outer == 0: ys[(prec, w)] = (v['y'], v['counts'])
    for w in entry['fp16']:
        yf = np.frombuffer(ys[('fp16', w)][0], np.float16).astype(np.float64); ym = np.frombuffer(ys[('mx', w)][0], np.float16).astype(np.float64)
        entry['rel_l2'][w] = float(np.linalg.norm(ym - yf) / np.linalg.norm(yf)); entry['counts_equal'][w] = bool(ys[('fp16', w)][1] == ys[('mx', w)][1])
    results[plan] = entry; json.dump(results, open(OUT / 'results.json', 'w'))
    for prec in ('fp16', 'mx'): shutil.rmtree(OUT / plan / f'qpc_{prec}', ignore_errors=True)
    for w in entry['fp16']:
        f, m = st.median(entry['fp16'][w]), st.median(entry['mx'][w])
        print(f"plan {plan:>22s} {w:8s} fp16 {f:7.3f}  mx {m:7.3f}  ratio {m/f:.3f}  relL2 {entry['rel_l2'][w]:.2e}  counts {entry['counts_equal'][w]}  stable {entry['stable']}", flush=True)
print('ALLACTIVE_DONE')
