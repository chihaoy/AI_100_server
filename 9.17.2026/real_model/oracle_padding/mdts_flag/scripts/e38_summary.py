#!/usr/bin/env python3
"""Experiment 1 (capacity attribution A-E) summary over the full-model sessions E38b/E38c. Usage: e38_summary.py <e38 dir>
Per chunk length, log latency is fitted as program effect + session effect by least squares over every timed pass (each
session holds an anchor program, so the design is connected); program latencies are reported at the mean session.
The common setting is every program at the default tile size, the tuned setting each program at its fastest measured
tile size. Telemetry is summarized over the active samples only (every SoC above the 595 MHz idle clock), which excludes
program load and the clock ramps between passes; board power uses all four readings of a sample (one per SoC query).
Logits of every run at a chunk length are compared with one reference run."""
import sys, os, re, glob, json, itertools, numpy as np
O = sys.argv[1]
runs = {}
for d in sorted(glob.glob(f'{O}/run_T*_S*_*')):
    m = re.fullmatch(r'run_T(\d+)_(S\d+)_([A-E])_(def|512|1024)(_again)?', os.path.basename(d))
    if not m or not os.path.exists(f'{d}/result.json'): continue
    T, S, cfg, tile, again = m.groups(); r = json.load(open(f'{d}/result.json'))
    runs[os.path.basename(d)] = dict(T=int(T), S=S, prog=f'{cfg}_{tile}', cfg=cfg, tile=tile, ms=r['timing']['median_ms'],
                                     p10=r['timing']['p10_ms'], p90=r['timing']['p90_ms'], gib=[d_['active_gib'] for d_ in r['device_memory']],
                                     counts=np.array(r['routing_counts']), dir=d)

def telemetry(name):
    f = f'{O}/telemetry_{name[4:]}.txt'
    if not os.path.exists(f): return None
    fr, sp, bp = [], [], []
    for blk in open(f).read().split('@ ')[1:]:
        a = [float(x) for x in re.findall(r'NSP Frequency\(Mhz\):([\d.]+)', blk)]; s = [float(x) for x in re.findall(r'SOC power\(Watts\):([\d.]+)', blk)]
        b = [float(x) for x in re.findall(r'Board power\(Watts\):([\d.]+)', blk)]
        # active sample: every SoC above the idle clock (595 MHz), which drops the clock ramps at pass boundaries; each
        # sample queries the four SoCs in turn and every query reports the board power, so all four readings are kept
        if len(a) == 4 and len(s) == 4 and b and min(a) > 600: fr.append(a); sp.append(s); bp += b
    if not fr: return None
    fr, sp = np.array(fr), np.array(sp)
    return dict(n=len(fr), mhz_median=np.median(fr, 0).tolist(), mhz_min=fr.min(0).tolist(), soc_w=sp.mean(0).tolist(), board_w_mean=float(np.mean(bp)), board_w_peak=float(max(bp)))

ORDER = 'ABCDE'
TIERS = {128: [(8, 128), (8, 48), (16, 16)], 256: [(8, 256), (8, 64), (16, 32)], 512: [(8, 512), (8, 128), (16, 64)]}
def rows_per_card(cfg, T):   # provisioned expert rows per card
    return {'A': 32 * T, 'B': 32 * T, 'C': 32 * T, 'D': 16 * T + 16 * (T // 8), 'E': sum(l * c for l, c in TIERS[T])}[cfg]
def program_bytes(T):        # from the session logs: "program X_tile: N bytes, sha256 ..."
    b = {}
    for f in glob.glob(f'{O}/e38b_T{T}_S*.txt'):
        for m in re.finditer(r'program ([A-E]_(?:def|512|1024)): (\d+) bytes', open(f).read()): b[m.group(1)] = int(m.group(2))
    return b
out = {}
for T in sorted({r['T'] for r in runs.values()}):
    rs = {n: r for n, r in runs.items() if r['T'] == T}
    progs = sorted({r['prog'] for r in rs.values()}, key=lambda p: (p[0], {'def': 0, '512': 1, '1024': 2}[p.split('_')[1]]))
    sess = sorted({r['S'] for r in rs.values()})
    X = np.zeros((len(rs), len(progs) + len(sess) - 1)); y = np.zeros(len(rs))
    for i, r in enumerate(rs.values()):
        X[i, progs.index(r['prog'])] = 1; j = sess.index(r['S'])
        if j: X[i, len(progs) + j - 1] = 1
        y[i] = np.log(r['ms'])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    sfx = np.concatenate([[0], beta[len(progs):]]); shift = sfx.mean()
    fit = {p: float(np.exp(beta[k] + shift)) for k, p in enumerate(progs)}
    resid = y - X @ beta
    raw = {p: {s: float(np.mean([r['ms'] for r in rs.values() if r['prog'] == p and r['S'] == s])) for s in sess if any(r['prog'] == p and r['S'] == s for r in rs.values())} for p in progs}
    ref = next(iter(rs.values()))['dir']; L0 = np.fromfile(f'{ref}/logits_f32.bin', np.float32)
    ident = {n: bool(np.array_equal(np.fromfile(f"{r['dir']}/logits_f32.bin", np.float32), L0)) for n, r in rs.items() if os.path.exists(f"{r['dir']}/logits_f32.bin")}
    tele = {}
    for p in progs:
        ts = [t for t in (telemetry(n) for n, r in rs.items() if r['prog'] == p) if t]
        if ts: tele[p] = dict(n=sum(t['n'] for t in ts), mhz_median=np.median([t['mhz_median'] for t in ts], 0).tolist(), mhz_min=np.min([t['mhz_min'] for t in ts], 0).tolist(),
                              soc_w=np.mean([t['soc_w'] for t in ts], 0).tolist(), board_w_mean=float(np.mean([t['board_w_mean'] for t in ts])), board_w_peak=float(max(t['board_w_peak'] for t in ts)))
    mem = {p: next(r['gib'] for r in rs.values() if r['prog'] == p) for p in progs}; pb = program_bytes(T)
    out[T] = dict(fit=fit, raw=raw, sessions={s: float(np.exp(v - shift)) for s, v in zip(sess, sfx)}, resid_pct=float(100 * np.abs(resid).max()),
                  bit_identical=all(ident.values()), n_runs=len(rs), telemetry=tele, memory_gib=mem, program_bytes=pb, rows_per_card={p: rows_per_card(p[0], T) for p in progs})

    print(f'\n## T={T}: {len(rs)} timed passes in sessions {", ".join(sess)}; largest residual of the program + session fit {out[T]["resid_pct"]:.2f}%; '
          f'session factors {", ".join(f"{s} {v:.4f}" for s, v in out[T]["sessions"].items())}; all logits bit-identical: {out[T]["bit_identical"]}')
    print('\n| Program | fitted | sessions (mean of both passes) | active samples: MHz per SoC median (min) | SoC W | board W mean / peak | GiB/SoC | program GB | rows/card |')
    print('|---|---:|---|---|---|---|---:|---:|---:|')
    for p in progs:
        t = tele.get(p); tl = (f"{[int(v) for v in t['mhz_median']]} ({[int(v) for v in t['mhz_min']]}) | {[round(v) for v in t['soc_w']]} | {t['board_w_mean']:.0f} / {t['board_w_peak']:.0f}" if t else '– | – | –')
        print(f"| {p} | {fit[p]:.1f} ms | {', '.join(f'{s} {v:.1f}' for s, v in raw[p].items())} | {tl} | {np.mean(mem[p]):.1f} | {pb[p] / 1e9:.1f} | {rows_per_card(p[0], T)} |" if p in pb else
              f"| {p} | {fit[p]:.1f} ms | {', '.join(f'{s} {v:.1f}' for s, v in raw[p].items())} | {tl} | {np.mean(mem[p]):.1f} | – | {rows_per_card(p[0], T)} |")
    best = {c: min((p for p in progs if p[0] == c), key=fit.get) for c in ORDER if any(p[0] == c for p in progs)}
    common = {c: f'{c}_def' for c in ORDER if f'{c}_def' in fit}
    for name, sel in (('common setting (default tiles)', common), ('tuned (fastest measured tile per program)', best)):
        if len(sel) < 2: continue
        print(f'\n{name}: ' + ', '.join(f'{sel[c]} {fit[sel[c]]:.1f} ms' for c in sel))
        steps = [('A to B', 'A', 'B'), ('B to C', 'B', 'C'), ('C to D', 'C', 'D'), ('D to E', 'D', 'E'), ('C to E', 'C', 'E'), ('A to E', 'A', 'E')]
        print('  ' + '; '.join(f'{k} {fit[sel[a]] / fit[sel[b]]:.3f}x' for k, a, b in steps if a in sel and b in sel))
        out[T][name.split(' ')[0]] = {c: sel[c] for c in sel}
json.dump({str(T): v for T, v in out.items()}, open(f'{O}/e38_attribution.json', 'w'), indent=1)
