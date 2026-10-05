#!/usr/bin/env python3
"""E46 summary: per chunk length, log latency fitted as program + session effect by least squares over every timed pass
(every session holds the anchor s5os), programs reported at the mean session; ratios for the re-baselined ladder and the
attribution, the head-group sum fix on the full model (s2 -> s2os, E -> Eos), and bit-identity checks.
Usage: e46_summary.py <e46 dir>"""
import sys, os, re, glob, json, numpy as np
O = sys.argv[1]
runs = {}
for d in sorted(glob.glob(f'{O}/run_T*_S*_*')):
    m = re.fullmatch(r'run_T(\d+)_(S\d+)_([A-Za-z0-9]+?)(_again)?', os.path.basename(d))
    if not m or not os.path.exists(f'{d}/result.json'): continue
    r = json.load(open(f'{d}/result.json'))
    runs.setdefault(int(m.group(1)), []).append(dict(sess=m.group(2), prog=m.group(3), ms=r['timing']['median_ms'], dir=d, l2=r['logits']['relative_l2']))
for T in sorted(runs, reverse=True):
    rs = runs[T]; progs = sorted({r['prog'] for r in rs}); sess = sorted({r['sess'] for r in rs})
    X = np.zeros((len(rs), len(progs) + len(sess) - 1)); y = np.log([r['ms'] for r in rs])
    for i, r in enumerate(rs):
        X[i, progs.index(r['prog'])] = 1
        j = sess.index(r['sess'])
        if j: X[i, len(progs) + j - 1] = 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta
    sf = np.concatenate([[0.0], beta[len(progs):]]); fit = {p: float(np.exp(beta[i] + sf.mean())) for i, p in enumerate(progs)}
    print(f'\n## T={T}: {len(rs)} timed passes in sessions {", ".join(sess)}; largest residual of the fit {100 * np.abs(np.expm1(resid)).max():.2f}%')
    for p in progs:
        l2 = [r['l2'] for r in rs if r['prog'] == p]
        print(f'   {p:5s} {fit[p]:8.1f} ms  (passes: ' + ', '.join(f"{r['sess']} {r['ms']:.1f}" for r in rs if r['prog'] == p) + f'; rel L2 vs FP32 {np.mean(l2):.4f})')
    g = lambda a, b: f'{fit[a] / fit[b]:.3f}x' if a in fit and b in fit else '-'
    print(f'   head-group sum fix, full model: naive T/T {g("s2", "s2os")}, RankTier {g("E", "Eos")}')
    print(f'   re-baselined: reductions s2os -> s5os {g("s2os", "s5os")}; padding s5os -> Eos {g("s5os", "Eos")}; naive T/T -> RankTier {g("s2os", "Eos")}; capacities C -> E {g("Cos", "Eos")}')
    L = lambda p: next((np.fromfile(f"{r['dir']}/logits_f32.bin", np.float32) for r in rs if r['prog'] == p and os.path.exists(f"{r['dir']}/logits_f32.bin")), None)
    for a, b in (('s5os', 'Eos'), ('Cos', 'Eos'), ('s2', 's2os'), ('E', 'Eos')):
        x, z = L(a), L(b)
        if x is not None and z is not None: print(f'   {b} vs {a}: bit-identical {np.array_equal(x, z)}, rel L2 {np.linalg.norm(z.astype(np.float64) - x) / np.linalg.norm(x.astype(np.float64)):.2e}')
