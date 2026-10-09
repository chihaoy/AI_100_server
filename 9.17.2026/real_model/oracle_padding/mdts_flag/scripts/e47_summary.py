#!/usr/bin/env python3
"""E47 summary: per chunk length, log latency fitted as program + session effect by least squares over every timed pass of the
given experiments (E46 and E47: every session holds the anchor s5os), programs reported at the mean session. Prints the ladder
with the head-group sum fixed and the attention rewrite and the routing-metadata step kept separate, the capacity attribution at
tuned tiles, accuracy against FP32, and bit-identity checks between programs that should agree.
Also the E48 combine variants at capacity T against D. Usage: e47_summary.py <experiment dir> [<experiment dir> ...]"""
import sys, os, re, glob, json, numpy as np
runs = {}
for O in sys.argv[1:]:
    for d in sorted(glob.glob(f'{O}/run_T*')):
        m = re.fullmatch(r'run_T(\d+)_([A-Z]\d+)_([A-Za-z0-9]+?)(_again|_p\d+)?', os.path.basename(d))
        if not m or not os.path.exists(f'{d}/result.json'): continue
        r = json.load(open(f'{d}/result.json')); lg = r['logits']
        runs.setdefault(int(m.group(1)), []).append(dict(sess=m.group(2), prog=m.group(3), ms=r['timing']['median_ms'], dir=d, l2=lg['relative_l2'],
                                                         nxt=lg['next_token'] == lg['reference_next_token']))
LADDER = [('s2', "2  expert/head parallel, head-group ReduceSum"), ('s2os', "2' + elementwise head-group sum (attention)"),
          ('s3sos', '3a + tree prefix scans (routing metadata)'), ('s3os', '3b + zero-read removal, tiled lane sum'),
          ('s4os', '4  + token-owned combine'), ('s5os', '5  + elementwise final sum'), ('Dos', '6  + rank sort, two tiers (E38 D)'),
          ('Eos', '7  + three tiers (RankTier)')]
GROUPS = [('attention rewrite', 's2', 's2os'), ('routing metadata (scans)', 's2os', 's3sos'), ('dense-combine repairs', 's3sos', 's3os'),
          ('token-owned representation', 's3os', 's4os'), ('elementwise final sum', 's4os', 's5os'), ('MoE reductions, scans included', 's2os', 's5os'),
          ('MoE output aggregation only', 's3sos', 's5os'), ('padding series', 's5os', 'Eos'), ('naive T/T -> RankTier', 's2os', 'Eos')]
ATTR = [('Aos', 'A static, fixed order, T'), ('Bos', 'B gather, fixed order, T'), ('Cos', 'C rank, T'), ('Dos', 'D rank, two tiers'), ('Eos', 'E RankTier')]
COMB = [('s5os', 'D  token-owned (per-card gathers, local sums)'), ('lg', 'local top-k, global rows into unsplit outputs'), ('ls', 'local top-k, explicit per-card slices'), ('ctopk', 'C-topk, CtxGather3D on all lanes'), ('ctopkg', 'C-topk, ONNX Gather on all lanes'),
        ('tcadd', 'C-lanes, token-centric on the lane axis'), ('tcsl', 'C-slices, explicit per-card lane slices'), ('tcc', 'D with the TokenCombine custom op (HVX, VTCM)'), ('s3os', 'repaired dense combine')]
SAME = [('s2os', 's3sos'), ('s2os', 's3os'), ('Eos', 'Cos'), ('Eos', 'Dos'), ('Eos', 'Bos'), ('Eos', 'Aos'), ('s5os', 's4os'), ('s2', 's2os'),
        ('s5os', 'ctopk'), ('s5os', 'ctopkg'), ('s5os', 'tcadd'), ('s5os', 'tcsl'), ('s5os', 'tcc'), ('s5os', 'lg'), ('s5os', 'ls'), ('Cos', 'cdf'), ('Eos', 'edf')]
for T in sorted(runs, reverse=True):
    rs = runs[T]; progs = sorted({r['prog'] for r in rs}); sess = sorted({r['sess'] for r in rs})
    X = np.zeros((len(rs), len(progs) + len(sess) - 1)); y = np.log([r['ms'] for r in rs])
    for i, r in enumerate(rs):
        X[i, progs.index(r['prog'])] = 1
        j = sess.index(r['sess'])
        if j: X[i, len(progs) + j - 1] = 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta
    sf = np.concatenate([[0.0], beta[len(progs):]]); fit = {p: float(np.exp(beta[i] + sf.mean())) for i, p in enumerate(progs)}
    l2 = {p: np.mean([r['l2'] for r in rs if r['prog'] == p]) for p in progs}; nxt = {p: all(r['nxt'] for r in rs if r['prog'] == p) for p in progs}
    print(f'\n## T={T}: {len(rs)} timed passes in sessions {", ".join(sess)}; session factors {", ".join(f"{np.exp(s - sf.mean()):.3f}" for s in sf)}; '
          f'largest residual of the fit {100 * np.abs(np.expm1(resid)).max():.2f}%')
    for p in progs:
        print(f'   {p:6s} {fit[p]:8.1f} ms  (passes: ' + ', '.join(f"{r['sess']} {r['ms']:.1f}" for r in rs if r['prog'] == p) + ')')
    print(f'\n   | Ladder, T={T} | ms | change | rel L2 vs FP32 | next token |\n   |---|---:|---:|---:|:-:|')
    prev = None
    for p, name in LADDER:
        if p not in fit: continue
        ch = f'{100 * (fit[p] / fit[prev] - 1):+.1f}%' if prev else ''
        print(f'   | {name} | {fit[p]:.1f} | {ch} | {l2[p]:.4f} | {"yes" if nxt[p] else "no"} |'); prev = p
    print(f'\n   | Increment, T={T} | from | to | speedup |\n   |---|---|---|---:|')
    for name, a, b in GROUPS:
        if a in fit and b in fit: print(f'   | {name} | {a} | {b} | {fit[a] / fit[b]:.3f}x |')
    if any(p in fit for p, _ in ATTR):
        print(f'\n   | Attribution at tuned tiles, T={T} | ms | vs E |\n   |---|---:|---:|')
        for p, name in ATTR:
            if p in fit: print(f'   | {name} | {fit[p]:.1f} | {fit[p] / fit["Eos"]:.3f}x |' if 'Eos' in fit else f'   | {name} | {fit[p]:.1f} | |')
    if any(p in fit for p, _ in COMB[1:7]):
        print(f'\n   | Combine at capacity T, T={T} | ms | vs D |\n   |---|---:|---:|')
        for p, name in COMB:
            if p in fit: print(f'   | {name} | {fit[p]:.1f} | {fit[p] / fit["s5os"]:.3f}x |')
    if 'cdf' in fit or 'edf' in fit:
        g2 = lambda a, b: f'{fit[a] / fit[b]:.3f}x' if a in fit and b in fit else '-'
        print(f'\n   | 2x2, T={T} | repaired dense | token-owned | dense / token-owned |\n   |---|---:|---:|---:|')
        for row, d_, t_ in (('uniform capacity T', 'cdf', 'Cos'), ('RankTier tiers', 'edf', 'Eos')):
            print(f'   | {row} | {fit.get(d_, float("nan")):.1f} | {fit.get(t_, float("nan")):.1f} | {g2(d_, t_)} |')
        print(f'   | tiers speedup | {g2("cdf", "edf")} | {g2("Cos", "Eos")} | |')
    if 't4' in fit or 't5' in fit:
        print(f'\n   | Tier sweep, T={T} | ms | vs E |\n   |---|---:|---:|')
        for p, name in (('Cos', 'C  uniform capacity T'), ('Dos', 'D  2 tiers'), ('Eos', 'E  3 tiers (RankTier)'), ('t4', '4 tiers'), ('t5', '5 tiers')):
            if p in fit: print(f'   | {name} | {fit[p]:.1f} | {fit[p] / fit["Eos"]:.3f}x |')
    L = lambda p: next((np.fromfile(f"{r['dir']}/logits_f32.bin", np.float32).astype(np.float64) for r in rs
                        if r['prog'] == p and os.path.exists(f"{r['dir']}/logits_f32.bin")), None)
    print()
    for a, b in SAME:
        x, z = L(a), L(b)
        if x is not None and z is not None:
            print(f'   {b} vs {a}: bit-identical {np.array_equal(x, z)}, rel L2 {np.linalg.norm(z - x) / np.linalg.norm(x):.2e}, same argmax {int(x.argmax()) == int(z.argmax())}')
