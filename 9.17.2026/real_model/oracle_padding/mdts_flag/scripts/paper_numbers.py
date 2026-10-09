#!/usr/bin/env python3
"""Every full-model number the reframed paper cites, from one program + session fit of log latency over all timed passes of
E46-E55 (every session holds the anchor s5os, ladder step 5). Writes paper_numbers.json and prints the values per chunk length:
the ladder with production, its increments, the capacity attribution, the combine variants and ownership representations
against D, the capacity x aggregation 2x2, the custom op, and the relative L2 error against FP32 of each program.
Usage: paper_numbers.py <full_model dir> <out json>"""
import sys, os, re, glob, json, numpy as np
F, OUTJ = sys.argv[1], sys.argv[2]
runs = {}
for e in ('e46', 'e47', 'e48', 'e50', 'e51', 'e52', 'e54', 'e55'):
    for d in sorted(glob.glob(f'{F}/{e}/run_T*')):
        m = re.fullmatch(r'run_T(\d+)_([A-Z]\d+)_([A-Za-z0-9]+?)(_again|_p\d+)?', os.path.basename(d))
        if not m or not os.path.exists(f'{d}/result.json'): continue
        r = json.load(open(f'{d}/result.json')); lg = r['logits']
        runs.setdefault(int(m.group(1)), []).append(dict(sess=f'{e}:{m.group(2)}', prog=m.group(3), ms=r['timing']['median_ms'],
                                                         l2=lg['relative_l2'], nxt=lg['next_token'] == lg['reference_next_token']))
out = {}
for T in sorted(runs):
    rs = runs[T]; progs = sorted({r['prog'] for r in rs}); sess = sorted({r['sess'] for r in rs})
    X = np.zeros((len(rs), len(progs) + len(sess) - 1)); y = np.log([r['ms'] for r in rs])
    for i, r in enumerate(rs):
        X[i, progs.index(r['prog'])] = 1
        j = sess.index(r['sess'])
        if j: X[i, len(progs) + j - 1] = 1
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta
    sf = np.concatenate([[0.0], beta[len(progs):]]); fit = {p: float(np.exp(beta[i] + sf.mean())) for i, p in enumerate(progs)}
    l2 = {p: float(np.mean([r['l2'] for r in rs if r['prog'] == p])) for p in progs}
    nxt = {p: all(r['nxt'] for r in rs if r['prog'] == p) for p in progs}
    g = lambda a, b: fit[a] / fit[b] if a in fit and b in fit else None
    d = dict(fit=fit, l2=l2, next_token_matches=nxt, passes=len(rs), sessions=len(sess), max_residual_pct=float(100 * np.abs(np.expm1(resid)).max()),
             increments=dict(attention=g('s2', 's2os'), scans=g('s2os', 's3sos'), dense_repairs=g('s3sos', 's3os'), token_owned=g('s3os', 's4os'),
                             final_sum=g('s4os', 's5os'), aggregation=g('s3sos', 's5os'), reductions=g('s2os', 's5os'), padding=g('s5os', 'Eos'),
                             naive_to_ranktier=g('s2os', 'Eos'), production_to_ranktier=g('prod', 'Eos'), production_to_naive=g('prod', 's2os'),
                             two_tiers_to_three=g('Dos', 'Eos')),
             attribution={k: g(k, 'Eos') for k in ('Aos', 'Bos', 'Cos', 'Dos')},
             vs_D={k: g(k, 's5os') for k in ('lg', 'ls', 'ctopk', 'ctopkg', 'tcadd', 'tcsl', 's3os', 'tcc')},
             twobytwo=dict(dense_uniform=fit.get('cdf'), dense_tiers=fit.get('edf'), owned_uniform=fit.get('Cos'), owned_tiers=fit.get('Eos'),
                           dense_change_pct=100 * (fit['edf'] / fit['cdf'] - 1) if 'edf' in fit and 'cdf' in fit else None,
                           owned_change_pct=100 * (fit['Eos'] / fit['Cos'] - 1) if 'Eos' in fit and 'Cos' in fit else None),
             tokens_per_s={k: 1000 * T / fit[k] for k in ('prod', 's2os', 's5os', 'Eos') if k in fit})
    out[T] = d
json.dump(out, open(OUTJ, 'w'), indent=1)
for T, d in sorted(out.items(), reverse=True):
    f = d['fit']; print(f'\n## T={T}: {d["passes"]} passes in {d["sessions"]} sessions, largest residual {d["max_residual_pct"]:.2f}%')
    print('   ladder ms: ' + ', '.join(f'{k} {f[k]:.1f}' for k in ('prod', 's2', 's2os', 's3sos', 's3os', 's4os', 's5os', 'Dos', 'Eos') if k in f))
    print('   increments: ' + ', '.join(f'{k} {v:.3f}' for k, v in d['increments'].items() if v))
    print('   attribution vs E: ' + ', '.join(f'{k} {v:.3f}' for k, v in d['attribution'].items() if v))
    print('   vs D: ' + ', '.join(f'{k} {v:.3f}' for k, v in d['vs_D'].items() if v))
    t = d['twobytwo']; print(f"   2x2: dense {t['dense_uniform']:.1f} -> {t['dense_tiers']:.1f} ({t['dense_change_pct']:+.1f}%), owned {t['owned_uniform']:.1f} -> {t['owned_tiers']:.1f} ({t['owned_change_pct']:+.1f}%)" if t['dense_uniform'] else '   2x2: -')
    print('   tokens/s: ' + ', '.join(f'{k} {v:.0f}' for k, v in d['tokens_per_s'].items()))
    print('   rel L2 vs FP32 (next token): ' + ', '.join(f"{k} {d['l2'][k]:.3f}{'' if d['next_token_matches'][k] else '*'}" for k in ('prod', 's2os', 's3os', 's4os', 's5os', 'Dos', 'Eos', 'tcc') if k in d['l2']))
