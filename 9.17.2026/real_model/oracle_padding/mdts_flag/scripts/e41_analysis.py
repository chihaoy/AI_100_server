#!/usr/bin/env python3
"""Experiment 4b (E41): do calibrated capacities generalize? Device routing counts of held-out chunks (run_e41.sh), each
workload's documents split into calibration and evaluation halves before chunking (e41_make_chunks.py).
The captured counts are in the lane order of the static program A (card-major: lane i holds expert ORDER[i]) and are mapped
back to expert order on load. Designs, per card (experts 32c..32c+31, the placement of programs A-E), three tiers of 8, 8
and 16 lanes as RankTier:
  ranked  lanes take the card's experts by run-time rank; tier i's capacity = h x the largest count at the tier's first rank
          over the calibration chunks, layers and cards (tier 0 at T), rounded up to 16, capped at T (tier_budget.py rule)
  fixed   lanes take the card's experts in a fixed order (per layer, by calibration mean count); tier i's capacity =
          h x the largest count of any expert at the tier's positions over the calibration chunks, same rounding
A chunk overflows if any lane in any layer on any card exceeds its capacity. Reports, per chunk length: the headroom sweep
(1.0, 1.15, 1.28, 1.5) for both designs, pooled and per workload; leave-one-workload-out; per-workload calibration; the deployed RankTier capacities on
the new evaluation set; provisioned rows against the overflow rate (fine headroom grid, written to e41_curves.json); the
sorted-load profile and the order-statistics bound n_(r) <= N/r. Usage: e41_analysis.py <mdts_flag dir>"""
import sys, glob, os, json, numpy as np
R = sys.argv[1]; O = f'{R}/full_model/e41'
LANES = (8, 8, 16); START = (0, 8, 16)
DEPLOYED = {128: (128, 48, 16), 256: (256, 64, 32), 512: (512, 128, 64)}
rup = lambda x, T: np.minimum(T, 16 * np.ceil(np.asarray(x) / 16)).astype(int)
ORDER = [32 * c + j for c in range(4) for j in range(16)] + [32 * c + 16 + j for c in range(4) for j in range(16)]   # program A's lanes
def expert_order(x): y = np.zeros_like(x); y[..., ORDER] = x; return y
def load(T):
    d = {}
    for f in sorted(glob.glob(f'{O}/counts_T{T}_*_*.npy')):
        w, s = os.path.basename(f)[len(f'counts_T{T}_'):-4].rsplit('_', 1); d.setdefault(w, {})[s] = expert_order(np.load(f).astype(np.int64))
    return d
def per_card(c): return c.reshape(c.shape[0], 48, 4, 32)                  # [n, 48, 4, 32]
def ranked_caps(cal, T, h):
    s = -np.sort(-per_card(cal), axis=-1)                                  # sorted per card
    return np.array([T] + [rup(h * s[..., START[i]].max(), T) for i in (1, 2)])
def fixed_order(cal):                                                       # [48, 4, 32] expert index at each position
    return np.argsort(-per_card(cal).mean(0), axis=-1, kind='stable')
def fixed_caps(cal, order, T, h):
    x = np.take_along_axis(per_card(cal), order[None], axis=-1)
    return np.array([rup(h * x[..., START[i]:START[i] + LANES[i]].max(), T) for i in range(3)])
def lane_caps(caps): return np.concatenate([[c] * l for c, l in zip(caps, LANES)])   # [32]
def overflow(ev, caps, order=None):
    x = per_card(ev); x = -np.sort(-x, axis=-1) if order is None else np.take_along_axis(x, order[None], axis=-1)
    exc = np.maximum(0, x - lane_caps(caps))
    return (exc.reshape(len(ev), -1).max(1) > 0), exc.sum((1, 2, 3))        # any overflow per chunk, overflowed assignments per chunk
rows = lambda caps: int((np.array(LANES) * caps).sum())
out = {}
for T in (128, 256, 512):
    D = load(T)
    if not D: continue
    cal = np.concatenate([v['cal'] for v in D.values()]); ev = np.concatenate([v['eval'] for v in D.values()])
    order = fixed_order(cal); res = {'n_cal': len(cal), 'n_eval': len(ev), 'workloads': {w: [len(v['cal']), len(v['eval'])] for w, v in D.items()}}
    print(f'\n=== T={T}: {len(cal)} calibration and {len(ev)} evaluation chunks ({", ".join(f"{w} {a}/{b}" for w, (a, b) in res["workloads"].items())}); naive T/T = {32 * T} rows per card')
    print('| h | ranked caps | rows | chunks with overflow | overflowed assignments (mean / max per chunk) | fixed caps | rows | chunks with overflow | overflowed (mean / max) |')
    print('|---:|---|---:|---:|---|---|---:|---:|---|')
    sweep = {}
    for h in (1.0, 1.15, 1.28, 1.5):
        rc, fc = ranked_caps(cal, T, h), fixed_caps(cal, order, T, h)
        ro, rx = overflow(ev, rc); fo, fx = overflow(ev, fc, order)
        sweep[h] = dict(ranked=dict(caps=rc.tolist(), rows=rows(rc), frac=float(ro.mean()), mean=float(rx.mean()), max=int(rx.max())),
                        fixed=dict(caps=fc.tolist(), rows=rows(fc), frac=float(fo.mean()), mean=float(fx.mean()), max=int(fx.max())))
        print(f'| {h} | {"/".join(map(str, rc))} | {rows(rc)} | {ro.mean():.1%} ({ro.sum()}) | {rx.mean():.2f} / {rx.max()} | {"/".join(map(str, fc))} | {rows(fc)} | {fo.mean():.1%} ({fo.sum()}) | {fx.mean():.2f} / {fx.max()} |')
    res['sweep'] = sweep
    dep = np.array(DEPLOYED[T]); do, dx = overflow(ev, dep)
    print(f'deployed RankTier {"/".join(map(str, dep))} ({rows(dep)} rows): chunks with overflow {do.mean():.1%} ({do.sum()} of {len(ev)}), overflowed assignments mean {dx.mean():.2f}, max {dx.max()}; by workload: ' +
          ', '.join(f'{w} {overflow(v["eval"], dep)[0].mean():.0%}' for w, v in D.items()))
    res['deployed'] = dict(caps=dep.tolist(), rows=rows(dep), frac=float(do.mean()), n=int(do.sum()))
    print('per workload at h=1.28 (pooled calibration): ' + ', '.join(f'{w} ranked {overflow(v["eval"], ranked_caps(cal, T, 1.28))[0].mean():.0%} / fixed {overflow(v["eval"], fixed_caps(cal, order, T, 1.28), order)[0].mean():.0%}' for w, v in D.items()))
    lowo = {}
    for w in D:
        c2 = np.concatenate([v['cal'] for k, v in D.items() if k != w]); o2 = fixed_order(c2); e2 = D[w]['eval']
        rc, fc = ranked_caps(c2, T, 1.28), fixed_caps(c2, o2, T, 1.28)
        lowo[w] = dict(ranked=float(overflow(e2, rc)[0].mean()), ranked_caps=rc.tolist(), fixed=float(overflow(e2, fc, o2)[0].mean()), fixed_caps=fc.tolist())
    res['leave_one_out'] = lowo
    perw = {}
    for w, v in D.items():   # calibrated on the workload's own calibration documents, evaluated on its own evaluation documents
        ow = fixed_order(v['cal']); rc, fc = ranked_caps(v['cal'], T, 1.28), fixed_caps(v['cal'], ow, T, 1.28)
        perw[w] = dict(ranked=float(overflow(v['eval'], rc)[0].mean()), ranked_rows=rows(rc), fixed=float(overflow(v['eval'], fc, ow)[0].mean()), fixed_rows=rows(fc))
    res['per_workload'] = perw
    print('per-workload calibration, h=1.28 (own calibration -> own evaluation; rows, chunks with overflow): ' + ', '.join(
        f'{w} ranked {v["ranked_rows"]} {v["ranked"]:.0%} / fixed {v["fixed_rows"]} {v["fixed"]:.0%}' for w, v in perw.items()))
    print('leave-one-workload-out, h=1.28 (chunks with overflow in the held-out workload): ' + ', '.join(f'{w} ranked {v["ranked"]:.0%} ({"/".join(map(str, v["ranked_caps"]))}) fixed {v["fixed"]:.0%} ({"/".join(map(str, v["fixed_caps"]))})' for w, v in lowo.items()))
    curve = {'ranked': [], 'fixed': []}
    for h in np.linspace(0.6, 3.0, 49):
        for nm, caps, od in (('ranked', ranked_caps(cal, T, h), None), ('fixed', fixed_caps(cal, order, T, h), order)):
            curve[nm].append([float(h), rows(caps), float(overflow(ev, caps, od)[0].mean())])
    res['curve'] = curve
    for target in (0.10, 0.01, 0.0):
        best = {nm: min((r for h, r, f in pts if f <= target), default=None) for nm, pts in curve.items()}
        print(f'fewest rows per card with at most {target:.0%} of evaluation chunks overflowing: ranked {best["ranked"]}, fixed {best["fixed"]}')
    x = np.take_along_axis(per_card(ev), order[None], axis=-1); mid = DEPLOYED[T][1]   # evaluation loads at the pooled fixed positions
    m2, m3 = x[..., 8:16].max(-1), x[..., 16:].max(-1)                              # [n, 48, 4]: busiest expert at positions 9-16 / 17-32
    perwl = {w: int(np.take_along_axis(per_card(v['eval']), order[None], axis=-1)[..., 16:].max()) for w, v in D.items()}
    res['fixed_positions'] = dict(mid_cap=mid, frac_9_16=float((m2 > mid).mean()), frac_17_32=float((m3 > mid).mean()), max_9_16=int(m2.max()), max_17_32=int(m3.max()), max_17_32_by_workload=perwl)
    print(f'fixed order (pooled calibration order), evaluation chunks: the busiest expert at positions 9-16 exceeds the deployed middle-tier capacity {mid} '
          f'in {(m2 > mid).mean():.0%} of (chunk, layer, card) cases (max {m2.max()}), at positions 17-32 in {(m3 > mid).mean():.0%} (max {m3.max()}); '
          'max at positions 17-32 by workload: ' + ', '.join(f'{w} {v}' for w, v in perwl.items()))
    s = -np.sort(-per_card(ev), axis=-1); N = s.sum(-1, keepdims=True); r = np.arange(1, 33)
    res['bound_max'] = float((s * r / N).max()); q = np.percentile(s, [50, 99, 100], axis=(0, 1, 2))
    print(f'sorted load, evaluation chunks: rank 1/8/9/16/17/32 median {q[0][[0, 7, 8, 15, 16, 31]].astype(int).tolist()}, p99 {q[1][[0, 7, 8, 15, 16, 31]].astype(int).tolist()}, max {q[2][[0, 7, 8, 15, 16, 31]].astype(int).tolist()}; '
          f'max over chunks of r * n_(r) / N = {res["bound_max"]:.3f} (the bound says <= 1)')
    out[T] = res
json.dump({str(k): v for k, v in out.items()}, open(f'{O}/e41_results.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
