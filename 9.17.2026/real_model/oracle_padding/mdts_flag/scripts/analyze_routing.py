#!/usr/bin/env python3
"""Real-case flexibility study on one layer: empties, hot-set stability, static (calibrated, leave-one-out) plans vs the
per-prompt oracle, drops and padded rows, with the E18 time model. Usage: analyze_routing.py <routing dir> <layer> [T]"""
import sys, glob, os, numpy as np
d, L = sys.argv[1], int(sys.argv[2]); Tsel = int(sys.argv[3]) if len(sys.argv) > 3 else 128
files = sorted(f for f in glob.glob(f'{d}/T{Tsel}_*.npz'))
names = [os.path.basename(f)[:-4] for f in files]; C = []
for f in files:
    z = np.load(f); C.append(np.bincount(z['routing_idx'][L].ravel(), minlength=128))
C = np.array(C); P, T = len(C), Tsel; assert (C.sum(1) == 8 * T).all()
print(f"layer {L}, T={T}, {P} prompts ({sum(n.startswith(f'T{T}_nat') for n in names)} natural, {sum('cat' in n for n in names)} concatenated)")
empty = (C == 0); act = (~empty).sum(1)
print(f"  active experts per prompt: min {act.min()} median {int(np.median(act))} max {act.max()}; empties {128-act.max()}..{128-act.min()}")
never = empty.all(0).sum(); mostly = (empty.mean(0) >= 0.9).sum()
print(f"  experts empty in every prompt: {never}; empty in >=90% of prompts: {mostly}; mean per-expert empty rate {empty.mean():.2f}")
top = np.argsort(-C, axis=1)[:, :64]; hot_sets = [set(t) for t in top]
jac = [len(hot_sets[i] & hot_sets[j]) / len(hot_sets[i] | hot_sets[j]) for i in range(P) for j in range(i + 1, P)]
print(f"  per-prompt hot set (top-64): pairwise Jaccard median {np.median(jac):.2f} min {np.min(jac):.2f}; max count per prompt median {int(np.median(C.max(1)))} (min {C.max(1).min()}, max {C.max(1).max()})")
# leave-one-out calibration: hot set = 64 most frequently-hot experts over the other prompts; capacities = max over the other prompts
rows = []
for i in range(P):
    others = [j for j in range(P) if j != i]
    freq = np.zeros(128)
    for j in others: freq[top[j]] += 1
    meanc = C[others].mean(0); hot = set(np.lexsort((-meanc, -freq))[:64]); cold = set(range(128)) - hot
    hot_l = sorted(hot); cold_l = sorted(cold)
    Ch_cal = max(C[j][hot_l].max() for j in others); Cc_cal = max(C[j][cold_l].max() for j in others)
    ch_need, cc_need = C[i][hot_l].max(), C[i][cold_l].max()
    drops_cal = int(np.maximum(C[i][hot_l] - Ch_cal, 0).sum() + np.maximum(C[i][cold_l] - Cc_cal, 0).sum())
    coverage = C[i][hot_l].sum() / (8 * T)                                     # share of assignments landing in the calibrated hot set
    in_hot = len(hot & hot_sets[i])                                            # how many of the prompt's own top-64 are in the calibrated hot set
    ora_h, ora_c = C[i].max(), np.sort(C[i])[::-1][64]                          # per-prompt oracle hc plan
    rows.append((names[i], Ch_cal, Cc_cal, ch_need, cc_need, drops_cal, coverage, in_hot, ora_h, ora_c, act[i]))
def q(v, p): return float(np.percentile(v, p))
Ch = np.array([r[1] for r in rows]); Cc = np.array([r[2] for r in rows]); chn = np.array([r[3] for r in rows]); ccn = np.array([r[4] for r in rows]); dr = np.array([r[5] for r in rows]); cov = np.array([r[6] for r in rows]); inh = np.array([r[7] for r in rows]); oh = np.array([r[8] for r in rows]); oc = np.array([r[9] for r in rows])
print(f"  calibrated hot set (leave-one-out): covers {100*cov.mean():.1f}% of assignments (min {100*cov.min():.1f}%); {inh.mean():.1f} of the prompt's own top-64 experts are in it (min {inh.min()})")
print(f"  static capacities from calibration (max over other prompts): C_hot {int(np.median(Ch))} (needed by the held-out prompt: median {int(np.median(chn))}, max {chn.max()}), C_cold {int(np.median(Cc))} (needed: median {int(np.median(ccn))}, max {ccn.max()})")
print(f"  dropped assignments under the static plan: {int((dr>0).sum())} of {P} prompts drop anything; total {dr.sum()} of {P*8*T} ({100*dr.sum()/(P*8*T):.3f}%); worst prompt {dr.max()}")
print(f"  per-prompt oracle hc plan: C_hot median {int(np.median(oh))}, C_cold median {int(np.median(oc))} -> padded rows {int(np.median(64*oh+64*oc))} ({np.median((64*oh+64*oc)/(8*T)):.1f}x real)")
print(f"  static plan padded rows: {int(np.median(64*Ch+64*Cc))} ({np.median((64*Ch+64*Cc)/(8*T)):.1f}x real); with the cold capacity at the drop-free maximum {ccn.max()}: {int(np.median(64*Ch)+64*ccn.max())}")
# time model (per card, ms): hot = 0.6 + 0.0045*C_hot (T>=256; at T=128 the hot stage is at its 0.6 floor), cold = 0.04 * active cold lanes per card
def tmodel(Chot, Ccold, active_cold_per_card, T):
    hot = max(0.6, 0.6 + 0.0045 * Chot) if T >= 256 else 0.62
    return hot + 0.04 * active_cold_per_card
if T >= 256:
    ora_t = [tmodel(r[8], r[9], (r[10] - 64) / 4, T) for r in rows]; sta_t = [tmodel(r[1], r[2], (r[10] - 64) / 4, T) for r in rows]
    print(f"  model expert-stage time per card: oracle {np.median(ora_t):.2f} ms, static calibrated {np.median(sta_t):.2f} ms (+{100*(np.median(sta_t)/np.median(ora_t)-1):.0f}%)")
print("  per prompt: name, C_hot cal/needed, C_cold cal/needed, drops, coverage%, own-top64-in-cal, active")
for r in rows[:12]: print(f"    {r[0]:14s} {r[1]:4d}/{r[3]:<4d} {r[2]:3d}/{r[4]:<3d} {r[5]:5d} {100*r[6]:6.1f} {r[7]:3d} {r[10]:4d}")
