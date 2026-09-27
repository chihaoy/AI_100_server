#!/usr/bin/env python3
"""Cross-workload routing comparison. Workloads are directories of routing npz files (from the collectors).
Usage: analyze_workloads.py <layer> <T> <name=dir> [<name=dir> ...]   (layer -1 = per-layer overlap profile)"""
import sys, glob, os, numpy as np
L, T = int(sys.argv[1]), int(sys.argv[2]); W = {}
for a in sys.argv[3:]:
    name, d = a.split('=', 1); files = sorted(glob.glob(f'{d}/T{T}_*.npz'))
    W[name] = np.array([np.stack([np.bincount(np.load(f)['routing_idx'][l].ravel(), minlength=128) for l in range(48)]) for f in files])   # [P, 48, 128]
names = list(W)
def hotset(C): return set(np.argsort(-C.mean(0))[:64])          # calibrated hot set from mean counts over prompts
def jacc(a, b): return len(a & b) / len(a | b)
if L >= 0:
    print(f"layer {L}, T={T}: " + ', '.join(f"{n}: {len(W[n])} prompts" for n in names))
    for n in names:
        C = W[n][:, L]; act = (C > 0).sum(1); never = (C == 0).all(0).sum(); mx = C.max(1); c65 = np.sort(C, 1)[:, ::-1][:, 64]
        print(f"  {n:10s} active/prompt {act.min()}-{act.max()} (median {int(np.median(act))}); never used {never}; max count median {int(np.median(mx))} (max {mx.max()}); 65th count median {int(np.median(c65))} (max {c65.max()})")
    print("  calibrated hot sets (top-64 by mean count): pairwise Jaccard")
    for i, a in enumerate(names):
        print("   ", f"{a:10s}", ' '.join(f"{jacc(hotset(W[a][:, L]), hotset(W[b][:, L])):.2f}" for b in names))
    print("  never-used sets: sizes and intersections")
    nev = {n: set(np.where((W[n][:, L] == 0).all(0))[0]) for n in names}
    for a in names: print("   ", f"{a:10s}", ' '.join(f"{len(nev[a] & nev[b]):3d}" for b in names))
    print("  cross calibration: hot set + capacities (max over the calibration workload) applied to another workload's prompts")
    print("   calib -> eval   coverage%  C_hot_cal  C_cold_cal  prompts that drop  dropped assignments%  needed C_cold (max)")
    for a in names:
        Ca = W[a][:, L]; hot = sorted(hotset(Ca)); cold = sorted(set(range(128)) - set(hot)); Ch = Ca[:, hot].max(); Cc = Ca[:, cold].max()
        for b in names:
            Cb = W[b][:, L]; cov = Cb[:, hot].sum() / Cb.sum(); drops = np.maximum(Cb[:, hot] - Ch, 0).sum(1) + np.maximum(Cb[:, cold] - Cc, 0).sum(1)
            print(f"   {a:>6s} -> {b:<6s} {100*cov:8.1f} {Ch:9d} {Cc:10d} {int((drops>0).sum()):6d}/{len(Cb):<3d} {100*drops.sum()/Cb.sum():18.3f} {Cb[:, cold].max():12d}")
else:
    print(f"per-layer Jaccard of calibrated hot sets (top-64 by mean count), T={T}")
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]]
    print("  layer " + ' '.join(f"{a[:4]}-{b[:4]}" for a, b in pairs) + "   never-used(all workloads) | universal-hot (in every hot set)")
    for l in range(48):
        hs = {n: hotset(W[n][:, l]) for n in names}; nev = set.intersection(*[set(np.where((W[n][:, l] == 0).all(0))[0]) for n in names]); uni = set.intersection(*hs.values())
        print(f"  {l:5d} " + ' '.join(f"{jacc(hs[a], hs[b]):9.2f}" for a, b in pairs) + f"   {len(nev):3d} | {len(uni):3d}")
