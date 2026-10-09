#!/usr/bin/env python3
"""Single-GPU tier calibration for the TensorRT/H100 experiments: one device owns all 128 experts, so tiers rank all of them.
Calibration uses only the E41 calibration split (device routing counts, all 48 layers, eight workloads); the E41 evaluation
split and the AI100 reference chunks used for timing are never looked at for the capacities. Rule as in the paper: tier 1 at
capacity T, tier i>1 at h x the largest calibration count at its first rank, rounded up to g = 16, capped at T (h = 1.28).
Reports, per chunk length and candidate structure: capacities, provisioned rows (against 128 T for uniform capacity), the
fraction of E41 evaluation chunks with an overflow in any layer, and whether the timing traces fit. Writes tiers.json.
Usage: calibrate_tiers.py <mdts_flag dir> <out json> [trace dir ...]"""
import sys, glob, os, json, numpy as np
R, OUT, TR = sys.argv[1], sys.argv[2], sys.argv[3:]
O = f'{R}/full_model/e41'; H, G, E = 1.28, 16, 128
ORDER = [32 * c + j for c in range(4) for j in range(16)] + [32 * c + 16 + j for c in range(4) for j in range(16)]
def expert_order(x): y = np.zeros_like(x); y[..., ORDER] = x; return y
STRUCTS = {'2t_64_64': (64, 64), '3t_32_32_64': (32, 32, 64), '3t_16_48_64': (16, 48, 64), '4t_8_24_32_64': (8, 24, 32, 64),
           '5t_8_8_16_32_64': (8, 8, 16, 32, 64)}
res = {}
for T in (128, 256, 512):
    cal = np.concatenate([expert_order(np.load(f)) for f in sorted(glob.glob(f'{O}/counts_T{T}_*_cal.npy'))])     # [n, 48, 128]
    ev = np.concatenate([expert_order(np.load(f)) for f in sorted(glob.glob(f'{O}/counts_T{T}_*_eval.npy'))])
    scal, sev = -np.sort(-cal, -1), -np.sort(-ev, -1)
    m = scal.max((0, 1))                                                     # per-rank maximum over calibration chunks and layers
    print(f'\n=== T={T}: {len(cal)} calibration / {len(ev)} evaluation chunks; per-rank calibration max at ranks 1,8,16,32,64,128: '
          f'{m[[0, 7, 15, 31, 63, 127]].tolist()}')
    tr = {}
    for d in TR:
        ids = np.load(f'{d}/ids_T{T}.npy'); tr[d] = -np.sort(-np.bincount(ids.ravel(), minlength=E))
    res[T] = dict(n_cal=len(cal), n_eval=len(ev), rank_max=m.tolist(), structures={})
    for name, L in STRUCTS.items():
        a = np.cumsum((0,) + L)[:-1]
        caps = [T] + [int(min(T, G * np.ceil(H * m[a[i]] / G))) for i in range(1, len(L))]
        lane = np.concatenate([[c] * l for c, l in zip(caps, L)])
        ov = (sev > lane).any(-1).any(-1)                                    # chunk overflows in any layer
        rows = int(sum(l * c for l, c in zip(L, caps)))
        fits = {os.path.basename(d.rstrip('/')): bool((t <= lane).all()) for d, t in tr.items()}
        res[T]['structures'][name] = dict(lanes=list(L), caps=caps, rows=rows, rows_vs_uniform=rows / (E * T), eval_overflow=float(ov.mean()), traces_fit=fits)
        print(f'  {name:16s} caps {"/".join(map(str, caps)):22s} rows {rows:6d} ({rows / (E * T):.1%} of uniform)  eval chunks with overflow {ov.mean():.2%}  traces fit {fits}')
json.dump(res, open(OUT, 'w'), indent=1)
