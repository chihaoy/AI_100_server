#!/usr/bin/env python3
"""Placement probe, card level: which lanes does each device's partial sum contain?

The graph exported with --emit-partial outputs the per-device partials [D,T,H] *before* the cross-card
reduce. Fit each slice d as a linear combination of the 64 per-lane CPU contributions (y_lane.npy, [P,T,H],
random weights so lanes are near-orthogonal). Coefficients come out 0/1 -> a direct readout of lane -> slice,
no hypothesis assumed.

  python3 moe_probe_lane2device.py <dir> <partial-activation-0-inf-0.bin>
"""
import sys, os, json, numpy as np
d = sys.argv[1]; out = sys.argv[2]
info = json.load(open(os.path.join(d, "info.json")))
D, T, H, P = info["num_devices"], info["T"], info["H"], info["P"]
part = np.fromfile(out, dtype=np.float16).astype(np.float32).reshape(D, T, H)
yl = np.load(os.path.join(d, "y_lane.npy")).reshape(P, T * H)               # [P, TH]
A = yl.T                                                                    # [TH, P]
print(f"D={D} T={T} H={H} P={P}   |partial| max={np.abs(part).max():.3g}   sum-of-slices vs y_ref: "
      f"max|err|={np.abs(part.sum(0).reshape(-1) - np.load(os.path.join(d,'y_ref.npy')).reshape(-1)).max():.3g}")
C = np.zeros((D, P))
for dd in range(D):
    c, res, *_ = np.linalg.lstsq(A, part[dd].reshape(-1), rcond=None)
    C[dd] = c
    fit = A @ c
    print(f"slice {dd}: lanes with coef>0.5 -> {np.nonzero(c > 0.5)[0].tolist()}   "
          f"residual max|err|={np.abs(fit - part[dd].reshape(-1)).max():.3g}   "
          f"coef range on those {c[c>0.5].min():.3f}..{c[c>0.5].max():.3f}, others |max| {np.abs(c[c<=0.5]).max():.3f}")
print("\ncoefficient matrix (rows = device slice, cols = lane), rounded:")
np.set_printoptions(linewidth=250)
for dd in range(D):
    print(f"  d{dd}: " + "".join("#" if v > 0.5 else "." for v in C[dd]))
contig = np.array([[1.0 if p // (P // D) == dd else 0.0 for p in range(P)] for dd in range(D)])
strided = np.array([[1.0 if p % D == dd else 0.0 for p in range(P)] for dd in range(D)])
print(f"\nmatches contiguous (lane p -> slice p//{P//D}):", bool(np.allclose(np.round(C), contig)))
print(f"matches strided    (lane p -> slice p%{D}):", bool(np.allclose(np.round(C), strided)))
