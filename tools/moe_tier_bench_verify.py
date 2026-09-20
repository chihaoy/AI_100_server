#!/usr/bin/env python3
"""Compare a tier-bench QPC output (y-activation-0-inf-0.bin, fp16 [T,H]) with the CPU references.

  y_ref.npy   : plain per-expert loop with the tier capacity-drop rule (fp32)
  y_torch.npy : the exported torch module's own forward (fp16)
A tiered variant is only trustworthy if the card matches y_ref (the drop rule holds on hardware).
"""
import sys, os, numpy as np
d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "outdir", "y-activation-0-inf-0.bin")
y = np.fromfile(out, dtype=np.float16).astype(np.float32)
ref = np.load(os.path.join(d, "y_ref.npy")).reshape(-1)
yt = np.load(os.path.join(d, "y_torch.npy")).reshape(-1)
assert y.shape == ref.shape, (y.shape, ref.shape)
def rep(name, a, b):
    err = np.abs(a - b); scale = np.abs(b).max()
    rel = err / (np.abs(b) + 1e-3)
    print(f"  vs {name:8s}: max|err|={err.max():.4g} (max|ref|={scale:.4g})  rel-to-max={err.max()/scale:.3g}  "
          f"median rel={np.median(rel):.3g}  p99 rel={np.percentile(rel,99):.3g}  corr={np.corrcoef(a,b)[0,1]:.6f}")
    return err.max() / scale
r1 = rep("y_ref", y, ref); r2 = rep("y_torch", y, yt)
# per-token worst rows (helps spot capacity-drop mismatches: a wrong drop rule shows as whole-token errors)
T = ref.size // 2048
row_err = np.abs(y - ref).reshape(T, -1).max(1)
print(f"  worst tokens: {np.argsort(row_err)[-5:][::-1].tolist()} err={np.sort(row_err)[-5:][::-1].round(4).tolist()}")
# Known artefact: a token with very large expert activations shows ~3-5% error even in the uniform
# no-drop graph (fp16 intermediates on the card); a wrong drop rule would instead corrupt many tokens.
scale = np.abs(ref).max()
bad = int((row_err > 0.01 * scale).sum())
print(f"  tokens with err > 1% of max|ref|: {bad}/{T}")
print("  VERDICT:", "PASS" if bad <= max(2, T // 100) else "FAIL (card output does not match the capacity-drop reference)")
