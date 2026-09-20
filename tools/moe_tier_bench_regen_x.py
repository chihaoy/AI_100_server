#!/usr/bin/env python3
"""Regenerate x.bin / y_torch.npy / y_ref.npy of an existing tier-bench variant with a NON-degenerate input
(x drawn from its own generator, seed+1000), without re-exporting the ONNX (the graph does not contain x).
Also prints the routing statistics: tokens per expert and which tokens exceed each lane's capacity.
  python3 moe_tier_bench_regen_x.py <variant_dir> [--shared-weights]
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_tier_bench_export as ex

ap = argparse.ArgumentParser(); ap.add_argument("var"); ap.add_argument("--shared-weights", action="store_true")
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
info = json.load(open(os.path.join(a.var, "info.json")))
groups = [tuple(g) for g in info["groups"]]
m = ex.MoELayerEP(info["T"], info["H"], info["I"], info["K"], info["S"], groups, info["num_devices"], seed=a.seed,
                  capacity_check=info.get("capacity_check", "oob"), stage_widths=info.get("stage_widths"),
                  shared_weights=a.shared_weights).eval()
gx = torch.Generator().manual_seed(a.seed + 1000)
x = (torch.randn(info["T"], info["H"], generator=gx) * 1.0).half()
with torch.no_grad():
    rw = m.route(x); y = m(x); ref = ex.reference(m, x)
x.numpy().tofile(os.path.join(a.var, "x.bin"))
np.save(os.path.join(a.var, "y_torch.npy"), y.float().numpy()); np.save(os.path.join(a.var, "y_ref.npy"), ref.numpy())
cnt = (rw > 0).sum(0).numpy(); P = m.P
caps = np.zeros(m.E, int); a0 = 0
for gi, (tier, n) in enumerate(groups):
    for p in range(n):
        for s in range(m.S): caps[s * P + a0 + p] = m.cap(tier, s)
    a0 += n
over = [(int(e), int(cnt[e]), int(caps[e])) for e in range(m.E) if cnt[e] > caps[e]]
dropped = set()
for e, c, cap in over:
    sel = torch.nonzero(rw[:, e] > 0).squeeze(1); dropped |= set(sel[cap:].tolist())
print(f"{a.var}: nonzero experts per token = {int((rw > 0).sum(1)[0])} (top-{info['K']}), tokens per expert min {cnt.min()} mean {cnt.mean():.1f} max {cnt.max()}")
print(f"  experts over capacity: {len(over)}  {over[:12]}{' ...' if len(over) > 12 else ''}")
print(f"  tokens with at least one dropped assignment: {len(dropped)}/{info['T']}  -> {sorted(dropped)[:24]}{' ...' if len(dropped) > 24 else ''}")
print(f"  torch forward vs reference max|err| = {(y.float() - ref).abs().max().item():.4g}")
json.dump({"experts_over_capacity": over, "tokens_dropped": sorted(dropped)}, open(os.path.join(a.var, "drops.json"), "w"))
