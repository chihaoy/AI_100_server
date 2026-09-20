#!/usr/bin/env python3
"""Decode: is the number of tokens expert e (layer l) receives at step t related to what it receives at step t+1?
Input: npz from moe_collect_decode_routing.py (dec [N, L, B, K]). counts[t, l, e] = #sequences in the batch whose
token at step t was routed to expert e in layer l (0..B). Reports, per lag k: Pearson corr(counts[t], counts[t+k])
per layer, exact-equality rate, E[count_{t+1} | count_t = n], per-sequence top-k set Jaccard, and random baselines."""
import argparse, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--out"); a = ap.parse_args()
z = np.load(a.npz, allow_pickle=True); m = json.loads(str(z["meta"])); dec = z["dec"]; N, L, B, K = dec.shape; E = m["E"]
lines = []
def say(s=""): print(s); lines.append(s)
say(f"# {m['bench']}  batch B={B}  decode steps N={N}  L={L} E={E} K={K}   (routing from CPU bf16 HF model, greedy)")
counts = np.zeros((N, L, E), np.int32)
for t in range(N):
    for l in range(L): counts[t, l] = np.bincount(dec[t, l].reshape(-1), minlength=E)
u = B * K / E
say(f"# per step & layer: {B} sequences x {K} = {B*K} assignments over {E} experts; uniform share {u:.1f}/expert")
mx = counts.max(2); say(f"\n## A. imbalance within one decode step (per (step,layer)): busiest/uniform median {np.median(mx/u):.2f}  zero-experts median {np.median((counts==0).sum(2)):.0f}  busiest/B median {np.median(mx/B):.2f}")
say("\n## B. step t vs step t+k: Pearson corr of the 128-expert count vectors, same layer (mean over layers and t; [min..max over layers])")
rng = np.random.default_rng(0)
def corr_lag(c, k):
    r = []
    for l in range(L):
        v = [np.corrcoef(c[t, l], c[t+k, l])[0, 1] for t in range(N-k) if c[t, l].std() > 0 and c[t+k, l].std() > 0]
        r.append(np.mean(v))
    return np.array(r)
for k in [1, 2, 4, 8, 16, 31]:
    if k < N:
        r = corr_lag(counts, k); say(f"  lag {k:2d}: corr = {r.mean():.3f}   [{r.min():.3f} .. {r.max():.3f}]")
# baselines: (i) same layer, counts with experts randomly permuted at t+1 (destroys identity) -> ~0; (ii) t vs t+1 but different LAYER
perm = counts.copy()
for t in range(N):
    for l in range(L): perm[t, l] = perm[t, l][rng.permutation(E)]
say(f"  baseline, expert identity shuffled       : corr = {np.mean([np.corrcoef(counts[t,l], perm[t+1,l])[0,1] for t in range(N-1) for l in range(L)]):.3f}")
say(f"  baseline, step t layer l vs step t+1 layer l+1: corr = {np.mean([np.corrcoef(counts[t,l], counts[t+1,l+1])[0,1] for t in range(N-1) for l in range(L-1)]):.3f}")
say(f"  reference, prefill-pooled hot experts vs each decode step (does decode reuse the prompt's hot experts?): see C")
say("\n## C. conditional: given expert e got n tokens at step t, what does it get at step t+1?  (all layers, all t)")
c0, c1 = counts[:-1].reshape(-1), counts[1:].reshape(-1)
say("   n(t)   cells    E[n(t+1)]   P(n(t+1)=0)   P(|Δ|<=1)   P(n(t+1)>=n(t)/2)")
for n in range(0, min(B, 12) + 1):
    sel = c0 == n
    if sel.sum() < 20: continue
    nn = c1[sel]; say(f"   {n:3d}  {sel.sum():7d}   {nn.mean():7.2f}     {np.mean(nn==0):6.3f}      {np.mean(np.abs(nn-n)<=1):6.3f}      {np.mean(nn>=n/2):6.3f}")
say(f"   overall: P(n(t+1)>0 | n(t)>0) = {np.mean(c1[c0>0]>0):.3f}   vs base rate P(n>0) = {np.mean(c0>0):.3f}")
say(f"            exact equality n(t+1)==n(t): {np.mean(c0==c1):.3f}   (shuffled-identity baseline {np.mean(perm[:-1].reshape(-1)==perm[1:].reshape(-1)):.3f})")
say("\n## D. per sequence: Jaccard of its top-8 set at step t vs t+k, same layer (mean; random 8-of-128 baseline ~0.032)")
def jacc(A, Bs):
    A, Bs = set(A.tolist()), set(Bs.tolist()); return len(A & Bs) / len(A | Bs)
for k in [1, 2, 4, 8, 16]:
    if k < N:
        j = np.mean([jacc(dec[t, l, b], dec[t+k, l, b]) for t in range(N-k) for l in range(L) for b in range(B)])
        say(f"  lag {k:2d}: mean Jaccard {j:.3f}")
jx = np.mean([jacc(dec[t, l, b], dec[t+1, l, (b+1) % B]) for t in range(N-1) for l in range(L) for b in range(B)])
say(f"  baseline: sequence b at t vs sequence b+1 at t+1 (different sequence): {jx:.3f}")
say("\n## E. batch-level active set (experts with >=1 token) at t vs t+1: Jaccard, mean over layers/t")
act = counts > 0
ja = np.mean([np.sum(act[t,l] & act[t+1,l]) / np.sum(act[t,l] | act[t+1,l]) for t in range(N-1) for l in range(L)])
ja_r = np.mean([np.sum(act[t,l] & (perm[t+1,l]>0)) / np.sum(act[t,l] | (perm[t+1,l]>0)) for t in range(N-1) for l in range(L)])
say(f"  active-set Jaccard adjacent steps: {ja:.3f}   (identity-shuffled baseline {ja_r:.3f});  active experts per step median {np.median(act.sum(2)):.0f}/128")
say("\n## F. per-layer lag-1 corr (L0..L{}):".format(L-1)); say("   " + " ".join(f"{v:.2f}" for v in corr_lag(counts, 1)))
if a.out: open(a.out, "w").write("\n".join(lines) + "\n")
