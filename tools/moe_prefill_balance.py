#!/usr/bin/env python3
"""Prefill: is the number of tokens each expert receives balanced?
Per prompt (one prefill of T tokens) and per layer: counts[e] = #tokens routed to expert e (sum = T*K).
Uniform would give every expert T*K/E. Reports single-prompt skew, then pools B independent prompts
(a prefill batch) to see whether pooling makes it balanced. Also per-card load under the EP layout
(card d = (e mod 64)//16, from perop_moe/README.md Phase 3)."""
import argparse, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--out"); ap.add_argument("--trials", type=int, default=200)
a = ap.parse_args()
z = np.load(a.npz, allow_pickle=True); meta = json.loads(str(z["meta"]))
n, L, E, K = len(meta["prompts"]), meta["layers"], meta["experts"], meta["top_k"]
D, P = 4, 64; card_of = (np.arange(E) % P) // (P // D)
def gini(x):
    x = np.sort(np.asarray(x, float)); m = x.mean()
    return 0.0 if m == 0 else (2*np.sum((np.arange(1, len(x)+1))*x)/(len(x)*x.sum()) - (len(x)+1)/len(x))
# counts[p, l, e]
counts = np.zeros((n, L, E), np.int32); T = np.zeros(n, int)
for p in range(n):
    r = z[f"p{p}"]; T[p] = r.shape[1]
    for l in range(L):
        counts[p, l] = np.bincount(r[l].reshape(-1), minlength=E)
lines = []
def say(s=""): print(s); lines.append(s)
say(f"# {meta['bench']}: {n} prompts, T min/median/max = {T.min()}/{int(np.median(T))}/{T.max()}, L={L} E={E} K={K}")
say(f"# uniform would give each expert T*K/E = T/{E//K} tokens; an expert can receive at most T (one slot per token).")
say("\n## A. single prompt (batch=1 prefill), per (prompt, layer): distribution over all %d cells" % (n*L))
u = (T[:, None] * K / E)                         # [n,1]
mx = counts.max(2); ratio = mx / u; zeros = (counts == 0).sum(2)
g = np.array([[gini(counts[p, l]) for l in range(L)] for p in range(n)])
top8_share = np.sort(counts, 2)[:, :, -8:].sum(2) / (T[:, None] * K)
over_half = (counts > T[:, None, None] / 2).sum(2); over_q = (counts > T[:, None, None] / 4).sum(2)
def q(x): return f"median {np.median(x):.2f}  p10 {np.percentile(x,10):.2f}  p90 {np.percentile(x,90):.2f}  max {x.max():.2f}"
say(f"  busiest expert / uniform share : {q(ratio)}")
say(f"  busiest expert / T (fraction)  : {q(mx / T[:, None])}")
say(f"  experts with 0 tokens (of 128) : {q(zeros.astype(float))}")
say(f"  Gini over 128 experts          : {q(g)}")
say(f"  top-8 experts' share of tokens : {q(top8_share)}   (uniform: {8/E:.3f})")
say(f"  #experts with > T/2 tokens     : {q(over_half.astype(float))}    #experts with > T/4: {q(over_q.astype(float))}")
say("  per-layer mean of busiest/uniform (L0..L47):")
say("   " + " ".join(f"{v:.1f}" for v in ratio.mean(0)))
say("\n## B. per-card load under EP (card d holds experts with (e mod 64)//16 == d), single prompt")
card = np.stack([counts[:, :, card_of == d].sum(2) for d in range(D)], 2)       # [n,L,4]
cm = card.max(2) / card.mean(2)
say(f"  card max/mean per (prompt,layer): {q(cm)}    layers with max/mean>1.5 (fraction): {(cm>1.5).mean():.2f}")
say("\n## C. pooling B independent prompts (one prefill batch of B prompts): does it become balanced?")
say("  B    busiest/uniform (median over trials,layers)   Gini   zero-experts   card max/mean   pooled tokens")
rng = np.random.default_rng(0)
for B in [1, 2, 4, 8, 16, 32, 64, 100]:
    rr, gg, zz, cc, tt = [], [], [], [], []
    for t in range(a.trials if B < n else 1):
        idx = rng.choice(n, B, replace=False); c = counts[idx].sum(0)             # [L,E]
        Tb = T[idx].sum(); ub = Tb * K / E
        rr.append(np.median(c.max(1) / ub)); gg.append(np.median([gini(c[l]) for l in range(L)]))
        zz.append(np.median((c == 0).sum(1))); cb = np.stack([c[:, card_of == d].sum(1) for d in range(D)], 1); cc.append(np.median(cb.max(1)/cb.mean(1))); tt.append(Tb)
    say(f"  {B:3d}  {np.median(rr):6.2f}                                 {np.median(gg):.3f}   {np.median(zz):5.1f}         {np.median(cc):.3f}         {int(np.median(tt))}")
say("\n## D. is the busiest expert the same one across prompts? (per layer: how often the per-prompt argmax equals the pooled argmax)")
pooled_arg = counts.sum(0).argmax(1); same = (counts.argmax(2) == pooled_arg[None, :]).mean(0)
say(f"  fraction of prompts whose busiest expert == pooled busiest, per layer: median {np.median(same):.2f}  min {same.min():.2f}  max {same.max():.2f}")
if a.out: open(a.out, "w").write("\n".join(lines) + "\n")
