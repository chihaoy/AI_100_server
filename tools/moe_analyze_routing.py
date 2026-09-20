#!/usr/bin/env python3
"""Phase 2 analysis of routing captured by moe_collect_routing.py.

Answers the two questions Phase 1 left standing:

  A. EXPERT COVERAGE vs seq_len -- the real curve, not the uniform-routing model.
     Phase 1 measured that the AI 100 lowering gathers expert weights per
     (token, expert) PAIR: Gather output is [T*8, 2048, 768]. So per layer it
     moves  T*top_k*9.0 MiB  of expert weights, against  128*9.0 MiB = 1.125 GiB
     for loading the whole expert bank once. Break-even is T = num_experts/top_k.
     Coverage says how much of the bank a batch of T tokens actually needs.

  B. WORKLOAD DIVERGENCE -- do GSM8K and coding pick different experts?
     If yes, no single static placement can be optimal for both (a structural
     result). If no, placement is a one-off engineering choice.

Usage:
  python3 moe_analyze_routing.py a.npz [b.npz] [--csv out.csv]
"""
import sys, json, argparse, numpy as np

def load(p):
    z = np.load(p, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    arrs = [z[f"p{i}"] for i in range(len(meta["prompts"]))]   # each [L, T, K]
    return meta, arrs

def counts(arrs, E):
    """(layer, expert) -> how many (token, slot) picks"""
    L = arrs[0].shape[0]
    c = np.zeros((L, E), dtype=np.int64)
    for a in arrs:
        for l in range(L):
            np.add.at(c[l], a[l].ravel(), 1)
    return c

def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0: return 0.0
    return float((2*np.arange(1, n+1) - n - 1).dot(x) / (n * x.sum()))

def js(p, q):
    p = p/p.sum(); q = q/q.sum(); m = (p+q)/2
    def kl(a, b):
        mask = a > 0
        return float((a[mask]*np.log2(a[mask]/b[mask])).sum())
    return 0.5*kl(p, m) + 0.5*kl(q, m)

def coverage_curve(arrs, E, Ts):
    """fraction of the E experts touched by the first T tokens, averaged over prompts+layers"""
    out = {}
    for T in Ts:
        vals = []
        for a in arrs:
            L, Tp, K = a.shape
            if Tp < T: continue
            for l in range(L):
                vals.append(len(np.unique(a[l, :T])) / E)
        out[T] = (float(np.mean(vals)), len(vals)//arrs[0].shape[0]) if vals else (float("nan"), 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()

    sets = []
    for p in a.npz:
        meta, arrs = load(p)
        E, K, L = meta["experts"], meta["top_k"], meta["layers"]
        toks = sum(m["tokens"] for m in meta["prompts"])
        print("=" * 78)
        print(f"{meta['bench']}   {len(arrs)} prompts, {toks} tokens, {L}L x {E}E x top-{K}")
        c = counts(arrs, E)
        tot = c.sum(1)                       # per-layer total picks
        print(f"  routing decisions: {c.sum():,}   (= tokens x layers x top_k)")

        # ---- H3: is the distribution skewed? ----
        allc = c.sum(0).astype(float)
        exp = allc.mean()
        print(f"\n  --- 全模型聚合的 expert 使用分布 ---")
        print(f"    mean/expert = {exp:,.0f}   max = {allc.max():,.0f} ({allc.max()/exp:.2f}x)"
              f"   min = {allc.min():,.0f} ({allc.min()/exp:.2f}x)")
        print(f"    Gini = {gini(allc):.4f}   cv = {allc.std()/exp*100:.1f}%"
              f"   top-10 expert 占比 = {np.sort(allc)[-10:].sum()/allc.sum()*100:.1f}%"
              f" (均匀时 {10/E*100:.1f}%)")
        # per-layer skew
        gl = [gini(c[l].astype(float)) for l in range(L)]
        print(f"    per-layer Gini: min={min(gl):.3f} (L{int(np.argmin(gl))})"
              f"  max={max(gl):.3f} (L{int(np.argmax(gl))})  mean={np.mean(gl):.3f}")

        # ---- A: coverage curve ----
        Ts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        cov = coverage_curve(arrs, E, Ts)
        print(f"\n  --- expert 覆盖率 vs seq_len (真实 routing) ---")
        print(f"    {'T':>5} {'实测覆盖':>9} {'均匀模型':>9} {'权重搬运':>10} {'vs 全载':>8}")
        for T in Ts:
            m, n = cov[T]
            if n == 0 or np.isnan(m): continue
            uni = 1 - (1 - 1/E)**(K*T)
            move = T*K*9.0                       # MiB, 3 matrices x 3.0 MiB each
            bank = E*9.0
            print(f"    {T:5d} {m*100:8.1f}% {uni*100:8.1f}% {move:9.0f}MiB {move/bank:7.2f}x")
        sets.append((meta["bench"], allc, c, cov))

    # ---- B: workload divergence ----
    if len(sets) >= 2:
        (n1, a1, c1, _), (n2, a2, c2, _) = sets[0], sets[1]
        print("=" * 78)
        print(f"=== H2: {n1} vs {n2} 的 routing 分布差异 ===")
        print(f"  全模型聚合 JS divergence = {js(a1, a2):.5f} bits   (0=同分布, 1=完全不相交)")
        per = [js(c1[l].astype(float), c2[l].astype(float)) for l in range(c1.shape[0])]
        print(f"  per-layer JS: mean={np.mean(per):.4f}  min={min(per):.4f} (L{int(np.argmin(per))})"
              f"  max={max(per):.4f} (L{int(np.argmax(per))})")
        r1 = np.argsort(-a1); r2 = np.argsort(-a2)
        for k in (8, 16, 32):
            ov = len(set(r1[:k].tolist()) & set(r2[:k].tolist()))
            print(f"  top-{k:2d} 热门 expert 重叠 = {ov}/{k} ({ov/k*100:.0f}%)   随机期望 {k*k/len(a1):.1f}")
        print("\n  === 判读 ===")
        j = js(a1, a2)
        if j < 0.01:
            print("  分布几乎一致 -> 静态放置对两种 workload 同样适用;这是工程问题不是研究问题。")
        elif j < 0.05:
            print("  分布有可测差异但不大 -> 静态放置的损失有限,需结合覆盖率判断值不值得做。")
        else:
            print("  分布显著不同 -> 不存在一个静态放置能同时对两种 workload 最优。结构性问题。")

    if a.csv and sets:
        import csv as _csv
        with open(a.csv, "w", newline="") as fh:
            w = _csv.writer(fh); w.writerow(["bench", "layer", "expert", "picks"])
            for bench, _, c, _ in sets:
                for l in range(c.shape[0]):
                    for e in range(c.shape[1]):
                        w.writerow([bench, l, e, int(c[l, e])])
        print(f"\nwrote {a.csv}")

if __name__ == "__main__":
    main()
