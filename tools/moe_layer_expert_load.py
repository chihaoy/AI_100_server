#!/usr/bin/env python3
"""单层 MoE 的 per-expert token 负载统计。

回答:这一层里每个 expert 最终要处理多少 token?分布长什么样?稳不稳?

最后一项(split-half 稳定性)是 profile-guided 路线成立的前提 ——
教授原话:"合法,前提是 pattern 不是完全随机"。
"""
import argparse
import json

import numpy as np


def load(path, W):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    E, K = meta["experts"], meta["top_k"]
    out = []
    for k in d.files:
        if k == "meta":
            continue
        a = d[k]
        if a.shape[1] < W:
            continue
        a = a[:, :W, :]
        out.append(np.stack([np.bincount(a[l].ravel(), minlength=E)
                             for l in range(a.shape[0])]))
    return np.stack(out), E, K          # [n_win, L, E]


def bar(v, vmax, width=28):
    return "#" * int(round(v / vmax * width)) if vmax > 0 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("-l", "--layer", type=int, default=0)
    ap.add_argument("-W", type=int, default=48)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    r, E, K = load(a.npz, a.W)          # [n, L, E]
    n, L, _ = r.shape
    x = r[:, a.layer, :]                # [n, E]  这一层
    print(f"{a.npz}  layer {a.layer}   {n} 个窗口 x {a.W} token   E={E}, top_k={K}")
    print(f"每窗口总分配 = W*K = {a.W*K},  每 expert 平均 = {a.W*K/E:.2f}\n")

    mean, mx = x.mean(0), x.max(0)
    p95 = np.percentile(x, 95, axis=0)
    zero = (x == 0).mean(0)
    order = np.argsort(-mean)

    print(f"最忙的 {a.top} 个 expert:")
    print(f"  {'expert':>7}{'均值':>7}{'中位':>6}{'p95':>6}{'最大':>6}{'为零%':>7}  分布")
    vmax = mean.max()
    for e in order[:a.top]:
        print(f"  {e:>7}{mean[e]:>7.2f}{np.median(x[:,e]):>6.0f}{p95[e]:>6.0f}"
              f"{mx[e]:>6.0f}{zero[e]*100:>6.0f}%  {bar(mean[e], vmax)}")
    print(f"\n最闲的 5 个:")
    for e in order[-5:]:
        print(f"  {e:>7}{mean[e]:>7.2f}{np.median(x[:,e]):>6.0f}{p95[e]:>6.0f}"
              f"{mx[e]:>6.0f}{zero[e]*100:>6.0f}%  {bar(mean[e], vmax)}")

    nz = int((mean == 0).sum())
    print(f"\n全层汇总:  最忙 expert 均值 {mean.max():.2f} (= 总体均值的 "
          f"{mean.max()/mean.mean():.1f}x)")
    print(f"           全部 {n} 个窗口都为零的 expert: {nz}/{E} "
          f"-> 这些连权重都不必加载")
    g = np.sort(mean); cum = np.cumsum(g) / g.sum()
    gini = 1 - 2 * np.trapezoid(cum, dx=1/len(g))
    print(f"           Gini = {gini:.3f}   "
          f"最忙 16 个 expert 吃掉 {np.sort(mean)[::-1][:16].sum()/mean.sum()*100:.1f}% 的 token")

    # split-half 稳定性:前一半窗口的 per-expert 均值 vs 后一半
    h = n // 2
    m1, m2 = x[:h].mean(0), x[h:].mean(0)
    r_pearson = np.corrcoef(m1, m2)[0, 1]
    t1 = set(np.argsort(-m1)[:16]); t2 = set(np.argsort(-m2)[:16])
    print(f"\nsplit-half 稳定性 (前 {h} 窗口 vs 后 {n-h} 窗口):")
    print(f"  per-expert 均值相关   r = {r_pearson:.3f}")
    print(f"  top-16 expert 集合重合 {len(t1&t2)}/16 (随机期望 {16*16/E:.1f})")

    # 全部层的稳定性,确认不是挑出来的
    rs = []
    for l in range(L):
        y = r[:, l, :]
        rs.append(np.corrcoef(y[:h].mean(0), y[h:].mean(0))[0, 1])
    rs = np.array(rs)
    print(f"\n全部 {L} 层的 split-half 相关: 均值 {rs.mean():.3f}  "
          f"最小 {rs.min():.3f}  最大 {rs.max():.3f}  (>0.5 的层: {(rs>0.5).sum()}/{L})")


if __name__ == "__main__":
    main()
