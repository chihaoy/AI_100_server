#!/usr/bin/env python3
"""逐层列出最忙的 N 个 expert(按收到的 token 数)。

token 数 = 该 expert 被窗口内多少个 token 选进 top-k(在窗口间取平均)。
括号内 = 占窗口 token 总数的百分比;top-k=8 时全部 expert 的百分比和为 800%。
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/home/chihao/mllm/tools")
from moe_layer_expert_load import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("-W", type=int, default=48)
    ap.add_argument("-n", "--top", type=int, default=5)
    ap.add_argument("--stat", default="max",
                    choices=["max", "mean", "p50", "p95", "p99"],
                    help="跨 prompt 的聚合方式 (默认 max)")
    ap.add_argument("--csv", help="前 N 名的 CSV")
    ap.add_argument("--full-csv", help="完整 [L, E] 矩阵的 CSV")
    a = ap.parse_args()

    r, E, K = load(a.npz, a.W)          # [n_win, L, E]
    n, L, _ = r.shape
    if a.stat == "max":
        M = r.max(0)
    elif a.stat == "mean":
        M = r.mean(0)
    else:
        M = np.percentile(r, float(a.stat[1:]), axis=0)   # [L, E]
    print(f"{a.npz}   {n} 个 {a.W}-token 窗口   E={E}, top_k={K}")
    print(f"聚合方式 = {a.stat} (跨 {n} 条 prompt);括号 = 占窗口 {a.W} 个 token 的百分比")
    print(f"每 expert 的平均值基准 = W*K/E = {a.W*K/E:.2f} token\n")

    hdr = "  层  " + "".join(f"{'#'+str(i+1):>18}" for i in range(a.top)) + f"{'前%d合计'%a.top:>10}"
    print(hdr)
    print("-" * len(hdr))
    csv = ["layer,rank,expert,mean_tokens,pct_of_window"]
    for l in range(L):
        o = np.argsort(-M[l])[: a.top]
        cells = "".join(f"{e:>8}({M[l,e]/a.W*100:5.1f}%)" for e in o)
        print(f"  {l:<4}{cells}{M[l,o].sum()/M[l].sum()*100:>9.1f}%")
        for k, e in enumerate(o):
            csv.append(f"{l},{k+1},{e},{M[l,e]:.3f},{M[l,e]/a.W*100:.2f}")
    if a.full_csv:
        with open(a.full_csv, "w") as f:
            f.write("layer," + ",".join(f"e{e}" for e in range(E)) + "\n")
            for l in range(L):
                f.write(f"{l}," + ",".join(f"{v:g}" for v in M[l]) + "\n")
        print(f"\n-> {a.full_csv}  ({L}x{E} 完整矩阵, stat={a.stat})")
    if a.csv:
        open(a.csv, "w").write("\n".join(csv) + "\n")
        print(f"\n-> {a.csv}")


if __name__ == "__main__":
    main()
