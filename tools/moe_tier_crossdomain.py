#!/usr/bin/env python3
"""分档 padding 表的跨领域迁移实验。

回答教授要的判据:用 A 领域 profile 出的分档表去跑 B 领域,丢多少 token?
丢得少 -> 单张图够用(路线①);丢得多 -> 必须预编译多图(路线②)。

设计:
  * 定长窗口 W:每个 prompt 截前 W 个 token,消除两个 benchmark 的长度差异
  * 每个 benchmark 按 prompt 序号切 fit/eval 两半,fit 半学表、eval 半评估
    (域内也用 held-out,否则自评有乐观偏差)
  * 分档:对每个 (layer, expert),取 fit 集上所需行数的 p95,向上取到最近档位
  * 丢弃:eval 集上 max(0, 实际行数 - 档位),按总 assignment 数归一
"""
import argparse
import itertools

import numpy as np


def windows(path, W):
    """返回 [n_win, L, E] 的行数张量:每个窗口每层每 expert 需要几行。"""
    d = np.load(path, allow_pickle=True)
    import json
    meta = json.loads(str(d["meta"]))
    E = meta["experts"]
    out = []
    for k in d.files:
        if k == "meta":
            continue
        a = d[k]                      # [L, T, K]
        if a.shape[1] < W:
            continue
        a = a[:, :W, :]               # 截到定长
        cnt = np.stack([np.bincount(a[l].ravel(), minlength=E)
                        for l in range(a.shape[0])])
        out.append(cnt)
    return np.stack(out), E           # [n, L, E]


def fit_tiers(rows, tiers, q):
    """rows [n, L, E] -> tier 表 [L, E]"""
    need = np.percentile(rows, q, axis=0)                # [L, E]
    t = np.array(tiers)
    idx = np.searchsorted(t, need, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    return t[idx]


def evaluate(rows, tier):
    """-> (丢弃率, 每层分配行数)"""
    over = np.maximum(0, rows - tier[None, :, :])
    return over.sum() / rows.sum(), tier.sum() / tier.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", default="perop_moe/routing/routing_gsm8k.npz")
    ap.add_argument("--humaneval", default="perop_moe/routing/routing_humaneval.npz")
    ap.add_argument("-W", type=int, default=48, help="定长窗口 token 数")
    ap.add_argument("-q", type=float, default=95, help="分档取的分位数")
    a = ap.parse_args()

    src = {"gsm8k": a.gsm8k, "humaneval": a.humaneval}
    data, E = {}, None
    for name, p in src.items():
        r, E = windows(p, a.W)
        n = len(r)
        data[name] = {"fit": r[: n // 2], "eval": r[n // 2:]}
        print(f"{name:>10}: {n} 个窗口 (fit {n//2} / eval {n-n//2}), "
              f"每层每 expert 平均 {r.mean():.2f} 行")
    L = data["gsm8k"]["fit"].shape[1]
    tiers = [0] + [t for t in (2, 4, 8, 16, 32) if t < a.W] + [a.W]
    base = E * a.W                                    # 当前 EP:每层 E*T 行
    print(f"\n档位 {tiers}   窗口 W={a.W}   分位 p{a.q:g}")
    print(f"当前 EP 每层执行 E*W = {E}*{a.W} = {base} 行\n")

    fits = {"gsm8k": data["gsm8k"]["fit"],
            "humaneval": data["humaneval"]["fit"],
            "union": np.concatenate([data["gsm8k"]["fit"],
                                     data["humaneval"]["fit"]])}
    print(f"{'分档表来自':<12}{'评估于':<12}{'丢弃率':>9}{'行/层':>9}{'vs当前':>9}")
    print("-" * 52)
    for fname, feval in itertools.product(fits, ["gsm8k", "humaneval"]):
        tier = fit_tiers(fits[fname], tiers, a.q)
        drop, rows = evaluate(data[feval]["eval"], tier)
        mark = "  <- 域内" if fname == feval else ("  <- 跨域" if fname != "union" else "")
        print(f"{fname:<12}{feval:<12}{drop*100:>8.2f}%{rows:>9.0f}"
              f"{base/rows:>8.2f}x{mark}")


if __name__ == "__main__":
    main()
