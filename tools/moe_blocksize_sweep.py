#!/usr/bin/env python3
"""块大小 sweep:MegaBlocks 式"补齐到块的倍数"在本模型形状上的收益。

对比基线是 QEfficient EXPERT_PARALLEL 的实际行为:每层执行 E * T_pad 行
(与路由无关),T_pad = 静态 seq_len。

用法:
    python3 tools/moe_blocksize_sweep.py perop_moe/routing/routing_gsm8k.npz \
                                         perop_moe/routing/routing_humaneval.npz

输入 npz 由 tools/moe_collect_routing.py 产生,键 p0..pN 形状 [L, T, K]。
"""
import argparse
import json
import os

import numpy as np

BLOCKS = [128, 64, 32, 16, 8, 4, 1]


def sweep(path, t_pad):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    E, K = meta["experts"], meta["top_k"]
    base_rows = E * t_pad

    tot = {b: 0 for b in BLOCKS}
    ideal = base = nlay = alive = 0
    skipped = 0
    for k in d.files:
        if k == "meta":
            continue
        a = d[k]  # [L, T, K]
        L, T, _ = a.shape
        if T > t_pad:  # 超出静态窗口的 prompt 不可比,跳过
            skipped += 1
            continue
        for l in range(L):
            cnt = np.bincount(a[l].ravel(), minlength=E)  # 每 expert 分到多少行
            ideal += int(cnt.sum())
            base += base_rows
            alive += int((cnt > 0).sum())
            nlay += 1
            for b in BLOCKS:
                tot[b] += int((np.ceil(cnt / b) * b).sum())
    return dict(meta=meta, E=E, K=K, tot=tot, ideal=ideal, base=base,
                nlay=nlay, alive=alive, skipped=skipped)


def report(name, r):
    nlay, base, ideal = r["nlay"], r["base"], r["ideal"]
    print(f"--- {name}:  {nlay} 层实例 (E={r['E']}, top_k={r['K']}"
          f"{', 跳过 %d 个超窗 prompt' % r['skipped'] if r['skipped'] else ''}) ---")
    print(f"  平均每层活跃 expert: {r['alive']/nlay:.1f} / {r['E']}")
    print(f"  平均每活跃 expert 行数: {ideal/r['alive']:.1f}")
    print(f"  {'块大小':<12}{'行/层':>12}{'vs 当前':>10}{'利用率':>10}")
    print(f"  {'当前(EP)':<12}{base/nlay:>12.0f}{1.0:>9.2f}x{ideal/base*100:>9.1f}%")
    for b in BLOCKS:
        lab = f"MB-{b}" if b > 1 else "理想(无补齐)"
        print(f"  {lab:<12}{r['tot'][b]/nlay:>12.0f}"
              f"{base/r['tot'][b]:>9.2f}x{ideal/r['tot'][b]*100:>9.1f}%")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--t-pad", type=int, default=128,
                    help="编译时的静态 seq_len (默认 128)")
    a = ap.parse_args()
    for p in a.npz:
        report(os.path.basename(p).replace("routing_", "").replace(".npz", ""),
               sweep(p, a.t_pad))


if __name__ == "__main__":
    main()
