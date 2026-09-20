#!/usr/bin/env python3
"""用实测 routing 算 EP 在不同 decode batch size 下的卡间负载与 expert 覆盖率。

decode 时 batch=B 表示 B 条独立序列各出 1 个 token。从真实 routing 里随机抽 B 个
token(尽量来自不同 prompt),按 EP 的静态分配 card d = (e mod P)//(P/D) 折算每卡负载。

回答: EP 在大 batch decode 下还有没有意义。
"""
import argparse, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--routing", nargs="+", required=True)
ap.add_argument("--num-devices", type=int, default=4)
ap.add_argument("--num-cores", type=int, default=16)
ap.add_argument("--cores-per-expert", type=int, default=1)
ap.add_argument("--trials", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from QEfficient.transformers.models.pytorch_transforms import _resolve_expert_parallel_layout

rng = np.random.default_rng(a.seed)
D = a.num_devices
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

for path in a.routing:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"])); n = len(meta["prompts"])
    E, K, L = meta["experts"], meta["top_k"], meta["layers"]
    _, stages, P, eps = _resolve_expert_parallel_layout(
        num_experts=E, num_devices=D, num_cores=a.num_cores, cores_per_expert=a.cores_per_expert)
    lanes_per_dev = P // D
    card_of = np.array([(e % P) // lanes_per_dev for e in range(E)])

    # 把所有 (prompt, position) 的 top-k 摊平成一个 token 池, 保留层维度
    arrs = [z[f"p{i}"] for i in range(n)]              # 每个 [L, T_i, K]
    pool = np.concatenate([a_[:, :, :] for a_ in arrs], axis=1)   # [L, N, K]
    N = pool.shape[1]

    print("=" * 78)
    print(f"### {meta['bench']}   token 池 N={N}, {L} 层, {E} experts, top-{K}")
    print(f"    EP 布局: P={P} lanes, {stages} stages, 每卡 {lanes_per_dev} lanes / {E//D} experts")
    print()
    print(f"{'batch':>6} {'覆盖率':>8} {'卡间max/mean':>13} {'最忙卡占比':>11} {'某卡全idle概率':>14}")
    for B in BATCHES:
        if B > N: break
        ratios, covs, idle = [], [], 0
        for _ in range(a.trials):
            idx = rng.choice(N, size=B, replace=False)
            l = rng.integers(0, L)                      # 随机取一层
            sel = pool[l, idx, :].ravel()               # [B*K] 被选中的 expert
            load = np.bincount(card_of[sel], minlength=D).astype(float)
            covs.append(len(np.unique(sel)) / E)
            ratios.append(load.max() / load.mean())
            if load.min() == 0: idle += 1
        print(f"{B:6d} {np.mean(covs)*100:7.1f}% {np.mean(ratios):13.3f} "
              f"{np.mean(ratios)/D*100:10.1f}% {idle/a.trials*100:13.1f}%")
    print()

print("=" * 78)
print("判读:")
print("  卡间 max/mean → 1.0 表示负载均衡, EP 的不均衡问题消失")
print("  覆盖率 → 100% 表示所有 expert 都被碰到, 此时 EP 与 TP 的权重搬运量相同")
print("  两者同时发生 = EP 既不再有害, 也不再有任何好处")
