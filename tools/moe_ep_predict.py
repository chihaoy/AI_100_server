#!/usr/bin/env python3
"""阶段 1: 用实测 routing 预测 expert-parallel 下的四卡负载。不碰卡。

EP 的分配是【静态】的,导出时定死:
    lane p = e mod P                    P = num_parallelized_experts
    card d = p // (P / num_devices)
P 由 QEfficient 的 _resolve_expert_parallel_layout 从
(num_experts, num_devices, num_cores, cores_per_expert) 算出。

关于"负载"的口径 —— 这一点很关键:
  1.23 的 cumsum_scatter_gather_update_expert_blocked 里
      packed_chunk_size = seq_len // num_packed_chunks
  循环覆盖【完整 seq_len】,多余行 mask 掉。所以每条 lane 每个 slot 都算满 T 行,
  **计算量固定、与 routing 无关**。

  => 卡间不均衡【不会】表现为"某张卡 idle",而是表现为"有用功占比不同"。
     本脚本因此报告:
       - 每卡分到的 (token, expert) 对数         = 有用功
       - 每卡的有用功占比                        = 有用 / 已算(padding 后)
       - 若改成按容量截断,需要多大的 capacity factor 才不丢 token

用法:
  python3 moe_ep_predict.py --routing a.npz [b.npz] [--num-devices 4] [--cores-per-expert 1]
"""
import argparse, json, sys
import numpy as np


def expert_counts(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    n = len(meta["prompts"])
    arrs = [z[f"p{i}"] for i in range(n)]
    L, _, K = arrs[0].shape
    E = meta["experts"]
    c = np.zeros((L, E), dtype=np.int64)
    for a in arrs:
        for l in range(L):
            np.add.at(c[l], a[l].ravel(), 1)
    return meta, c, arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", nargs="+", required=True)
    ap.add_argument("--num-devices", type=int, default=4)
    ap.add_argument("--num-cores", type=int, default=16)
    ap.add_argument("--cores-per-expert", type=int, default=1)
    a = ap.parse_args()

    try:
        from QEfficient.transformers.models.pytorch_transforms import _resolve_expert_parallel_layout
    except ImportError:
        sys.exit("需要 QEfficient >= 1.23 (含 transformers/moe)")

    D = a.num_devices
    first_meta, _, _ = expert_counts(a.routing[0])
    E, K = first_meta["experts"], first_meta["top_k"]
    total_cores, stages, P, eps = _resolve_expert_parallel_layout(
        num_experts=E, num_devices=D, num_cores=a.num_cores, cores_per_expert=a.cores_per_expert
    )
    lanes_per_dev = P // D
    print("=" * 84)
    print(f"EP 布局 (QEfficient _resolve_expert_parallel_layout 算出)")
    print(f"  num_experts={E}  num_devices={D}  num_cores={a.num_cores}  cores_per_expert={a.cores_per_expert}")
    print(f"  total_avl_cores={total_cores}  num_pipeline_stages={stages}")
    print(f"  num_parallelized_experts(P)={P}  experts_per_soc={eps}  每卡 lane={lanes_per_dev}")
    card_of = np.array([(e % P) // lanes_per_dev for e in range(E)])
    for d in range(D):
        own = np.flatnonzero(card_of == d)
        print(f"    card{d}: {own[:3].tolist()}…{own[-2:].tolist()}  共 {len(own)} 个 expert")
    print("=" * 84)

    for path in a.routing:
        meta, C, arrs = expert_counts(path)
        L = C.shape[0]
        # 每层每卡的 (token,expert) 对数
        load = np.zeros((L, D), dtype=np.int64)
        for e in range(E):
            load[:, card_of[e]] += C[:, e]

        agg = load.sum(0).astype(float)
        per_layer_ratio = [load[l].max() / load[l].mean() for l in range(L) if load[l].mean() > 0]
        worst = int(np.argmax(per_layer_ratio))

        print(f"\n### {meta['bench']}  —  {len(arrs)} prompts, "
              f"{sum(m['tokens'] for m in meta['prompts'])} tokens, {L} 层")
        print(f"  【全模型聚合】每卡 (token,expert) 对数")
        for d in range(D):
            print(f"    card{d}: {int(agg[d]):>10,d}   ({agg[d]/agg.mean()*100:6.2f}% of mean)")
        print(f"    卡间 max/mean = {agg.max()/agg.mean():.4f}   cv = {agg.std()/agg.mean()*100:.2f}%")

        print(f"  【逐层】max/mean")
        print(f"    平均 {np.mean(per_layer_ratio):.4f}   最差 {max(per_layer_ratio):.4f} (L{worst})"
              f"   最好 {min(per_layer_ratio):.4f} (L{int(np.argmin(per_layer_ratio))})")
        v = load[worst].astype(float)
        print(f"    最差层 L{worst} 四卡: " + "  ".join(f"{x/v.mean():.3f}" for x in v))
        print(f"    超过 1.1 的层: {sum(r > 1.1 for r in per_layer_ratio)}/{L}"
              f"   超过 1.2 的层: {sum(r > 1.2 for r in per_layer_ratio)}/{L}")

        # 有用功占比: 每卡每层已算 = lanes_per_dev * stages * T 行 = (E/D) * T
        toks = np.array([m["tokens"] for m in meta["prompts"]])
        T_tot = int(toks.sum())
        # 每卡每层算 (E/D) x T 行(padding 后); agg 是 48 层的总和,分母也要乘 L
        padded_per_card = (E // D) * T_tot * L
        print(f"  【有用功占比】(padding 后每卡每层算 {E//D} x T 行, 共 {L} 层)")
        for d in range(D):
            print(f"    card{d}: {agg[d]/padded_per_card*100:5.2f}%", end="")
        print(f"    (理想均衡时 {K/E*100:.2f}%)")

        # 若改成按容量截断,需要多大 capacity factor
        cf_needed = []
        for l in range(L):
            mean_per_card = load[l].mean()
            cf_needed.append(load[l].max() / mean_per_card if mean_per_card > 0 else 1.0)
        print(f"  【若改成容量截断】per-card capacity factor 需要 "
              f"{max(cf_needed):.2f} 才能零丢弃 (平均 {np.mean(cf_needed):.2f})")

    print("\n" + "=" * 84)
    print("判读:")
    print("  * 当前 1.23 的 kernel 把每条 lane 都算满 T 行(masked),所以卡间【cycle 均匀】,")
    print("    上面的不均衡表现为【有用功占比不同】,不是 idle。")
    print("  * 若未来把 chunk 循环改成按 per-card 容量截断,上面的 capacity factor 就是")
    print("    需要预留的余量;那时不均衡才会变成真实的时间损失。")


if __name__ == "__main__":
    main()
