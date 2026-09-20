#!/usr/bin/env python3
"""按 (op-site, card) 汇总 perop_percore.py 产出的 CSV。

每行 = 该 op 在一张卡上 16 个核的合计。只保留真正的模型 op(名字以 /model/ 或 /lm_head 开头);
CSV 里其余数千条是 inputseminc_* / outputseminc_* / aicopstats 这类运行时记账项。

用法: python3 perop_percard_table.py <percore.csv> [--metric pcycles|events] [--top N]
"""
import csv, sys, collections, argparse

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--metric", default="pcycles", choices=["pcycles", "events"])
ap.add_argument("--top", type=int, default=0, help="0 = 全部")
a = ap.parse_args()

rows = [x for x in csv.DictReader(open(a.csv))
        if x["site"].startswith(("/model/", "/lm_head"))]
if not rows:
    sys.exit("没有模型 op —— 检查 CSV 是不是 perop_percore.py 的输出")

agg = collections.defaultdict(collections.Counter)
for x in rows:
    agg[x["site"]][int(x["card"])] += int(x[a.metric])
cards = sorted({int(x["card"]) for x in rows})
order = sorted(agg, key=lambda s: -sum(agg[s].values()))
if a.top:
    order = order[: a.top]
tot = sum(sum(v.values()) for v in agg.values())

print(f"# {a.csv}")
print(f"# 模型 op-site: {len(agg)} 个   指标: {a.metric}   每格 = 该 op 在该卡各核的合计")
print(f"# 注意: pcycles 含计时抖动(单核 ±5%);要证明放置请用 SDK summary 的 count / out KB(静态量)")
print()
w = 46
print(f"{'op-site':{w}s} " + " ".join(f"{'card'+str(c):>12s}" for c in cards) + f" {'占比':>7s} {'极差':>7s}")
for s in order:
    v = [agg[s].get(c, 0) for c in cards]
    sp = (max(v) / min(v) - 1) * 100 if min(v) > 0 else float("inf")
    name = s.replace("/model/layers.N/", "L.N/")
    print(f"{name[:w]:{w}s} " + " ".join(f"{x:12,d}" for x in v)
          + f" {sum(v)/tot*100:6.2f}% {sp:6.2f}%")
