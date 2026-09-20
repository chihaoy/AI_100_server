#!/usr/bin/env python3
"""Per-ONNX-node wall-clock spans on one core, from a merged qaic-opstats trace.

For each graph node that runs on the chosen (card, core), print: number of DMA events and their
summed duration, HMX and HVX compute durations, first/last timestamp and the span. Sorted by
first timestamp, so the output reads as the core's timeline through the MoE layer.

  python3 moe_tier_bench_nodespans.py <variant_dir> [--card 0] [--core 5] [--min-span 0.05]
"""
import argparse, collections, glob, json, os, re, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir"); ap.add_argument("--card", type=int, default=0); ap.add_argument("--core", type=int, default=5)
    ap.add_argument("--min-span", type=float, default=0.05, help="ms; hide nodes with shorter spans")
    ap.add_argument("--sample", type=int, default=1)
    a = ap.parse_args()
    tr = sorted(glob.glob(os.path.join(a.variant_dir, "out", f"*inf-{a.sample}-*merged*.trace.json"))) or sorted(glob.glob(os.path.join(a.variant_dir, "out", "*merged*.trace.json"))) or sorted(glob.glob(os.path.join(a.variant_dir, "out", f"*inf-{a.sample}-*.trace.json"))) or sorted(glob.glob(os.path.join(a.variant_dir, "out", "*.trace.json")))
    t = json.load(open(tr[0])); ev = t["traceEvents"]
    names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    inv = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X":
            inv[names.get((e["pid"], e["tid"]), "?")].append(e)
    node = lambda n: re.sub(r"^libjit_\S+ libjit_\S+ ", "", re.sub(r"/__\d+$", "", n or ""))
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 1e18, 0.0, ""])
    pre = f"QAicGraph_slice{a.card:02d}_Core_{a.core}_"
    if not any(k.startswith(pre) for k in inv):          # single-card traces have no slice id
        pre = f"QAicGraph_Core_{a.core}_"
    for eng in ["DMAIssue_DMA", "HVX_DMA", "HMX", "HVX"]:
        for e in inv[pre + eng]:
            if e["name"].startswith("sync") or e["name"].startswith("barrier") or "aicendcyclestats" in e["name"]:
                continue
            n = node(e["args"].get("opName", ""))
            r = agg[n]
            if eng.endswith("_DMA"):
                r[0] += 1; r[1] += e["dur"]
            elif eng == "HMX":
                r[2] += e["dur"]
            else:
                r[3] += e["dur"]
            r[4] = min(r[4], e["ts"]); r[5] = max(r[5], e["ts"] + e["dur"]); r[6] = e["name"].split()[0][:18]
    ex = max(inv[pre[:-1]], key=lambda e: e.get("dur", 0))
    print(f"{os.path.basename(a.variant_dir.rstrip('/'))} card{a.card} core{a.core}: execution {ex['ts']/1e3:.2f}..{(ex['ts']+ex['dur'])/1e3:.2f} ms")
    print(f"  {'node':28s} {'dma_n':>5s} {'dma_ms':>7s} {'hmx_ms':>7s} {'hvx_ms':>7s} {'first':>7s} {'last':>7s} {'span':>6s}  kernel")
    for k in sorted(agg, key=lambda k: agg[k][4]):
        r = agg[k]
        if (r[5] - r[4]) / 1e3 < a.min_span:
            continue
        print(f"  {k[:28]:28s} {r[0]:5d} {r[1]/1e3:7.3f} {r[2]/1e3:7.3f} {r[3]/1e3:7.3f} {r[4]/1e3:7.2f} {r[5]/1e3:7.2f} {(r[5]-r[4])/1e3:6.2f}  {r[6]}")


if __name__ == "__main__":
    main()
