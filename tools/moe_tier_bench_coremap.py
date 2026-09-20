#!/usr/bin/env python3
"""How does the compiler place expert GEMMs onto the 16 cores of one card?

Reads the merged qaic-opstats trace of one tier_bench variant and, for every MatMul node
(one node = one lane group x one projection), reports per card:
  * how many HMX conv tiles the node has, how many cores run them, tiles per core
  * how many KB of MXFP6 weight each core dequantizes for that node
    (one expert matrix gate/up/down = 2048*768*2 B = 3072 KB in fp16)
  * the wall-clock window of the node's HMX tiles, to see whether nodes run sequentially

Reading the numbers:  per-core deq KB == 3072  -> that core owns one whole expert matrix
                      per-core deq KB == 3072*L/16 -> the group's L lanes/card are column-split
                                                     evenly over all 16 cores (no lane <-> core)

  python3 moe_tier_bench_coremap.py <variant_dir> [--card 0] [--timeline]
"""
import argparse, collections, glob, json, os, re

LANE_KB = 2048 * 768 * 2 // 1024  # 3072


def load(var, card):
    tr = (glob.glob(os.path.join(var, "out", "*inf-1-*merged*.trace.json"))
          or glob.glob(os.path.join(var, "out", "*inf-1-*.trace.json"))
          or glob.glob(os.path.join(var, "out", "*.trace.json")))[0]
    ev = json.load(open(tr))["traceEvents"]
    names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    pre = f"QAicGraph_slice{card:02d}_Core_"
    if not any(v.startswith(pre) for v in names.values()):
        pre = "QAicGraph_Core_"
    conv = collections.defaultdict(list)                  # (node, core) -> [(tid, ts, dur)]
    deq = collections.defaultdict(lambda: [0, 0.0])        # (node, core) -> [n, KB]
    for e in ev:
        if e.get("ph") != "X":
            continue
        th = names.get((e["pid"], e["tid"]), "?")
        if not th.startswith(pre):
            continue
        a = e["args"]; k = a.get("opKind", "")
        if k not in ("aicconvolutiond32", "blockdequantize_mxfp6") or "opPCycle" not in a:
            continue
        m = re.match(r"(/[^/]+)/__(\d+)$", a["opName"])
        if not m:
            continue
        core = int(th[len(pre):].split("_")[0]); node = m.group(1)
        if k == "aicconvolutiond32":
            conv[(node, core)].append((int(m.group(2)), e["ts"], e["dur"]))
        else:
            r = deq[(node, core)]; r[0] += 1
            try:
                r[1] += int(a["opPCycle"]) / float(a["PCyclesPerKB"])
            except (ValueError, ZeroDivisionError):
                pass
    return tr, conv, deq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir"); ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--timeline", action="store_true", help="also print each node's HMX window in start order")
    ap.add_argument("--per-core", action="store_true", help="print every core's tile ids for each node")
    a = ap.parse_args()
    tr, conv, deq = load(a.variant_dir, a.card)
    info = json.load(open(os.path.join(a.variant_dir, "info.json")))
    nd = info.get("num_devices", 1)
    print(f"{os.path.basename(a.variant_dir.rstrip('/'))} card{a.card}: groups(width, lanes)={info['groups']} devices={nd}")
    print(f"  {'node':11s} {'tiles':>5s} {'cores':>5s} {'conv/core':>9s} {'deqKB/core':>14s}  lanes/card->expected KB/core if column-split")
    nodes = sorted({n for n, _ in conv}, key=lambda s: (len(s), s))
    for n in nodes:
        cores = sorted(c for (nn, c) in conv if nn == n)
        tiles = len({t for (nn, c), v in conv.items() if nn == n for t, _, _ in v})
        pc = [len(conv[(n, c)]) for c in cores]; kb = [deq.get((n, c), [0, 0])[1] for c in cores]
        tot_kb = sum(kb); lanes = tot_kb / LANE_KB
        print(f"  {n:11s} {tiles:5d} {len(cores):5d} {min(pc):4d}..{max(pc):<4d} {min(kb):6.0f}..{max(kb):<6.0f}  "
              f"{lanes:4.1f} lanes -> {tot_kb/16:6.0f} KB/core" + ("   <== 1 expert per core" if abs(min(kb) - LANE_KB) < 8 and abs(max(kb) - LANE_KB) < 8 else ""))
        if a.per_core:
            for c in cores:
                tids = sorted(t for t, _, _ in conv[(n, c)])
                print(f"      core{c:2d}: {len(tids)} tiles ids {tids[0]}..{tids[-1]}  deq {deq.get((n, c), [0, 0])[1]:.0f} KB")
    if a.timeline:
        print("\n  HMX conv window per node (ms), start order; 'gap' = start minus previous node's end")
        span = {}
        for (n, c), v in conv.items():
            s = span.setdefault(n, [1e18, 0])
            for _, ts, d in v:
                s[0] = min(s[0], ts); s[1] = max(s[1], ts + d)
        prev = None
        for n, (s, e) in sorted(span.items(), key=lambda kv: kv[1][0]):
            gap = "" if prev is None else f"gap {(s-prev)/1e3:+.2f}"
            print(f"  {n:11s} {s/1e3:6.2f}..{e/1e3:6.2f}  {gap}")
            prev = max(prev or 0, e)


if __name__ == "__main__":
    main()
