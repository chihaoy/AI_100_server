#!/usr/bin/env python3
"""Real-model (48-layer EP QPC) trace: per layer, per MoE MatMul node, the HMX window on every core of one card.

Answers "how are the two experts that live on one core computed?": if the stage-0 GEMMs (/mlp/MatMul_1..3) and
the stage-1 GEMMs (/mlp/MatMul_4..6) of the same layer occupy disjoint, ordered windows on the same core, the two
experts are computed serially as two separate batched nodes (batch = lanes per card, one lane per core), not as a
batch=2 GEMM on the core.

  python3 moe_ep_stage_timeline.py <merged.trace.json> [--card 0] [--layers 0,12,24,47]
"""
import argparse, collections, json, re, sys

LANE_KB = 2048 * 768 * 2 // 1024
PAT = re.compile(r"^(/model/layers\.(\d+)/mlp/MatMul(?:_\d+)?)/__(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--layers", default="0,12,24,47")
    ap.add_argument("--stage0", default="MatMul,MatMul_1,MatMul_2", help="gate,up,down node names of stage 0 (real 48-layer QPC default)")
    ap.add_argument("--stage1", default="MatMul_3,MatMul_4,MatMul_5")
    ap.add_argument("--core-detail", type=int, default=None, help="print every core's stage-0 / stage-1 GEMM window for this layer")
    a = ap.parse_args()
    S0 = set(a.stage0.split(",")); S1 = set(a.stage1.split(","))
    show = {int(x) for x in a.layers.split(",")}
    ev = json.load(open(a.trace))["traceEvents"]
    names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    pre = f"QAicGraph_slice{a.card:02d}_Core_"
    win = collections.defaultdict(lambda: [1e18, 0.0, 0, 0.0])       # (layer,node,core) -> [first, last, ntiles, hmx_dur]
    deq = collections.defaultdict(float)                              # (layer,node,core) -> KB
    for e in ev:
        if e.get("ph") != "X":
            continue
        th = names.get((e["pid"], e["tid"]), "?")
        if not th.startswith(pre):
            continue
        ar = e["args"]; k = ar.get("opKind", "")
        if k not in ("aicconvolutiond32", "blockdequantize_mxfp6"):
            continue
        m = PAT.match(ar.get("opName", ""))
        if not m:
            continue
        core = int(th[len(pre):].split("_")[0]); layer = int(m.group(2)); node = m.group(1).split("/")[-1]
        key = (layer, node, core)
        if k == "aicconvolutiond32":
            w = win[key]; w[0] = min(w[0], e["ts"]); w[1] = max(w[1], e["ts"] + e["dur"]); w[2] += 1; w[3] += e["dur"]
        else:
            try:
                deq[key] += int(ar["opPCycle"]) / float(ar["PCyclesPerKB"])
            except (KeyError, ValueError, ZeroDivisionError):
                pass
    layers = sorted({k[0] for k in win})
    nodes = sorted({k[1] for k in win}, key=lambda s: (len(s), s))
    print(f"card{a.card}: {len(layers)} layers, MoE MatMul nodes = {nodes}")
    print(f"deq KB/core == {LANE_KB} means one whole expert matrix per core; 2x that would mean 2 experts batched on the core\n")
    for L in layers:
        if L not in show:
            continue
        print(f"--- layer {L} ---")
        print(f"  {'node':10s} {'cores':>5s} {'tiles/core':>10s} {'deqKB/core min..max':>20s} {'window over cores: first..last (ms)':>40s} {'window/core span min..max':>26s}")
        for n in nodes:
            ks = [k for k in win if k[0] == L and k[1] == n]
            if not ks:
                continue
            cores = sorted(k[2] for k in ks)
            tiles = [win[k][2] for k in ks]; kb = [deq.get(k, 0) for k in ks]
            f = min(win[k][0] for k in ks); l = max(win[k][1] for k in ks)
            span = [(win[k][1] - win[k][0]) / 1e3 for k in ks]
            print(f"  {n:10s} {len(cores):5d} {min(tiles):4d}..{max(tiles):<4d} {min(kb):8.0f}..{max(kb):<8.0f} {f/1e3:18.2f}..{l/1e3:<8.2f} {min(span):10.2f}..{max(span):<8.2f}")
        # per-core ordering check: stage-0 GEMMs end vs stage-1 GEMMs start on the same core
        s0, s1 = S0, S1
        gaps = []
        for c in range(16):
            e0 = max((win[(L, n, c)][1] for n in s0 if (L, n, c) in win), default=None)
            b1 = min((win[(L, n, c)][0] for n in s1 if (L, n, c) in win), default=None)
            if e0 is not None and b1 is not None:
                gaps.append((b1 - e0) / 1e3)
            if a.core_detail == L and e0 is not None:
                b0 = min(win[(L, n, c)][0] for n in s0 if (L, n, c) in win)
                e1 = max(win[(L, n, c)][1] for n in s1 if (L, n, c) in win)
                d0 = sum(win[(L, n, c)][3] for n in s0 if (L, n, c) in win) / 1e3
                d1 = sum(win[(L, n, c)][3] for n in s1 if (L, n, c) in win) / 1e3
                print(f"    core{c:2d}: stage0 GEMM {b0/1e3:7.2f}..{e0/1e3:<7.2f} (hmx busy {d0:.2f} ms)   stage1 GEMM {b1/1e3:7.2f}..{e1/1e3:<7.2f} (hmx busy {d1:.2f} ms)   gap {(b1-e0)/1e3:+.3f}")
        if gaps:
            print(f"  per-core gap (stage1 first GEMM start - stage0 last GEMM end): min {min(gaps):+.3f} ms  max {max(gaps):+.3f} ms  cores {len(gaps)}")
    # all-layer summary of the ordering
    s0, s1 = S0, S1
    overl = 0; tot = 0; mins = []
    for L in layers:
        for c in range(16):
            e0 = max((win[(L, n, c)][1] for n in s0 if (L, n, c) in win), default=None)
            b1 = min((win[(L, n, c)][0] for n in s1 if (L, n, c) in win), default=None)
            if e0 is None or b1 is None:
                continue
            tot += 1; g = (b1 - e0) / 1e3; mins.append(g)
            if g < 0:
                overl += 1
    if tot:
        print(f"\nALL LAYERS card{a.card}: (layer,core) pairs = {tot}; stage-1 GEMM starts before stage-0 GEMM ends on {overl} of them; gap min {min(mins):+.3f} ms, median {sorted(mins)[len(mins)//2]:+.3f} ms")


if __name__ == "__main__":
    main()
