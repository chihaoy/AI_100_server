#!/usr/bin/env python3
"""Do the 16 cores of a card wait for the slowest one?  Per-core view of one tier-bench variant.

For card 0 (default) prints, per core: how many whole expert matrices it dequantizes for the gate node
(= how many experts it owns), its HMX busy time, when its last expert GEMM ends, when its Core-N-Execution
ends, and how long it spends in sync/barrier events.  If the cores synchronise, every core's execution
end is the same regardless of its own GEMM end, and the light cores show the difference as sync time.

  python3 moe_sync_straggler.py <variant_dir> [--card 0] [--sample 1]
"""
import argparse, collections, glob, json, os, re

LANE_KB = 2048 * 768 * 2 // 1024
SL = re.compile(r"QAicGraph_(?:slice(\d+)_)?Core_(\d+)(?:_(\w+))?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir"); ap.add_argument("--card", type=int, default=0); ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--gate-node", default=None, help="node whose deq KB counts experts per core (default: first MatMul_* with >=1 lane)")
    a = ap.parse_args()
    d = a.variant_dir
    tr = (sorted(glob.glob(os.path.join(d, "out", f"*inf-{a.sample}-*merged*.trace.json"))) or sorted(glob.glob(os.path.join(d, "out", "*merged*.trace.json")))
          or sorted(glob.glob(os.path.join(d, "out", "*.trace.json"))))[0]
    ev = json.load(open(tr))["traceEvents"]
    names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    C = collections.defaultdict(lambda: dict(exec_end=0, exec_start=0, gemm_last=0, gemm_first=1e18, hmx_busy=0.0, hvx_busy=0.0, sync=0.0, barrier=0.0, deq={},
                                             s0_end=0, s1_start=1e18, busy_all=0.0))
    S0 = {"/MatMul_1", "/MatMul_2", "/MatMul_3"}; S1 = {"/MatMul_4", "/MatMul_5", "/MatMul_6"}
    for e in ev:
        if e.get("ph") != "X":
            continue
        m = SL.match(names.get((e["pid"], e["tid"]), "?"))
        if not m or int(m.group(1) or 0) != a.card:
            continue
        core, eng = int(m.group(2)), m.group(3); c = C[core]; n = e["name"]; ar = e.get("args", {})
        if eng is None:
            if e.get("dur", 0) > c["exec_end"] - c["exec_start"]:
                c["exec_start"], c["exec_end"] = e["ts"], e["ts"] + e["dur"]
            continue
        if not (n.startswith("sync") or n.startswith("barrier") or "aicendcyclestats" in n):
            c["busy_all"] += e["dur"]                       # every engine thread: HMX, HVX, DMA issue, *_DMA
        if eng not in ("HMX", "HVX"):
            continue
        if n.startswith("sync"):
            c["sync"] += e["dur"]; continue
        if n.startswith("barrier"):
            c["barrier"] += e["dur"]; continue
        c[eng.lower() + "_busy"] += e["dur"]
        k = ar.get("opKind", "")
        if k == "aicconvolutiond32" and "/MatMul" in ar.get("opName", ""):
            c["gemm_last"] = max(c["gemm_last"], e["ts"] + e["dur"]); c["gemm_first"] = min(c["gemm_first"], e["ts"])
            node = ar["opName"].rsplit("/__", 1)[0]
            if node in S0:
                c["s0_end"] = max(c["s0_end"], e["ts"] + e["dur"])
            if node in S1:
                c["s1_start"] = min(c["s1_start"], e["ts"])
        if k == "blockdequantize_mxfp6":
            mm = re.match(r"(/[^/]+)/__\d+$", ar.get("opName", ""))
            if mm and "opPCycle" in ar:
                try:
                    c["deq"][mm.group(1)] = c["deq"].get(mm.group(1), 0) + int(ar["opPCycle"]) / float(ar["PCyclesPerKB"])
                except (ValueError, ZeroDivisionError):
                    pass
    info = json.load(open(os.path.join(d, "info.json")))
    gate = a.gate_node
    if gate is None:
        cand = sorted({k for c in C.values() for k in c["deq"] if c["deq"][k] > 512}, key=lambda s: (len(s), s))
        gate = cand[0] if cand else None
    t0 = min(c["exec_start"] for c in C.values())
    ends = [c["exec_end"] for c in C.values()]; gl = [c["gemm_last"] for c in C.values() if c["gemm_last"]]
    print(f"{os.path.basename(d.rstrip('/'))} card{a.card}: groups={info['groups']} S={info['S']} lanes/card={info['P']//info['num_devices']}   experts-per-core read from {gate}")
    print(f"  {'core':>4} {'experts(gate)':>13} {'hmx_busy':>9} {'busy_all':>9} {'idle':>7} {'s0_gemm_end':>11} {'s1_gemm_start':>13} {'gemm_last':>10} {'exec_end':>9}   (ms, relative to first core start; idle = exec span - sum of engine busy, engines overlap so it is a lower bound)")
    for core in sorted(C):
        c = C[core]; nexp = c["deq"].get(gate, 0) / LANE_KB if gate else float("nan")
        span = (c["exec_end"] - c["exec_start"]) / 1e3
        s0 = (c["s0_end"] - t0) / 1e3 if c["s0_end"] else float("nan"); s1 = (c["s1_start"] - t0) / 1e3 if c["s1_start"] < 1e17 else float("nan")
        print(f"  {core:>4} {nexp:13.2f} {c['hmx_busy']/1e3:9.3f} {c['busy_all']/1e3:9.3f} {span - c['busy_all']/1e3:7.2f} {s0:11.2f} {s1:13.2f} {(c['gemm_last']-t0)/1e3:10.2f} {(c['exec_end']-t0)/1e3:9.2f}")
    print(f"  gemm_last over cores: min {(min(gl)-t0)/1e3:.2f}  max {(max(gl)-t0)/1e3:.2f}  spread {(max(gl)-min(gl))/1e3:.2f} ms   |   exec_end: min {(min(ends)-t0)/1e3:.2f}  max {(max(ends)-t0)/1e3:.2f}  spread {(max(ends)-min(ends))/1e3:.2f} ms")


if __name__ == "__main__":
    main()
