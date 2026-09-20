#!/usr/bin/env python3
"""Phase timeline per card from a qaic-opstats merged trace of a tier-bench variant.

Splits each card's inference into
  expert phase : start .. last aicconvolutiond32 end (cumsum / gather / GEMM / scatter of all stages)
  reduce phase : .. end of Core-0-Execution (lane reduce einsum + cross-card reduce + output)
and reports, per card, the per-core span of the expert phase (to see core imbalance inside it).

  python3 moe_tier_bench_phases.py <variant_dir> [--core-detail]
"""
import argparse, collections, glob, json, os, re, statistics, sys


def load(trace):
    t = json.load(open(trace)); ev = t["traceEvents"]
    names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    byth = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") == "X":
            byth[names.get((e["pid"], e["tid"]), "?")].append(e)
    return byth


SL = re.compile(r"QAicGraph_(?:slice(\d+)_)?Core_(\d+)(?:_(\w+))?$")


def analyse(byth):
    cards = collections.defaultdict(lambda: collections.defaultdict(dict))   # card -> core -> info
    for th, L in byth.items():
        m = SL.match(th)
        if not m:
            continue
        card, core, eng = int(m.group(1) or 0), int(m.group(2)), m.group(3)
        d = cards[card][core]
        if eng is None:                                  # Core-N-Execution span
            e = max(L, key=lambda e: e.get("dur", 0))
            d["exec_start"], d["exec_end"] = e["ts"], e["ts"] + e["dur"]
        elif eng in ("HMX", "HVX"):
            g = [e for e in L if e["name"].startswith("aicconvolutiond32")]
            if g:
                d["gemm_first"] = min(d.get("gemm_first", 1e18), min(e["ts"] for e in g))
                d["gemm_last"] = max(d.get("gemm_last", 0), max(e["ts"] + e["dur"] for e in g))
                d["gemm_n"] = d.get("gemm_n", 0) + len(g)
                d["gemm_dur"] = d.get("gemm_dur", 0) + sum(e["dur"] for e in g)
            r = [e for e in L if e["name"].startswith("aicbatchedreduceadd") and e["dur"] > 50]
            if r:
                d["red_first"] = min(d.get("red_first", 1e18), min(e["ts"] for e in r))
                d["red_last"] = max(d.get("red_last", 0), max(e["ts"] + e["dur"] for e in r))
                d["red_dur"] = d.get("red_dur", 0) + sum(e["dur"] for e in r)
            c = [e for e in L if e["name"].startswith("cumsum")]
            if c:
                d["cumsum_dur"] = d.get("cumsum_dur", 0) + sum(e["dur"] for e in c)
            dq = [e for e in L if e["name"].startswith("blockdequantize")]
            if dq:
                d["deq_dur"] = d.get("deq_dur", 0) + sum(e["dur"] for e in dq)
            busy = sum(e["dur"] for e in L if not (e["name"].startswith("sync") or e["name"].startswith("barrier")))
            d["busy_" + eng] = busy
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--core-detail", action="store_true")
    ap.add_argument("--sample", type=int, default=1)
    a = ap.parse_args()
    tr = sorted(glob.glob(os.path.join(a.variant_dir, "out", f"*inf-{a.sample}-*merged*.trace.json")))
    if not tr:
        tr = sorted(glob.glob(os.path.join(a.variant_dir, "out", "*merged*.trace.json")))
    if not tr:   # single-card runs: no merge step, no slice id in thread names
        tr = sorted(glob.glob(os.path.join(a.variant_dir, "out", f"*inf-{a.sample}-*.trace.json"))) or sorted(glob.glob(os.path.join(a.variant_dir, "out", "*.trace.json")))
    if not tr:
        sys.exit("no merged trace")
    info = json.load(open(os.path.join(a.variant_dir, "info.json")))
    cards = analyse(load(tr[0]))
    print(f"{os.path.basename(a.variant_dir.rstrip('/'))}: groups={info['groups']} rows={info['padded_rows_total']} ({info['row_ratio']:.3f}x)   [{os.path.basename(tr[0])[:40]}]")
    print(f"{'card':>4} {'exec_end':>9} {'gemm_last':>10} {'expert%':>8} {'reduce_ms':>10} {'gemm_last min/max over cores':>30} {'gemm dur/core min/max':>24} {'HVX+HMX busy/core min/max':>26}")
    for card, cores in sorted(cards.items()):
        ee = max(d.get("exec_end", 0) for d in cores.values())
        gl = [d["gemm_last"] for d in cores.values() if "gemm_last" in d]
        gd = [d["gemm_dur"] for d in cores.values() if "gemm_dur" in d]
        busy = [d.get("busy_HVX", 0) + d.get("busy_HMX", 0) for d in cores.values()]
        rf = [d["red_first"] for d in cores.values() if "red_first" in d]
        gmax = max(gl) if gl else 0
        print(f"{card:>4} {ee/1e3:>8.2f}ms {gmax/1e3:>9.2f}ms {gmax/ee:>7.1%} {(ee-gmax)/1e3:>9.2f}ms {min(gl)/1e3:>13.2f}/{max(gl)/1e3:<14.2f} {min(gd)/1e3:>10.2f}/{max(gd)/1e3:<12.2f} {min(busy)/1e3:>11.2f}/{max(busy)/1e3:<12.2f}")
        if a.core_detail:
            for core, d in sorted(cores.items()):
                print(f"      c{core:<2} exec {d.get('exec_end',0)/1e3:6.2f}ms  gemm[{d.get('gemm_first',0)/1e3:6.2f}..{d.get('gemm_last',0)/1e3:6.2f}] n={d.get('gemm_n',0):3d} dur={d.get('gemm_dur',0)/1e3:5.2f}ms  deq={d.get('deq_dur',0)/1e3:5.2f}ms  cumsum={d.get('cumsum_dur',0)/1e3:5.2f}ms  reduce[{d.get('red_first',0)/1e3:6.2f}..{d.get('red_last',0)/1e3:6.2f}] dur={d.get('red_dur',0)/1e3:5.2f}ms  busy HVX={d.get('busy_HVX',0)/1e3:5.2f} HMX={d.get('busy_HMX',0)/1e3:5.2f}")


if __name__ == "__main__":
    main()
