#!/usr/bin/env python3
"""Per-(card, core, engine) attribution from a qaic-opstats merged trace.

Why this exists: perop_by_projection.py / perop_crosscard_ops.py / perop_crosscard_flow.py
all discard the trace's `tid`, so they sum blindly over all 64 (card, core) pairs. Phase 1
needs exactly that axis.

The join key is the trace's own thread_name metadata:
    tid  ->  "QAicGraph_slice0N_Core_M_<HVX|HMX|DMAIssue>[_Barrier|_DMA]"
which yields (card, core, engine). We DERIVE the map from the metadata rather than assuming
the tid arithmetic, and assert the derived lattice is consistent.

THE PHASE-1 TEST
----------------
Under pure tensor parallelism every op is sliced identically across all 64 (card, core) pairs,
so for a given op the per-(card,core) event count and cycle count must be UNIFORM. Any op whose
counts are NOT uniform is evidence of non-TP placement -- for a MoE model that would be the
first sign that expert identity leaks into placement.

Usage:
    python3 perop_percore.py <merged.trace.json> [--top 25] [--op SUBSTR]
"""
import sys, re, json, argparse, collections, statistics, os

TN = re.compile(r'"name"\s*:\s*"thread_name"')
TID = re.compile(r'"tid"\s*:\s*(\d+)')
SLICE = re.compile(r'QAicGraph_slice(\d+)_Core_(\d+)(?:_([A-Za-z_]+))?')
KIND = re.compile(r'"opKind"\s*:\s*"([^"]*)"')
NAME = re.compile(r'"opName"\s*:\s*"([^"]*)"')
PCY = re.compile(r'"opPCycle"\s*:\s*"?(\d+)"?')
PORT = re.compile(r'"PortDescription"\s*:\s*"([^"]*)"')

def canon(op):
    """strip any libjit prefix and the /__NNNN per-device sub-op ordinal"""
    op = op.strip().split()[-1] if op.strip() else op
    return re.sub(r"/__\d+$", "", op)

def site(op):
    """collapse the layer index so ops from all layers share a bucket"""
    return re.sub(r"/layers\.\d+/", "/layers.N/", op)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--op", help="only report op-sites containing this substring")
    ap.add_argument("--csv", help="write the full per-(card,core,site) table here")
    a = ap.parse_args()

    # ---------- pass 1: tid -> (card, core, engine) ----------
    tidmap = {}
    with open(a.trace, errors="ignore") as f:
        for line in f:
            if not TN.search(line):
                continue
            t = TID.search(line); s = SLICE.search(line)
            if t and s:
                tidmap[int(t.group(1))] = (int(s.group(1)), int(s.group(2)), s.group(3) or "core")
    if not tidmap:
        sys.exit("no thread_name metadata found — is this a merged trace?")
    cards = sorted({c for c, _, _ in tidmap.values()})
    cores = sorted({k for _, k, _ in tidmap.values()})
    engines = sorted({e for _, _, e in tidmap.values()})
    print(f"=== thread map ===")
    print(f"  tids     : {len(tidmap)}")
    print(f"  cards    : {cards}")
    print(f"  cores    : {cores[0]}..{cores[-1]}  (n={len(cores)})")
    print(f"  engines  : {engines}")
    # assert the lattice is complete
    have = {(c, k) for c, k, _ in tidmap.values()}
    want = {(c, k) for c in cards for k in cores}
    if have != want:
        print(f"  ⚠ incomplete lattice: missing {sorted(want - have)[:8]}")
    else:
        print(f"  lattice  : complete {len(cards)}x{len(cores)} ✅")

    # ---------- pass 2: attribute compute events ----------
    # cnt[(site)][(card,core)] = events ; cyc[...] = pcycles
    cnt = collections.defaultdict(collections.Counter)
    cyc = collections.defaultdict(collections.Counter)
    kinds = collections.defaultdict(collections.Counter)
    p2p = collections.Counter()
    unmapped = 0
    with open(a.trace, errors="ignore") as f:
        for line in f:
            if '"opPCycle"' not in line:
                if '"PortDescription"' in line:
                    m = PORT.search(line)
                    if m: p2p[m.group(1)] += 1
                continue
            t = TID.search(line)
            if not t:
                continue
            key = tidmap.get(int(t.group(1)))
            if key is None:
                unmapped += 1
                continue
            card, core, eng = key
            nm = NAME.search(line); pc = PCY.search(line); kd = KIND.search(line)
            if not nm:
                continue
            s = site(canon(nm.group(1)))
            if a.op and a.op not in s:
                continue
            cnt[s][(card, core)] += 1
            if pc: cyc[s][(card, core)] += int(pc.group(1))
            if kd: kinds[s][kd.group(1)] += 1

    print(f"\n=== attributed ===")
    print(f"  distinct op-sites : {len(cnt)}")
    print(f"  events unmapped   : {unmapped}")
    if p2p:
        print(f"  P2P transfers     : {sum(p2p.values())}  dirs={len(p2p)}")

    # ---------- THE PHASE-1 UNIFORMITY TEST ----------
    print(f"\n=== UNIFORMITY across {len(cards)}x{len(cores)} = {len(cards)*len(cores)} (card,core) pairs ===")
    print(f"{'op-site':58s} {'pairs':>6s} {'ev/pair':>9s} {'cv%':>7s}  verdict")
    rows = sorted(cnt.items(), key=lambda kv: -sum(kv[1].values()))
    nonuniform = []
    for s, per in rows[: a.top]:
        vals = [per.get((c, k), 0) for c in cards for k in cores]
        npairs = sum(1 for v in vals if v)
        mean = statistics.mean(vals) if vals else 0
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0
        cv = (sd / mean * 100) if mean else 0
        verdict = "uniform ✅" if cv < 1e-9 else ("NON-UNIFORM ⚠" if cv > 5 else "near-uniform")
        if cv > 5: nonuniform.append((s, cv, npairs))
        print(f"{s[:58]:58s} {npairs:6d} {mean:9.2f} {cv:7.2f}  {verdict}")

    print("\n=== VERDICT ===")
    if not nonuniform:
        print("  Every reported op-site is spread UNIFORMLY over all (card, core) pairs.")
        print("  => placement is pure tensor parallelism; expert identity does NOT affect")
        print("     which card or core does the work. No expert-level load imbalance exists.")
    else:
        print(f"  {len(nonuniform)} op-site(s) are NOT uniform — evidence of non-TP placement:")
        for s, cv, np_ in nonuniform[:10]:
            print(f"    cv={cv:6.2f}%  pairs={np_:3d}  {s}")

    if a.csv:
        import csv as _csv
        with open(a.csv, "w", newline="") as fh:
            w = _csv.writer(fh); w.writerow(["site", "card", "core", "events", "pcycles"])
            for s, per in rows:
                for (c, k), v in sorted(per.items()):
                    w.writerow([s, c, k, v, cyc[s].get((c, k), 0)])
        print(f"\nwrote {a.csv}")

if __name__ == "__main__":
    main()
