#!/usr/bin/env python3
"""Placement probe, core level: which core runs which lane, read straight off the node names.

For a variant exported with --split-lanes every GEMM node is named /lanes.<p>/... . Tally HMX conv tiles
and dequantised weight KB per (core, lane) from the merged opstats trace.

  python3 moe_probe_lane2core.py <variant_dir> [--card N]
"""
import argparse, collections, glob, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("var"); ap.add_argument("--card", type=int, default=0)
a = ap.parse_args()
tr = (glob.glob(os.path.join(a.var, "out", "*inf-1-*merged*.trace.json")) or glob.glob(os.path.join(a.var, "out", "*.trace.json")))[0]
ev = json.load(open(tr))["traceEvents"]
names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
pre = f"QAicGraph_slice{a.card:02d}_Core_"
if not any(v.startswith(pre) for v in names.values()):
    pre = "QAicGraph_Core_"
LANE = re.compile(r"/lanes\.(\d+)\.(\d+)(_\d+)?/")   # group, lane, stage suffix
conv = collections.defaultdict(int); deq = collections.defaultdict(float); win = collections.defaultdict(lambda: [1e18, 0])
other = collections.Counter()
for e in ev:
    if e.get("ph") != "X": continue
    th = names.get((e["pid"], e["tid"]), "?")
    if not th.startswith(pre): continue
    ar = e["args"]; k = ar.get("opKind", "")
    if k not in ("aicconvolutiond32", "blockdequantize_mxfp6") or "opPCycle" not in ar: continue
    core = int(re.match(r"\d+", th[len(pre):]).group())
    m = LANE.search(ar.get("opName", ""))
    if not m:
        other[(k, ar.get("opName", "").split("/")[1] if "/" in ar.get("opName", "") else ar.get("opName"))] += 1; continue
    lane = int(m.group(2)) + (100 if m.group(3) else 0)          # stage 1 lanes shown as 1xx
    if k == "aicconvolutiond32":
        conv[(core, lane)] += 1
        w = win[lane]; w[0] = min(w[0], e["ts"]); w[1] = max(w[1], e["ts"] + e.get("dur", 0))
    else:
        deq[(core, lane)] += float(ar["opPCycle"]) / float(ar.get("PCyclesPerKB", 1)) if "PCyclesPerKB" in ar else 0
cores = sorted({c for c, _ in conv}); lanes = sorted({l for _, l in conv})
print(f"{os.path.basename(a.var)} card{a.card}: {len(lanes)} lanes seen, {len(cores)} cores with conv tiles")
print("conv tiles per (core, lane):  rows=core, cols=lane")
print("      " + " ".join(f"{l:3d}" for l in lanes))
for c in range(16):
    print(f"core{c:2d} " + " ".join(f"{conv.get((c, l), 0):3d}" if conv.get((c, l), 0) else "  ." for l in lanes))
if deq:
    print("dequant KB per (core, lane):")
    print("      " + " ".join(f"{l:5d}" for l in lanes))
    for c in range(16):
        print(f"core{c:2d} " + " ".join(f"{deq.get((c, l), 0):5.0f}" if deq.get((c, l), 0) else "    ." for l in lanes))
print("\nHMX window per lane (ms from trace start, order of start):")
t0 = min(w[0] for w in win.values())
for l, w in sorted(win.items(), key=lambda kv: kv[1][0]):
    print(f"  lane {l:2d}: {(w[0]-t0)/1e3:7.3f} .. {(w[1]-t0)/1e3:7.3f}   cores {sorted(c for (c, ll) in conv if ll == l)}")
if other: print("\nconv/deq ops not under /lanes.*/:", dict(other))
