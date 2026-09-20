#!/usr/bin/env python3
"""Placement probe for the BATCHED graph: which core executes /taps.<g>.<p>/Mul (lane p's slice of the
batched down-proj output)?  python3 moe_probe_tap2core.py <variant_dir> [--card N]"""
import argparse, collections, glob, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("var"); ap.add_argument("--card", type=int, default=0); a = ap.parse_args()
tr = (glob.glob(os.path.join(a.var, "out", "*inf-1-*merged*.trace.json")) or glob.glob(os.path.join(a.var, "out", "*.trace.json")))[0]
ev = json.load(open(tr))["traceEvents"]
names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
pre = f"QAicGraph_slice{a.card:02d}_Core_"
if not any(v.startswith(pre) for v in names.values()): pre = "QAicGraph_Core_"
TAP = re.compile(r"/taps\.(\d+)\.(\d+)(_\d+)?/")
tally = collections.defaultdict(collections.Counter); kinds = collections.Counter()
for e in ev:
    if e.get("ph") != "X": continue
    th = names.get((e["pid"], e["tid"]), "?")
    if not th.startswith(pre): continue
    m = TAP.search(e["args"].get("opName", ""))
    if not m: continue
    core = int(re.match(r"\d+", th[len(pre):]).group())
    lane = int(m.group(2)) + (100 if m.group(3) else 0)
    tally[lane][core] += 1; kinds[e["args"].get("opKind")] += 1
print(f"{os.path.basename(a.var)} card{a.card}: tap op kinds {dict(kinds)}")
print("lane -> {core: n_events}")
for lane in sorted(tally):
    print(f"  lane {lane:3d}: {dict(sorted(tally[lane].items()))}")
