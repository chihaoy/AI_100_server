#!/usr/bin/env python3
"""Per-partition, per-card timeline from a merged qaic-opstats trace of a multi-partition (MDP manual partition) QPC:
core execution span, first/last GEMM (aicconvolutiond32), so one can see whether Partition1 starts before Partition0 ends.
  python3 tools/moe_mdp_partition_spans.py <trace.json | dir with out/*merged*.trace.json> [--sample 1]
"""
import collections, glob, json, os, re, sys
p = sys.argv[1]; sample = sys.argv[sys.argv.index("--sample") + 1] if "--sample" in sys.argv else "1"
if os.path.isdir(p):
    c = sorted(glob.glob(os.path.join(p, "out", f"*inf-{sample}-*merged*.trace.json"))) or sorted(glob.glob(os.path.join(p, "out", "*merged*.trace.json")))
    p = c[0]
ev = json.load(open(p))["traceEvents"]
names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
SL = re.compile(r"QAicGraph_(Partition\d+)_slice(\d+)_Core_(\d+)(?:_(\w+))?$")
d = collections.defaultdict(lambda: dict(es=1e18, ee=0, g0=1e18, g1=0, ncore=set()))
t0 = 1e18
for e in ev:
    if e.get("ph") != "X": continue
    m = SL.match(names.get((e["pid"], e["tid"]), "")); 
    if not m: continue
    part, sl, core, eng = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    k = (part, sl); r = d[k]
    if eng is None:
        r["es"] = min(r["es"], e["ts"]); r["ee"] = max(r["ee"], e["ts"] + e["dur"]); r["ncore"].add(core); t0 = min(t0, e["ts"])
    elif e["name"].startswith("aicconvolutiond32"):
        r["g0"] = min(r["g0"], e["ts"]); r["g1"] = max(r["g1"], e["ts"] + e["dur"])
print(f"{os.path.basename(p)[:60]}   (ms, relative to first core start)")
print(f"{'partition':12s} {'slice':>5s} {'exec start':>10s} {'exec end':>9s} {'first GEMM':>10s} {'last GEMM':>9s} cores")
for (part, sl), r in sorted(d.items()):
    print(f"{part:12s} {sl:5d} {(r['es']-t0)/1e3:10.2f} {(r['ee']-t0)/1e3:9.2f} {(r['g0']-t0)/1e3:10.2f} {(r['g1']-t0)/1e3:9.2f} {len(r['ncore'])}")
print(f"data ready (last exec end): {(max(r['ee'] for r in d.values())-t0)/1e3:.2f} ms")
