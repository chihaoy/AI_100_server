#!/usr/bin/env python3
# Extract the cross-CARD data flow (AllGather/AllReduce-style combine) from a merged 4-card
# qaic-opstats trace.json. Reads the aicmulticastvtcm_out/_in events, whose `PortDescription`
# gives the src->dst card, `opName` the layer/op, `opOutputSize` the bytes, `opUCycle` the cost.
#
#   python3 perop_crosscard_flow.py <merged trace.json>
import re, sys, collections

T = sys.argv[1]
name_re = re.compile(r'"name": "aicmulticastvtcm_(out|in)"')
port_re = re.compile(r'"PortDescription" : "([^"]*)"')
op_re   = re.compile(r'"opName" : "([^"]*)"')
sz_re   = re.compile(r'"opOutputSize" : "(\d+)"')
uc_re   = re.compile(r'"opUCycle" : "(\d+)"')

edges = collections.Counter()        # "src->dst" -> count
edge_bytes = collections.Counter()   # "src->dst" -> total bytes
edge_uc = collections.Counter()      # "src->dst" -> total ucycles
by_layer = collections.Counter()     # layer index -> transfer count (out only)
by_optype = collections.Counter()    # op site (proj/attn part) -> count
tot_out = tot_in = 0

def layer_of(op):
    m = re.search(r'/layers\.(\d+)/', op)
    return int(m.group(1)) if m else -1
def site_of(op):
    m = re.search(r'/layers\.\d+/([a-z_]+/[a-z_0-9]+)', op)
    return m.group(1) if m else op.split('/')[-2] if '/' in op else op

with open(T, "r", errors="ignore") as f:
    for line in f:
        m = name_re.search(line)
        if not m: continue
        direction = m.group(1)                 # out | in
        port = port_re.search(line); op = op_re.search(line)
        sz = sz_re.search(line); uc = uc_re.search(line)
        port = port.group(1) if port else "?"
        nbytes = int(sz.group(1)) if sz else 0
        nuc = int(uc.group(1)) if uc else 0
        if direction == "out":                 # "src->dst[,dst...]"
            tot_out += 1
            mm = re.match(r'(\d+)->(.+)', port)
            if mm:
                src = mm.group(1)
                for dst in mm.group(2).split(','):
                    e = f"card{src} -> card{dst.strip()}"
                    edges[e] += 1; edge_bytes[e] += nbytes; edge_uc[e] += nuc
            if op:
                by_layer[layer_of(op.group(1))] += 1
                by_optype[site_of(op.group(1))] += 1
        else:
            tot_in += 1

print(f"# trace: {T.split('/')[-1]}")
print(f"# cross-card multicast events:  out(send)={tot_out:,}  in(recv)={tot_in:,}\n")

print("=== card -> card transfers (from multicast_out PortDescription) ===")
print(f"{'edge':<20}{'count':>8}{'total_bytes':>14}{'total_ucycles':>15}")
for e in sorted(edges):
    print(f"{e:<20}{edges[e]:>8}{edge_bytes[e]:>14,}{edge_uc[e]:>15,}")

print("\n=== which op-sites trigger a cross-card send (top) ===")
for site, c in by_optype.most_common(12):
    print(f"  {c:>5}  {site}")

print("\n=== transfers per layer (out events) ===")
ls = sorted(k for k in by_layer if k >= 0)
if ls:
    per = by_layer[ls[0]]
    same = all(by_layer[k] == per for k in ls)
    print(f"  layers {ls[0]}..{ls[-1]}  ({len(ls)} layers), "
          + (f"{per} sends each (uniform)" if same else "varies: " + str(dict(by_layer))))
