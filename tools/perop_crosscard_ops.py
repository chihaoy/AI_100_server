#!/usr/bin/env python3
# Exact per-OP cross-card flow: for every aicmulticastvtcm_out event, which OP sent from which
# card to which card, how many bytes / cycles. Reads a merged 4-card qaic-opstats trace.json.
#
#   python3 perop_crosscard_ops.py <merged trace.json> [layer]
#     no layer  -> aggregate (op-site x src->dst) + write full per-transfer CSV
#     layer N   -> also print every exact transfer in that layer
import re, sys, collections

T = sys.argv[1]
want_layer = int(sys.argv[2]) if len(sys.argv) > 2 else None

name_re = re.compile(r'"name": "aicmulticastvtcm_out"')
port_re = re.compile(r'"PortDescription" : "([^"]*)"')
op_re   = re.compile(r'"opName" : "([^"]*)"')
sz_re   = re.compile(r'"opOutputSize" : "(\d+)"')
uc_re   = re.compile(r'"opUCycle" : "(\d+)"')

def layer_of(op):
    m = re.search(r'/layers\.(\d+)/', op); return int(m.group(1)) if m else -1
def clean(op):                                   # strip libjit prefix + trailing /__NNNN
    op = op.split()[-1]
    return re.sub(r'/__\d+$', '', op)

agg = collections.OrderedDict()                  # (site, src->dst) -> [count, bytes, ucycles]
rows = []                                        # full per-transfer list
with open(T, "r", errors="ignore") as f:
    for line in f:
        if not name_re.search(line): continue
        port = port_re.search(line); op = op_re.search(line)
        sz = sz_re.search(line); uc = uc_re.search(line)
        if not (port and op): continue
        opn = clean(op.group(1)); lyr = layer_of(op.group(1))
        nbytes = int(sz.group(1)) if sz else 0; nuc = int(uc.group(1)) if uc else 0
        mm = re.match(r'(\d+)->(.+)', port.group(1))
        if not mm: continue
        src = mm.group(1)
        for dst in mm.group(2).split(','):
            edge = f"{src}->{dst.strip()}"
            rows.append((lyr, opn, edge, nbytes, nuc))
            site = re.sub(r'/layers\.\d+/', '/layers.N/', opn)
            k = (site, edge)
            a = agg.setdefault(k, [0, 0, 0]); a[0]+=1; a[1]+=nbytes; a[2]+=nuc

print(f"# trace: {T.split('/')[-1]}   total cross-card sends: {len(rows):,}\n")

# aggregate: for each op-site, the src->dst breakdown
print("=== aggregate: op-site  →  card src->dst  (count / bytes / ucycles) ===")
sites = collections.OrderedDict()
for (site, edge), v in agg.items():
    sites.setdefault(site, []).append((edge, v))
for site, lst in sites.items():
    print(f"\n{site}")
    for edge, v in sorted(lst):
        print(f"    {edge:<8} count={v[0]:>4}  bytes={v[1]:>12,}  ucycles={v[2]:>10,}")

# exact per-transfer listing for one layer
if want_layer is not None:
    print(f"\n=== EXACT per-op transfers in layer {want_layer} ===")
    print(f"{'op':<48}{'src->dst':<10}{'bytes':>8}{'ucyc':>8}")
    for lyr, opn, edge, nb, nuc in rows:
        if lyr == want_layer:
            print(f"{opn.replace('/model/layers.'+str(lyr)+'/',''):<48}{edge:<10}{nb:>8}{nuc:>8}")

# full CSV
out = T.rsplit('/',1)[0] + "/crosscard_ops.csv"
with open(out, "w") as f:
    f.write("layer,op,src_card,dst_card,bytes,ucycles\n")
    for lyr, opn, edge, nb, nuc in rows:
        s,d = edge.split('->'); f.write(f"{lyr},{opn},{s},{d},{nb},{nuc}\n")
print(f"\n# full per-transfer CSV ({len(rows)} rows): {out}")
