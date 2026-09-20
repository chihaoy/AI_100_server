#!/usr/bin/env python3
"""Report what an MDP cut costs: which tensors cross card boundaries, and how many bytes.

Stage assignment is read from the partition config (compiler node namespace); tensor shapes come
from the ONNX. Only edges whose producer and consumer land on different cards cost PCIe traffic.

  python3 perop_cut_cost.py --mdp mdp_pp4_bal.json --onnx model.onnx --seq 128
"""
import argparse, collections, json, re, sys

LAYER_RE = re.compile(r'^/model/layers\.(\d+)/')
DTYPE_BYTES = {1: 4, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 16: 2}   # float,int32,int64,bool,fp16,double,bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdp", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seq", type=int, default=128, help="sequence length to size activations for")
    ap.add_argument("--batch", type=int, default=1, help="batch size for symbolic batch dims")
    ap.add_argument("--ctx", type=int, default=256, help="ctx_len for symbolic ctx dims")
    ap.add_argument("--bytes", type=int, default=2,
                    help="bytes per element on device (default 2: the ONNX says fp32, but the "
                         "QPC is built with -convert-to-fp16, so the wire format is fp16)")
    a = ap.parse_args()
    import onnx, onnx.shape_inference

    cfg = json.load(open(a.mdp))
    if "nodeList" not in cfg["partitions"][0]:
        sys.exit("this config has no nodeList (tensor-slice mode) -- nothing to analyse")

    # compiler node -> card.  Strip the /nNN sub-node suffix so names match ONNX.
    card, ncards = {}, len(cfg["partitions"])
    for s, p in enumerate(cfg["partitions"]):
        for n in p["nodeList"]:
            base, _, suf = n.rpartition("/")
            card[base if re.fullmatch(r"n\d+", suf) else n] = s

    m = onnx.load(a.onnx, load_external_data=False)
    # the export carries no value_info for intermediates; infer it (initializer dims are in the
    # proto even though the raw weights live in the external .data file)
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as e:
        print(f"(shape inference failed: {e})", file=sys.stderr)
    g = m.graph
    dims = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        t = vi.type.tensor_type
        # Symbolic dims. Named ones (batch_size/seq_len/ctx_len) resolve directly; shape
        # inference loses the names through Reshape and emits unk__NNNN, so those fall back to
        # position: for a transformer activation the first symbolic dim is batch, the rest seq.
        shape, sym_seen = [], 0
        for d in t.shape.dim:
            if d.dim_value:
                shape.append(d.dim_value); continue
            nm = d.dim_param.lower()
            if "batch" in nm:      shape.append(a.batch)
            elif "ctx" in nm:      shape.append(a.ctx)
            elif "seq" in nm:      shape.append(a.seq)
            else:                  shape.append(a.batch if sym_seen == 0 else a.seq)
            sym_seen += 1
        if shape:
            dims[vi.name] = (shape, t.elem_type)

    # constants and shape ops were folded by the compiler; inherit their card from the consumer
    prod = {o: n.name for n in g.node for o in n.output}
    init = {i.name for i in g.initializer} | {i.name for i in g.input}

    cross = collections.defaultdict(list)
    for n in g.node:
        cs = card.get(n.name)
        if cs is None:
            continue                                   # folded away -- no runtime cost
        for t in n.input:
            if t in init or t not in prod:
                continue
            ps = card.get(prod[t])
            if ps is not None and ps != cs:
                cross[(ps, cs)].append(t)

    print(f"cut: {ncards} cards, seq_len={a.seq}")
    grand = 0
    for (x, y), ts in sorted(cross.items()):
        uniq = sorted(set(ts))
        tot = 0
        print(f"\ncard{x} -> card{y}:  {len(uniq)} tensor(s)")
        for t in uniq:
            shape, et = dims.get(t, (None, None))
            if shape:
                nbytes = a.bytes
                for d in shape:
                    nbytes *= d
                tot += nbytes
                print(f"    {t:<52} {shape}  {nbytes/1024:>10,.1f} KB")
            else:
                print(f"    {t:<52} (shape unknown)")
        grand += tot
        print(f"    subtotal {tot/1024:,.1f} KB")
    print(f"\ntotal cross-card traffic: {grand/1024:,.1f} KB = {grand/1024/1024:.2f} MB per inference")


if __name__ == "__main__":
    main()
