#!/usr/bin/env python3
"""Write a manual MDP partition config that puts one lane group of a tier-bench MoE graph on one set of cards
and the rest on another set -- e.g. the 128-row group on cards 0,1 and the 64-row group on cards 2,3.

The compiler accepts several partitions in the -mdp-load-partition-config json, each with its own devices and an
explicit, topologically ordered `nodeList` ("The manual partitioner requires a node lists"; consumers must come
after producers, and partitions execute in list order).  Node names are the ONNX node names, with QEfficient custom
ops expanded into the compiler's sub-nodes (CtxScatter3DInt -> n15,n17,n18; CtxScatter3D -> n17,n18; CtxGather3D -> n14).

Rule used here: every node that (transitively) depends on the second group's MatMuls -> partition 1 (including the
final combine); everything else (router, index chains, first group) -> partition 0.

  python3 moe_mdp_manual_partition.py <variant_dir> --group1-first-matmul 7 --p0 0,1 --p1 2,3 [--out mdp2.json]
"""
import argparse, json, os, re
import onnx

SUFFIX = {"CtxScatter3DInt": ["n15", "n17", "n18"], "CtxScatter3D": ["n17", "n18"], "CtxGather3D": ["n14"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--group1-first-matmul", type=int, required=True, help="index k: /MatMul_k and above belong to group 1")
    ap.add_argument("--p0", default="0,1"); ap.add_argument("--p1", default="2,3")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = onnx.load(os.path.join(a.variant_dir, "moe_layer.onnx"), load_external_data=False).graph
    nodes = list(g.node); byname = {n.name: n for n in nodes}; prod = {o: n.name for n in nodes for o in n.output}

    def mm_group(name):
        m = re.match(r"/MatMul_(\d+)$", name)
        return None if not m else (1 if int(m.group(1)) >= a.group1_first_matmul else 0)

    anc = {}
    def ancestors(nm):
        if nm in anc:
            return anc[nm]
        s = set()
        for i in byname[nm].input:
            if i in prod:
                p = prod[i]; s.add(p); s |= ancestors(p)
        anc[nm] = s; return s

    p0, p1 = [], []
    for n in nodes:
        names = [f"{n.name}/{s}" for s in SUFFIX[n.op_type]] if n.op_type in SUFFIX else [n.name]
        if n.op_type == "Constant":
            feeds = {x.op_type for x in nodes if n.output and n.output[0] in x.input}
            if "ScatterElements" not in feeds:      # only the densify constant survives constant folding
                continue
        dep1 = mm_group(n.name) == 1 or any(mm_group(x) == 1 for x in ancestors(n.name))
        (p1 if dep1 else p0).extend(names)
    dev = lambda s: [{"deviceId": int(d), "numCores": a.cores} for d in s.split(",")]
    alld = sorted({int(d) for d in (a.p0 + "," + a.p1).split(",")})
    cfg = {"connections": [{"devices": alld, "type": "p2p"}],
           "partitions": [{"name": "Partition0", "devices": dev(a.p0), "nodeList": p0},
                          {"name": "Partition1", "devices": dev(a.p1), "nodeList": p1}]}
    out = a.out or os.path.join(a.variant_dir, "mdp2.json")
    json.dump(cfg, open(out, "w"), indent=1)
    print(f"wrote {out}: partition0 {len(p0)} nodes on devices {a.p0}, partition1 {len(p1)} nodes on devices {a.p1}")


if __name__ == "__main__":
    main()
