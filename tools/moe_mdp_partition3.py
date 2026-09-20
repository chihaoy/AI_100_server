#!/usr/bin/env python3
"""Partition config for a --dup-router tier-bench graph (path 1 of naive_vs_64_16): classify every node by which lane
group's weights it (transitively) depends on, using the ONNX initializer names (router / Wg.0 / Wu.0 / Wd.0 -> group 0,
router_dup / Wg.1 / ... -> group 1).

  group-0-only nodes -> P0 (devices --p0);  group-1-only nodes -> P1 (--p1)
  nodes depending on both (the combine Add, final reduce, output) -> P2 (--p2 devices), or merged into P0/P1
  (--merge-into 0|1), or must not exist (--separate-outputs graphs)
  nodes depending on neither (shared prologue, constants) -> the partition of their consumers if unanimous, else P0

Prints every edge that crosses from P0 to P1 or back (there should be none).
  python3 moe_mdp_partition3.py <variant_dir> --p0 0,1 --p1 2,3 [--p2 2,3 | --merge-into 1] [--out mdp3.json]
"""
import argparse, collections, json, os, re
import onnx

SUFFIX = {"CtxScatter3DInt": ["n15", "n17", "n18"], "CtxScatter3D": ["n17", "n18"], "CtxGather3D": ["n14"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant_dir")
    ap.add_argument("--p0", default="0,1"); ap.add_argument("--p1", default="2,3")
    ap.add_argument("--p2", default=None, help="devices for the combine partition, e.g. 2,3")
    ap.add_argument("--merge-into", type=int, default=None, help="put combine nodes into P0 or P1 instead of a third partition")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = onnx.load(os.path.join(a.variant_dir, "moe_layer.onnx"), load_external_data=False).graph
    nodes = list(g.node); byname = {n.name: n for n in nodes}; prod = {o: n.name for n in nodes for o in n.output}
    # classify the weight initializers by the MatMul that consumes them, in node order: the first S_R MatMuls that
    # read the graph input x with a [H, E] weight are the routers (router -> group 0, router_dup -> group 1); the
    # expert GEMMs (weights [n,H,I] / [n,I,H]) come next, 3*S per group.  (Initializer names are lost to constant
    # folding, so names cannot be used.)
    init_group = {}
    ishape = {i.name: list(i.dims) for i in g.initializer}
    routers, experts = [], []
    gin = {i.name for i in g.input}
    for n in nodes:
        if n.op_type != "MatMul": continue
        if n.input[0] in gin: routers.append((n.name, n.input[1]))          # router: x @ W  (W may be an Identity alias)
        else: experts.append((n.name, n.input[1]))
    assert len(routers) == 2, routers
    node_group = {routers[0][0]: 0, routers[1][0]: 1}
    # the Identity alias feeding router_dup's MatMul belongs to group 1 too
    for n in nodes:
        if n.op_type == "Identity" and n.output[0] == routers[1][1]: node_group[n.name] = 1
    half = len(experts) // 2
    for k, (nm, w) in enumerate(experts):
        node_group[nm] = 0 if k < half else 1
    print(f"routers {routers}  expert matmuls: {[e[0] for e in experts]}  (first {half} -> group 0)")
    dep = {}
    def deps(nm):
        if nm in dep: return dep[nm]
        s = set()
        if nm in node_group: s.add(node_group[nm])
        for i in byname[nm].input:
            if i in init_group: s.add(init_group[i])
            elif i in prod: s |= deps(prod[i])
        dep[nm] = s; return s
    consumers = collections.defaultdict(list)
    for n in nodes:
        for i in n.input:
            if i in prod: consumers[prod[i]].append(n.name)
    part = {}
    keep = []
    for n in nodes:
        if n.op_type == "Constant":
            feeds = {byname[c].op_type for c in consumers[n.name]}
            if "ScatterElements" not in feeds: continue     # only the densify constants survive constant folding
        keep.append(n.name)
        d = deps(n.name)
        part[n.name] = {frozenset(): None, frozenset({0}): 0, frozenset({1}): 1}.get(frozenset(d), 2)
    # neutral nodes: follow consumers (iterate to fixpoint from the outputs backwards)
    for _ in range(10):
        changed = False
        for nm in keep:
            if part[nm] is None:
                cs = {part[c] for c in consumers[nm] if c in part and part[c] is not None}
                if len(cs) == 1: part[nm] = cs.pop(); changed = True
        if not changed: break
    neutral = [nm for nm in keep if part[nm] is None]
    for nm in neutral: part[nm] = 0
    if a.merge_into is not None:
        for nm in keep:
            if part[nm] == 2: part[nm] = a.merge_into
    lists = {0: [], 1: [], 2: []}
    for nm in keep:
        n = byname[nm]
        names = [f"{nm}/{s}" for s in SUFFIX[n.op_type]] if n.op_type in SUFFIX else [nm]
        lists[part[nm]].extend(names)
    cross = [(prod[i], nm) for nm in keep for i in byname[nm].input if i in prod and prod[i] in part and {part[prod[i]], part[nm]} == {0, 1}]
    print(f"P0 {len(lists[0])} nodes, P1 {len(lists[1])} nodes, P2 {len(lists[2])} nodes; neutral->P0: {neutral}")
    print(f"edges between P0 and P1: {len(cross)} {cross[:10]}")
    print("P2 nodes:", lists[2])
    dev = lambda s: [{"deviceId": int(d), "numCores": a.cores} for d in s.split(",")]
    alld = sorted({int(d) for s in (a.p0, a.p1, a.p2 or "") for d in s.split(",") if d})
    parts = [{"name": "Partition0", "devices": dev(a.p0), "nodeList": lists[0]},
             {"name": "Partition1", "devices": dev(a.p1), "nodeList": lists[1]}]
    if lists[2]:
        assert a.p2, "combine nodes exist: give --p2 devices or --merge-into"
        parts.append({"name": "Partition2", "devices": dev(a.p2), "nodeList": lists[2]})
    cfg = {"connections": [{"devices": alld, "type": "p2p"}], "partitions": parts}
    out = a.out or os.path.join(a.variant_dir, "mdp3.json")
    json.dump(cfg, open(out, "w"), indent=1); print("wrote", out)


if __name__ == "__main__":
    main()
