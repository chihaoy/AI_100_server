#!/usr/bin/env python3
"""Re-cut an MDP partition config (the graph cut) for qaic-compile.

Input is the config the COMPILER dumps, not the raw ONNX -- the compiler folds constants and
renames custom ops (/CtxScatter -> /CtxScatter/n21), so raw ONNX node names are rejected:

  1) dump the compiler's own cut (lands ~15 s in; kill the compile once the file appears)
       qaic-compile ... -mdp-dump-partition-config=dump.json
  2) re-cut it
       perop_make_mdp.py --dump dump.json --mode pipeline --stages 4 -o mdp_pp4.json
  3) compile with it
       qaic-compile ... -mdp-load-partition-config=mdp_pp4.json

Modes:
  tensor    1 partition x N devices  -- compiler tensor-slices every node across all cards
                                       (today's mdp_ts_4.json; all-to-all AllReduce per layer)
  pipeline  N partitions x 1 device  -- layer ranges pinned per card, no intra-layer cross-card comm
  hybrid    P partitions x T devices -- P pipeline stages, T-way tensor slice inside each

--conn p2p  = card-to-card fabric        --conn host = route the exchange through host DRAM
"""
import argparse, collections, json, re, sys

LAYER_RE = re.compile(r'^/model/layers\.(\d+)/')


def load_dump(path):
    """-> [node names] in the compiler's namespace (unordered across partitions)."""
    d = json.load(open(path))
    names = [n for p in d["partitions"] for n in p.get("nodeList", [])]
    if not names:
        sys.exit(f"{path} has no nodeList -- dump it from a run that actually partitioned "
                 f"(the compiler emits nodeList only when it splits across >1 device)")
    return names


def topo_sort(names, onnx_path):
    """Order compiler node names by the ONNX graph's topological order.

    The compiler validates that a producer precedes its consumers *within* a partition, and the
    dump concatenates partitions, so its raw order is not globally topological. ONNX stores nodes
    topologically, so we key off that. Custom ops fan out to sub-nodes (ONNX
    /…/CtxScatter -> compiler /…/CtxScatter/n21), which sort after their base node by suffix.
    """
    import onnx
    g = onnx.load(onnx_path, load_external_data=False).graph
    idx = {n.name: i for i, n in enumerate(g.node)}
    unknown = []

    def key(c):
        if c in idx:
            return (idx[c], 0)
        base, _, suf = c.rpartition("/")
        if base in idx and re.fullmatch(r"n\d+", suf):
            return (idx[base], int(suf[1:]))
        unknown.append(c)
        return (len(idx), 0)

    out = sorted(names, key=key)
    if unknown:
        print(f"  warn: {len(unknown)} compiler nodes not found in ONNX, placed last "
              f"(e.g. {unknown[:3]})", file=sys.stderr)
    return out


def cut(names, nstages, boundaries=None):
    """-> {name: stage}. Layer nodes by layer index; pre/post nodes to first/last stage.

    Equal layer counts do NOT give equal stage times: stage 0 also carries the embedding and
    stage N-1 carries lm_head (hidden x vocab, the single largest MatMul in the model). Pipeline
    throughput is 1/max(stage_time), so the tail stage must get fewer layers. Pass explicit
    --boundaries once you have measured per-stage times.
    """
    layers = sorted({int(m.group(1)) for n in names if (m := LAYER_RE.match(n))})
    if not layers:
        sys.exit("no /model/layers.<i>/ nodes -- is this a decoder model?")
    if boundaries:
        cuts = [int(x) for x in boundaries.split(",")]
        if len(cuts) != nstages - 1 or cuts != sorted(cuts) or not all(0 < c < len(layers) for c in cuts):
            sys.exit(f"--boundaries needs {nstages-1} strictly increasing layer indices in 1..{len(layers)-1}")
        b = [0] + cuts + [len(layers)]
    else:
        b = [(len(layers) * s) // nstages for s in range(nstages + 1)]
    l2s = {li: s for s in range(nstages) for li in layers[b[s]:b[s + 1]]}

    stage, first_layer_seen = {}, False
    for n in names:
        m = LAYER_RE.match(n)
        if m:
            stage[n], first_layer_seen = l2s[int(m.group(1))], True
        else:                       # embed/rope prologue -> stage 0; norm/lm_head tail -> last
            stage[n] = nstages - 1 if first_layer_seen else 0
    return stage, layers, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="config from -mdp-dump-partition-config")
    ap.add_argument("--onnx", required=True,
                    help="the same ONNX, for topological order (weights are not loaded)")
    ap.add_argument("--mode", choices=["tensor", "pipeline", "hybrid"], default="pipeline")
    ap.add_argument("--stages", type=int, default=4, help="pipeline stages (partitions)")
    ap.add_argument("--tp", type=int, default=1, help="devices (tensor slices) per stage")
    ap.add_argument("--cores", type=int, default=16, help="cores per device")
    ap.add_argument("--conn", choices=["p2p", "host"], default="p2p")
    ap.add_argument("--devices", default="0,1,2,3")
    ap.add_argument("--boundaries", default=None,
                    help="explicit first-layer-of-each-later-stage, e.g. 10,20,29 for 4 stages "
                         "(default: equal layer counts, which under-loads the lm_head stage)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    devices = [int(x) for x in a.devices.split(",")]
    stages = 1 if a.mode == "tensor" else a.stages
    tp = len(devices) if a.mode == "tensor" else (1 if a.mode == "pipeline" else a.tp)
    if stages * tp > len(devices):
        sys.exit(f"need {stages} stages x {tp} tp = {stages*tp} devices, got {len(devices)}")

    names = topo_sort(load_dump(a.dump), a.onnx)
    conns = [{"devices": devices, "type": a.conn}]

    if a.mode == "tensor":
        cfg = {"connections": conns, "partitions": [
            {"name": "Partition0",
             "devices": [{"deviceId": d, "numCores": a.cores} for d in devices]}]}
        print(f"mode=tensor tp={tp} conn={a.conn} cores/dev={a.cores}  ({len(names)} nodes, "
              f"compiler slices all of them across cards {devices})")
    else:
        stage, layers, b = cut(names, stages, a.boundaries)
        buckets = collections.defaultdict(list)
        for n in names:
            buckets[stage[n]].append(n)
        parts = [{"name": f"Partition{s}", "nodeList": buckets[s],
                  "devices": [{"deviceId": d, "numCores": a.cores}
                              for d in devices[s * tp:(s + 1) * tp]]} for s in range(stages)]
        cfg = {"connections": conns, "partitions": parts}

        placed = [n for p in parts for n in p["nodeList"]]
        assert len(placed) == len(names) and len(set(placed)) == len(names), "assignment not 1:1"
        print(f"mode={a.mode} stages={stages} tp={tp} conn={a.conn} cores/dev={a.cores}")
        print(f"compiler nodes: {len(names)}")
        for s, p in enumerate(parts):
            devs = ",".join(str(d["deviceId"]) for d in p["devices"])
            print(f"  {p['name']}: layers {layers[b[s]]:>2}..{layers[b[s+1]-1]:>2} "
                  f"({b[s+1]-b[s]:>2} layers, {len(p['nodeList']):>4} nodes) -> card(s) {devs}")

    json.dump(cfg, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
