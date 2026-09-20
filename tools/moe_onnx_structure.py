#!/usr/bin/env python3
"""Phase 1: dump the MoE structure of an exported ONNX graph.

Answers, from the graph alone (no weights loaded):
  * how many nodes total, and how many per decoder layer
  * what the router / expert-gather / expert-matmul nodes are actually named
  * whether experts appear as 128 separate subgraphs (per-expert nodes) or as
    one gathered 3D tensor (index-driven), which decides whether per-expert
    attribution is even possible downstream
  * initializer shapes for the expert weights

Usage:  python3 moe_onnx_structure.py <model.onnx> [--layer 0]
"""
import sys, re, json, collections

def main():
    path = sys.argv[1]
    want_layer = 0
    if "--layer" in sys.argv:
        want_layer = int(sys.argv[sys.argv.index("--layer") + 1])

    import onnx
    # metadata only: do NOT pull in the ~60 GB of external weights
    m = onnx.load(path, load_external_data=False)
    g = m.graph

    print(f"=== {path}")
    print(f"nodes        : {len(g.node)}")
    print(f"initializers : {len(g.initializer)}")
    print(f"inputs       : {len(g.input)}   outputs: {len(g.output)}")
    print(f"opset        : {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")

    # ---- node-name shape, layer-normalised ----
    pat = collections.Counter()
    kinds = collections.Counter()
    for n in g.node:
        kinds[n.op_type] += 1
        nm = re.sub(r"layers\.\d+", "layers.N", n.name or "")
        nm = re.sub(r"/n\d+$", "/nK", nm)
        nm = re.sub(r"_\d+$", "_K", nm)
        pat[nm] += 1

    print("\n=== op_type histogram (top 25) ===")
    for k, v in kinds.most_common(25):
        print(f"  {v:8d}  {k}")

    # ---- everything inside one layer's mlp ----
    lay = f"/model/layers.{want_layer}/"
    mlp_nodes = [n for n in g.node if n.name and lay in n.name and "/mlp" in n.name]
    print(f"\n=== layer {want_layer}: nodes under /mlp  ({len(mlp_nodes)}) ===")
    for n in mlp_nodes:
        ins = ", ".join(i.split("/")[-1][:38] for i in n.input[:3])
        print(f"  {n.op_type:16s} {n.name}")
        print(f"  {'':16s}   in: {ins}")

    # ---- per-layer node counts, to see if layers are uniform ----
    per_layer = collections.Counter()
    for n in g.node:
        mm = re.search(r"/layers\.(\d+)/", n.name or "")
        if mm:
            per_layer[int(mm.group(1))] += 1
    if per_layer:
        vals = sorted(per_layer.items())
        print(f"\n=== nodes per layer ===")
        print(f"  layers seen : {len(per_layer)}")
        print(f"  min/max     : {min(per_layer.values())} / {max(per_layer.values())}")
        print(f"  first 5     : {vals[:5]}")

    # ---- expert-weight initializers ----
    print("\n=== initializers matching expert / gate (layer %d) ===" % want_layer)
    shown = 0
    for init in g.initializer:
        if lay in init.name or (f"layers.{want_layer}." in init.name):
            if any(k in init.name for k in ("expert", "gate", "up_proj", "down_proj")):
                dims = list(init.dims)
                print(f"  {init.name:70s} dims={dims}")
                shown += 1
                if shown > 12:
                    print("  ...")
                    break
    if not shown:
        print("  (none matched — printing 12 largest initializers instead)")
        big = sorted(g.initializer, key=lambda i: -int(__import__('math').prod(i.dims or [0])))[:12]
        for init in big:
            print(f"  {init.name:70s} dims={list(init.dims)}")

    # ---- verdict on per-expert attribution ----
    per_expert_named = [n for n in g.node if n.name and re.search(r"experts\.\d+", n.name)]
    print("\n=== VERDICT: is per-expert attribution possible from node names? ===")
    if per_expert_named:
        idxs = sorted({int(re.search(r"experts\.(\d+)", n.name).group(1)) for n in per_expert_named})
        print(f"  YES — {len(per_expert_named)} nodes carry an explicit expert index")
        print(f"  expert indices seen: {idxs[:8]} ... {idxs[-3:]}  (n={len(idxs)})")
    else:
        print("  NO — no node name carries an expert index.")
        print("  Experts are addressed by a runtime Gather into one 3D weight tensor,")
        print("  so the graph has ONE set of matmul nodes per layer regardless of which")
        print("  expert runs. Per-expert cost cannot come from the trace; it must come")
        print("  from the router indices (Phase 2) combined with the placement rule.")

if __name__ == "__main__":
    main()
