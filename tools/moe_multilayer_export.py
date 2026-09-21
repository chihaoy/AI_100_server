#!/usr/bin/env python3
"""Export consecutive real decoder layers into one ONNX network, with per-layer capacities.

The uniform and tuned graphs use identical weights, expert orders and inputs.
Each layer also returns its routing counts, so overflow can be checked on hardware.
Layer ONNX files use external weights; stitching never loads all model weights.
See 9.17.2026/multilayer/README.md for reproduction and limitations.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from safetensors import safe_open
from transformers import AutoConfig

from moe_e2e_layer_export import E2ELayer
from moe_tier_bench_export import MoELayerEP
from QEfficient.base.onnx_transforms import CustomOpTransform, FP16ClipTransform


class ObservedMoE(MoELayerEP):
    def forward(self, x):
        self.counts = (self.route(x) > 0).sum(0).to(torch.int32)
        return super().forward(x)


class ObservedLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        y = self.layer(x)
        return y, self.layer.moe.counts


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def metrics(actual, expected):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Output shape mismatch or nonfinite output")
    err = a - b
    return dict(max_abs=float(np.abs(err).max()),
                relative_l2=float(np.linalg.norm(err) / max(np.linalg.norm(b), 1e-12)),
                max_token_relative_l2=float(np.max(np.linalg.norm(err, axis=-1) /
                                                   np.maximum(np.linalg.norm(b, axis=-1), 1e-8))),
                exact=bool(np.array_equal(a, b)))


def save_layer(model, x, directory):
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (x,), str(path), input_names=["x"], output_names=["y", "counts"],
                          dynamo=False, opset_version=17, do_constant_folding=True)
    graph = onnx.load(path)
    CustomOpTransform.apply(graph)
    for init in graph.graph.initializer:
        FP16ClipTransform.apply(init, str(directory), 65504.0, -65504.0)
    onnx.save_model(graph, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=1024)
    # Capture inferred GEMM shapes without materializing external weight payloads.
    del graph
    graph = onnx.load(path, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(graph)
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else d.dim_param
                      for d in v.type.tensor_type.shape.dim]
              for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
    audit = [dict(node=n.name, inputs=[shapes.get(i) for i in n.input],
                  outputs=[shapes.get(i) for i in n.output])
             for n in inferred.graph.node if n.op_type == "MatMul" and "/moe/" in n.name]
    write_json(directory / "matmul_shapes.json", audit)


def stitch(root, layers):
    """Join ONNX graphs before compilation. There are no runtime layer boundaries."""
    combined = None
    previous = "x"
    counts = []
    opsets, functions = {}, {}
    for layer in layers:
        subdir = root / f"L{layer}"
        model = onnx.load(subdir / "model.onnx", load_external_data=False)
        for init in model.graph.initializer:
            for field in init.external_data:
                if field.key == "location":
                    field.value = os.path.relpath(subdir / field.value, root)
        model = onnx.compose.add_prefix(model, f"L{layer}_", rename_functions=False)
        mapping = {f"L{layer}_x": previous}
        if layer == layers[-1]:
            mapping[f"L{layer}_y"] = "y"
        for node in model.graph.node:
            for field in (node.input, node.output):
                for i, name in enumerate(field):
                    field[i] = mapping.get(name, name)
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
            value.name = mapping.get(value.name, value.name)
        counts.append(copy.deepcopy(model.graph.output[1]))
        for op in model.opset_import:
            if op.domain in opsets and opsets[op.domain] != op.version:
                raise ValueError("Mismatched opsets when stitching layers")
            opsets[op.domain] = op.version
        for fn in model.functions:
            key = (fn.domain, fn.name)
            if key in functions and functions[key].SerializeToString() != fn.SerializeToString():
                raise ValueError(f"Mismatched function {key}")
            functions[key] = fn
        if combined is None:
            combined = model
        else:
            combined.graph.node.extend(model.graph.node)
            combined.graph.initializer.extend(model.graph.initializer)
            combined.graph.value_info.extend(model.graph.value_info)
        previous = model.graph.output[0].name
        final_output = copy.deepcopy(model.graph.output[0])
    del combined.graph.output[:]
    combined.graph.output.extend([final_output, *counts])
    del combined.functions[:]
    combined.functions.extend(functions.values())
    path = root / "model.onnx"
    onnx.save_model(combined, path)
    onnx.checker.check_model(str(path))
    # Timing graph prunes only diagnostic outputs; it uses exactly the same weights.
    del combined.graph.output[1:]
    onnx.save_model(combined, root / "timing.onnx")
    print(f"STITCH_OK {path}: {len(layers)} layers, {len(combined.graph.node)} nodes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("/home/chihao/models/qwen3_30b_a3b/hf"))
    ap.add_argument("--e2e", type=Path, default=Path("/home/chihao/mllm/9.17.2026/e2e"))
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--input-npy", type=Path, help="Residual stream entering start-layer, [T,H]")
    ap.add_argument("--plan", type=Path, help='Optional {"layer_id": {"stage_widths": [C0,C1], "order": [...]}}')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = AutoConfig.from_pretrained(a.model, local_files_only=True)
    if a.layers < 1 or a.start_layer < 0 or a.start_layer + a.layers > cfg.num_hidden_layers:
        ap.error("Invalid layer interval")
    if not cfg.norm_topk_prob or cfg.num_experts != 128:
        ap.error("This experiment expects normalized top-k and 128 experts")
    if a.out.exists() and any(a.out.iterdir()):
        ap.error("Use an empty output directory; existing artifacts are never overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    input_path = a.input_npy or a.e2e / "ref" / f"h_L{a.start_layer}.npy"
    x = torch.from_numpy(np.load(input_path)).half()
    if x.ndim != 2 or x.shape[1] != cfg.hidden_size or not torch.isfinite(x).all():
        ap.error("Input must be finite [T, hidden_size]")
    T, H = x.shape
    x.numpy().tofile(a.out / "x.bin")
    plan = json.loads((a.plan or a.e2e / "plan.json").read_text())
    wmap = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"]
    layers = list(range(a.start_layer, a.start_layer + a.layers))
    info = dict(model=str(a.model.resolve()), input=str(input_path.resolve()), T=T, H=H,
                layers=layers, devices=4, precision="fp16 graph; MXFP6 compiler optional",
                reference_scope="one saved prompt, not held-out generalization", plan={}, cpu={})
    hidden = {variant: x for variant in ("uniform", "tuned")}
    for layer in layers:
        spec = plan[str(layer)]
        widths = spec.get("stage_widths") or [max(h, c) for h, c in zip(spec["hot_widths"], spec["cold_widths"])]
        order = spec.get("order") or json.loads((a.e2e / f"order_L{layer}.json").read_text())
        if len(widths) != 2 or any(type(c) is not int or not 1 <= c <= T for c in widths):
            raise ValueError(f"Invalid capacities at layer {layer}: {widths}")
        if sorted(order) != list(range(cfg.num_experts)):
            raise ValueError(f"Invalid expert permutation at layer {layer}")
        info["plan"][str(layer)] = dict(stage_widths=widths, order=order)
        print(f"Layer {layer}: uniform [128,128], tuned {widths}; loading weights", flush=True)
        handles = {}

        def W(key):
            shard = wmap[key]
            if shard not in handles:
                handles[shard] = safe_open(a.model / shard, framework="pt")
            return handles[shard].get_tensor(key).float()

        p = f"model.layers.{layer}."
        real = {"router": W(p + "mlp.gate.weight")[order]}
        for short, proj in (("Wg", "gate_proj"), ("Wu", "up_proj"), ("Wd", "down_proj")):
            real[short] = torch.stack([torch.stack([
                W(p + f"mlp.experts.{order[s * 64 + lane]}.{proj}.weight").T.half()
                for s in range(2)]) for lane in range(64)])
        moe = ObservedMoE(T, H, cfg.moe_intermediate_size, cfg.num_experts_per_tok, 2,
                          [(T, 64)], num_devices=4, stage_widths=[T, T], real=real)
        model = ObservedLayer(E2ELayer(cfg, W, layer, T, moe, True)).eval()
        del real
        handles.clear()
        outputs = {}
        for variant, caps in (("uniform", [T, T]), ("tuned", widths)):
            moe.stage_widths = caps
            with torch.no_grad():
                y, counts = model(hidden[variant])
            cnt = counts.numpy()
            limits = np.repeat(caps, 64)
            overflow = int(np.maximum(cnt - limits, 0).sum())
            if overflow:
                raise RuntimeError(f"CPU overflow at layer {layer} ({variant}): {overflow} assignments")
            if not torch.isfinite(y).all():
                raise RuntimeError(f"Nonfinite CPU output at layer {layer}")
            directory = a.out / variant / f"L{layer}"
            print(f"  {variant}: stage max counts {cnt.reshape(2,64).max(1).tolist()}, exporting", flush=True)
            save_layer(model, hidden[variant], directory)
            np.save(directory / "y_cpu.npy", y.float().numpy())
            np.save(directory / "counts_cpu.npy", cnt)
            outputs[variant] = y.float().numpy()
            hidden[variant] = y
        diff = metrics(outputs["tuned"], outputs["uniform"])
        info["cpu"][str(layer)] = diff
        print(f"  CPU tuned vs uniform: {diff}", flush=True)
        if diff["relative_l2"] > 1e-3:
            raise RuntimeError("CPU capacity change altered results beyond tolerance")
        del model, moe
        gc.collect()
        write_json(a.out / "info.partial.json", info)
    mdp = dict(connections=[dict(devices=list(range(4)), type="p2p")],
               partitions=[dict(name="Partition0", devices=[dict(deviceId=i, numCores=16) for i in range(4)])])
    write_json(a.out / "mdp.json", mdp)
    for variant in hidden:
        stitch(a.out / variant, layers)
        np.save(a.out / variant / "y_cpu.npy", hidden[variant].float().numpy())
    write_json(a.out / "info.json", info)
    print(f"EXPORT_COMPLETE {a.out}", flush=True)


if __name__ == "__main__":
    main()
