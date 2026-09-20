#!/usr/bin/env python3
"""Single-layer MoE DECODE_BMM (gather + bmm) micro-benchmark exporter.

Same router / shapes / random-weight convention as moe_tier_bench_export.py, but the MoE body is
QEfficient's decode flavour (transformers/moe/flavours.py::moe_decode_bmm): weights stay in the
canonical [E, H, I] layout, the top-k expert matrices are gathered by index for every (token, k)
pair and multiplied as a batched GEMM. This is the graph the real Qwen3-30B-A3B decode QPC uses
(TP across cards; the compiler slices the gathered tensors, no expert->card mapping).

  python3 moe_decode_bmm_export.py --out DIR --groups 128:64 -T <batch>
(--groups / --num-devices accepted for harness compatibility; only E = tier*n_lanes*S is used.)
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/chihao/qeff-venv/lib/python3.10/site-packages")


class MoELayerDecodeBMM(nn.Module):
    def __init__(self, T, H, I, K, E, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T, self.H, self.I, self.K, self.E = T, H, I, K, E
        self.router = nn.Parameter((torch.randn(E, H, generator=g) * 0.02).half())
        self.Wg = nn.Parameter((torch.randn(E, H, I, generator=g) * 0.02).half())
        self.Wu = nn.Parameter((torch.randn(E, H, I, generator=g) * 0.02).half())
        self.Wd = nn.Parameter((torch.randn(E, I, H, generator=g) * 0.02).half())

    def route(self, x):                                     # identical to the EP bench router
        logits = F.linear(x, self.router)
        probs = F.softmax(logits, dtype=torch.float, dim=-1).to(logits.dtype)
        top_w, top_i = torch.topk(probs, self.K, dim=-1)
        top_w = top_w / torch.einsum("bk->b", top_w).unsqueeze(-1)
        return top_i, top_w.to(x.dtype)

    def forward(self, x):                                   # == QEfficient moe_decode_bmm
        T, H = x.shape
        top_i, top_w = self.route(x)
        idx = top_i.reshape(-1)
        gate_proj, up_proj, down_proj = self.Wg[idx], self.Wu[idx], self.Wd[idx]
        expert_in = x.unsqueeze(1).expand(-1, self.K, -1).contiguous().view(-1, 1, H)
        gate = expert_in @ gate_proj
        up = expert_in @ up_proj
        down = (up * F.silu(gate)) @ down_proj
        out = down.view(T, self.K, H) * top_w.unsqueeze(-1)
        return torch.einsum("bnd->bd", out)


def reference(m, x):
    top_i, top_w = m.route(x)
    out = torch.zeros(x.shape[0], m.H)
    for t in range(x.shape[0]):
        for k in range(m.K):
            e = top_i[t, k].item()
            xs = x[t:t + 1].float()
            y = (F.silu(xs @ m.Wg[e].float()) * (xs @ m.Wu[e].float())) @ m.Wd[e].float()
            out[t] += y[0] * top_w[t, k].float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="128:64", help="tier:n_lanes (E = tier... no: E = n_lanes*S), harness compat")
    ap.add_argument("-T", type=int, default=1, help="decode batch (tokens per step)")
    ap.add_argument("-H", type=int, default=2048)
    ap.add_argument("-I", type=int, default=768)
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("-S", type=int, default=2)
    ap.add_argument("-E", type=int, default=None)
    ap.add_argument("--num-devices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.E is None:
        a.E = sum(int(item.split(":")[1]) for item in a.groups.split(",")) * a.S
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    m = MoELayerDecodeBMM(a.T, a.H, a.I, a.K, a.E, seed=a.seed).eval()
    x = torch.randn(a.T, a.H).half()
    info = dict(flavour="decode_bmm", T=a.T, H=a.H, I=a.I, K=a.K, E=a.E, num_devices=a.num_devices,
                gathered_pairs=a.T * a.K, gathered_weight_MiB=a.T * a.K * 3 * a.H * a.I * 2 / 2**20)
    print(json.dumps(info, indent=1))
    with torch.no_grad():
        y = m(x)
        x.numpy().tofile(os.path.join(a.out, "x.bin"))
        np.save(os.path.join(a.out, "y_torch.npy"), y.float().numpy())
        ref = reference(m, x)
        np.save(os.path.join(a.out, "y_ref.npy"), ref.numpy())
        err = (y.float() - ref).abs().max().item(); scale = ref.abs().max().item()
        print(f"CHECK max|err|={err:.4g} max|ref|={scale:.4g} rel={err/max(scale,1e-9):.3g}")
    onnx_path = os.path.join(a.out, "moe_layer.onnx")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(m, (x,), onnx_path, input_names=["x"], output_names=["y"],
                          dynamo=False, opset_version=17, do_constant_folding=True)
    import onnx
    from QEfficient.base.onnx_transforms import CustomOpTransform, FP16ClipTransform
    model = onnx.load(onnx_path)
    CustomOpTransform.apply(model)
    fmax = float(np.finfo(np.float16).max)
    for init in model.graph.initializer:
        FP16ClipTransform.apply(init, a.out, fmax, -fmax)
    onnx.save(model, onnx_path)
    print(f"EXPORT_OK {onnx_path}  ({os.path.getsize(onnx_path)/2**20:.0f} MB, {time.time()-t0:.0f}s)")
    ops = {}
    for nd in model.graph.node:
        ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
    info["onnx_nodes"] = len(model.graph.node); info["onnx_ops"] = ops
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    print("nodes:", len(model.graph.node), ops)


if __name__ == "__main__":
    main()
