#!/usr/bin/env python3
"""Export ONE decoder layer of Qwen3-30B-A3B as a 2-card QPC graph for the end-to-end path-2 test:
   input  h  [T,H] fp16  (residual stream entering the layer)
   graph  input_layernorm -> GQA attention (q/k norm, RoPE, causal) -> residual -> post_attention_layernorm
          -> MoE restricted to ONE lane group (router over all 128 experts, only lanes [l0,l1) materialised,
             per-stage padding widths)            [tier-bench MoELayerEP, real weights]
   output y  [T,H] fp16 = residual + MoE partial   (--with-residual, the hot QPC)   or   MoE partial (the cold QPC)
The host adds hot + cold outputs = residual stream entering the next layer.
  python3 tools/moe_e2e_layer_export.py --layer 3 --out DIR --lane-range 0,32 --stage-widths 128,32 --order order_L3.json --x-npy ref/h_L3.npy --with-residual
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/chihao/qeff-venv/lib/python3.10/site-packages")
from moe_tier_bench_export import MoELayerEP
from QEfficient.customop.rms_norm import CustomRMSNormFunc      # compiler-known RMSNorm: variance kept in fp32 on the card.
# With plain ops under -convert-to-fp16 the sum of squares overflows fp16 for the "massive activation" tokens (|h| ~ 1e3 from
# layer 2 on): that token is normalised to zero, its routing and expert outputs are garbage (found 2026-09-20 at layer 3).
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
HF = "/home/chihao/models/qwen3_30b_a3b/hf"


class E2ELayer(nn.Module):
    def __init__(self, cfg, W, l, T, moe, with_residual):
        super().__init__()
        H, nh, nkv, hd = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.nh, self.nkv, self.hd, self.T, self.eps, self.with_residual = nh, nkv, hd, T, cfg.rms_norm_eps, with_residual
        p = f"model.layers.{l}."
        P = lambda k: nn.Parameter(W(p + k), requires_grad=False)
        self.w_in, self.w_post, self.w_qn, self.w_kn = P("input_layernorm.weight"), P("post_attention_layernorm.weight"), P("self_attn.q_norm.weight"), P("self_attn.k_norm.weight")
        self.wq, self.wk, self.wv, self.wo = P("self_attn.q_proj.weight"), P("self_attn.k_proj.weight"), P("self_attn.v_proj.weight"), P("self_attn.o_proj.weight")
        rot = Qwen3MoeRotaryEmbedding(cfg); cos, sin = rot(torch.zeros(1, T, H), torch.arange(T)[None])
        self.register_buffer("cos", cos.float()); self.register_buffer("sin", sin.float())           # [1,T,hd]
        self.register_buffer("mask", torch.full((T, T), float("-inf")).triu(1))
        self.moe = moe

    def rms(self, x, w): return CustomRMSNormFunc.apply(x, w, self.eps)
    def rope(self, x):                                                                              # [1,h,T,hd]
        half = x.shape[-1] // 2; rx = torch.cat((-x[..., half:], x[..., :half]), -1); return x * self.cos[:, None] + rx * self.sin[:, None]

    def forward(self, h16):                                                                         # [T,H] fp16
        T = self.T; h = h16.float()
        x = self.rms(h, self.w_in)
        q = self.rms(F.linear(x, self.wq).view(1, T, self.nh, self.hd), self.w_qn).transpose(1, 2)
        k = self.rms(F.linear(x, self.wk).view(1, T, self.nkv, self.hd), self.w_kn).transpose(1, 2)
        v = F.linear(x, self.wv).view(1, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q), self.rope(k); rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        att = F.softmax((q @ k.transpose(-1, -2)) * (self.hd ** -0.5) + self.mask, dim=-1)
        a = (att @ v).transpose(1, 2).reshape(1, T, self.nh * self.hd)
        h = h + F.linear(a, self.wo)[0]
        x2 = self.rms(h, self.w_post)
        part = self.moe(x2.half()).float()                                                          # this lane group's MoE partial [T,H]
        return (h + part).half() if self.with_residual else part.half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lane-range", required=True, help="l0,l1 of the full 64-lane layout (hot 0,32 / cold 32,64)")
    ap.add_argument("--stage-widths", required=True, help="w0,w1 padding rows of this group's two stages")
    ap.add_argument("--order", required=True, help="lane-major expert order json (len 128)")
    ap.add_argument("--x-npy", required=True, help="reference residual stream entering this layer [T,H] (fp32 npy)")
    ap.add_argument("--with-residual", action="store_true"); ap.add_argument("-T", type=int, default=128); ap.add_argument("--num-devices", type=int, default=4)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    cfg = AutoConfig.from_pretrained(HF); H, I, K, S, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok, 2, cfg.num_experts
    wmap = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]; fh = {}
    def W(key):
        f = wmap[key]
        if f not in fh: fh[f] = safe_open(os.path.join(HF, f), framework="pt")
        return fh[f].get_tensor(key).float()
    l0, l1 = [int(v) for v in a.lane_range.split(",")]; P = l1 - l0; sw = [int(v) for v in a.stage_widths.split(",")]
    order = json.load(open(a.order)); P_full = len(order) // S; assert sorted(order) == list(range(E)) and P_full == 64
    L = a.layer; p = f"model.layers.{L}."
    router = W(p + "mlp.gate.weight")[order]                                                          # [E,H] lane-major
    g = lambda e, k: W(p + f"mlp.experts.{order[e]}.{k}.weight").T
    lanes = range(l0, l1)
    Wg = torch.stack([torch.stack([g(s * P_full + q, "gate_proj") for s in range(S)]) for q in lanes])
    Wu = torch.stack([torch.stack([g(s * P_full + q, "up_proj") for s in range(S)]) for q in lanes])
    Wd = torch.stack([torch.stack([g(s * P_full + q, "down_proj") for s in range(S)]) for q in lanes])
    moe = MoELayerEP(a.T, H, I, K, S, [(sw[0], P)], a.num_devices, stage_widths=sw, real=dict(router=router, Wg=Wg, Wu=Wu, Wd=Wd), lane_range=[l0, l1])
    m = E2ELayer(cfg, W, L, a.T, moe, a.with_residual).eval(); fh.clear()
    x = torch.from_numpy(np.load(a.x_npy)).float()[: a.T]; x16 = x.half()
    with torch.no_grad():
        y = m(x16)
    x16.numpy().tofile(os.path.join(a.out, "x.bin")); np.save(os.path.join(a.out, "y_torch.npy"), y.float().numpy())
    onnx_path = os.path.join(a.out, "moe_layer.onnx")
    with torch.no_grad():
        torch.onnx.export(m, (x16,), onnx_path, input_names=["x"], output_names=["y"], dynamo=False, opset_version=17, do_constant_folding=True)
    import onnx
    from QEfficient.base.onnx_transforms import CustomOpTransform, FP16ClipTransform
    mo = onnx.load(onnx_path); CustomOpTransform.apply(mo); fmax = float(np.finfo(np.float16).max)
    for init in mo.graph.initializer: FP16ClipTransform.apply(init, a.out, fmax, -fmax)
    for f in os.listdir(a.out):
        if f.endswith(".onnx.data") or (f.startswith("onnx__") and not f.endswith(".onnx")): os.remove(os.path.join(a.out, f))
    onnx.save(mo, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True, location="moe_layer.onnx.data", size_threshold=1024)
    json.dump(dict(layer=L, lane_range=[l0, l1], stage_widths=sw, with_residual=a.with_residual, T=a.T, order=a.order, x_npy=a.x_npy,
                   nodes=len(mo.graph.node)), open(os.path.join(a.out, "info.json"), "w"), indent=1)
    json.dump({"connections": [{"devices": [0, 1], "type": "p2p"}], "partitions": [{"devices": [{"deviceId": 0, "numCores": 16}, {"deviceId": 1, "numCores": 16}], "name": "Partition0"}]},
              open(os.path.join(a.out, "mdp.json"), "w"), indent=1)
    print(f"EXPORT_OK layer {L} lanes {l0}-{l1} widths {sw} residual={a.with_residual}: {onnx_path} ({os.path.getsize(onnx_path)/2**20:.0f} MB + data, {time.time()-t0:.0f}s, {len(mo.graph.node)} nodes)")


if __name__ == "__main__":
    main()
