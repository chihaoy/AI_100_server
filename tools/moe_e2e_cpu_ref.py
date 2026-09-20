#!/usr/bin/env python3
"""CPU fp32 reference for the end-to-end prefill test (Qwen3-30B-A3B, one GSM8K prompt, T tokens, no padding).
Explicit implementation of the Qwen3-MoE decoder layer (RMSNorm, GQA attention with q/k norm + RoPE, top-8 router with
norm_topk_prob, per-expert MLP), fp32 weights loaded layer by layer from the safetensors shards.
Saves the residual stream entering every layer (h_L0 .. h_L48), the MoE-block input of every layer (x2), per-layer
top-8 routing, final logits and argmax tokens.  Validates itself against (a) x_real_p41.npy (layer-0 MoE input computed
earlier by moe_layer0_router_oncard's module) and (b) the bf16-HF routing capture routing_gsm8k_300.npz.
  python3 tools/moe_e2e_cpu_ref.py --prompt 41 --T 128 --out 9.17.2026/e2e/ref
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_layer0_router_oncard as R
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

HF = R.HF
ap = argparse.ArgumentParser(); ap.add_argument("--prompt", type=int, default=41); ap.add_argument("--T", type=int, default=128)
ap.add_argument("--out", required=True); ap.add_argument("--layers", type=int, default=48); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_num_threads(a.threads)
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
cfg = AutoConfig.from_pretrained(HF); H, nh, nkv, hd, E, K, eps = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.num_experts, cfg.num_experts_per_tok, cfg.rms_norm_eps
wmap = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]
_open = {}
def W(key):
    f = wmap[key]
    if f not in _open: _open[f] = safe_open(os.path.join(HF, f), framework="pt")
    return _open[f].get_tensor(key).float()

tok = AutoTokenizer.from_pretrained(HF); ids = R.prompts(tok, a.prompt + 1, 4096)[a.prompt]
T = a.T; assert len(ids) >= T, (len(ids), T); ids = np.asarray(ids[:T], np.int64)
np.save(os.path.join(a.out, "input_ids.npy"), ids); ids.tofile(os.path.join(a.out, "input_ids_i64.bin")); np.arange(T, dtype=np.int64).tofile(os.path.join(a.out, "position_ids_i64.bin"))
log(f"prompt {a.prompt}: {len(R.prompts(tok, a.prompt + 1, 4096)[a.prompt])} tokens, using first {T}")

def rmsnorm(x, w): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
rot = Qwen3MoeRotaryEmbedding(cfg); cos, sin = rot(torch.zeros(1, T, H), torch.arange(T)[None]); cos, sin = cos.float(), sin.float()   # [1,T,hd]
def rope(x):                                                     # x [h, T, hd]
    half = x.shape[-1] // 2; rx = torch.cat((-x[..., half:], x[..., :half]), -1); return x * cos + rx * sin
mask = torch.full((T, T), float("-inf")).triu(1)

def layer(l, h):
    p = f"model.layers.{l}."
    x = rmsnorm(h, W(p + "input_layernorm.weight"))
    q = rmsnorm((x @ W(p + "self_attn.q_proj.weight").T).view(T, nh, hd), W(p + "self_attn.q_norm.weight")).transpose(0, 1)   # [nh,T,hd]
    k = rmsnorm((x @ W(p + "self_attn.k_proj.weight").T).view(T, nkv, hd), W(p + "self_attn.k_norm.weight")).transpose(0, 1)
    v = (x @ W(p + "self_attn.v_proj.weight").T).view(T, nkv, hd).transpose(0, 1)
    q, k = rope(q), rope(k); rep = nh // nkv; k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    att = F.softmax((q @ k.transpose(-1, -2)) * hd ** -0.5 + mask, -1)
    o = (att @ v).transpose(0, 1).reshape(T, nh * hd) @ W(p + "self_attn.o_proj.weight").T
    h = h + o
    x2 = rmsnorm(h, W(p + "post_attention_layernorm.weight"))
    logits = x2 @ W(p + "mlp.gate.weight").T; probs = F.softmax(logits, -1); top_p, top_i = torch.topk(probs, K, -1); top_p = top_p / top_p.sum(-1, keepdim=True)
    moe = torch.zeros_like(x2)
    for e in range(E):
        sel = (top_i == e); rows = sel.any(1).nonzero().squeeze(1)
        if len(rows) == 0: continue
        w = (top_p * sel)[rows].sum(1, keepdim=True)
        xs = x2[rows]; g = xs @ W(p + f"mlp.experts.{e}.gate_proj.weight").T; u = xs @ W(p + f"mlp.experts.{e}.up_proj.weight").T
        moe[rows] += w * ((F.silu(g) * u) @ W(p + f"mlp.experts.{e}.down_proj.weight").T)
    return h + moe, x2, top_i

with torch.no_grad():
    h = W("model.embed_tokens.weight")[torch.from_numpy(ids)]                     # [T,H] fp32
    z = np.load("/home/chihao/mllm/perop_moe/routing/routing_gsm8k_300.npz", allow_pickle=True); ref_route = z[f"p{a.prompt}"][:, :T, :]
    routing = np.zeros((a.layers, T, K), np.int16); ov = []
    for l in range(a.layers):
        np.save(os.path.join(a.out, f"h_L{l}.npy"), h.numpy())
        h, x2, top_i = layer(l, h)
        np.save(os.path.join(a.out, f"x2_L{l}.npy"), x2.numpy()); routing[l] = top_i.numpy()
        if l == 0:
            xr = np.load("/home/chihao/mllm/9.17.2026/real_l0/x_real_p41.npy").astype(np.float32)
            log(f"layer0 MoE input vs x_real_p41.npy: max|diff| {np.abs(x2.numpy()-xr).max():.4g} (max|x| {np.abs(xr).max():.3g})")
        o = np.mean([len(set(ref_route[l, t]) & set(routing[l, t])) / K for t in range(T)]); ov.append(o)
        cnt = np.bincount(routing[l].ravel(), minlength=E)
        log(f"layer {l:2d} done; top-8 overlap vs bf16-HF capture {o:.3f}; busiest expert {cnt.argmax()} gets {cnt.max()} tokens; |h| max {h.abs().max():.2f}")
        for f in list(_open): 
            if not any(wmap[k] == f for k in (f"model.layers.{l+1}.input_layernorm.weight",) if k in wmap): _open.pop(f)
    np.save(os.path.join(a.out, f"h_L{a.layers}.npy"), h.numpy()); np.save(os.path.join(a.out, "routing.npy"), routing)
    hn = rmsnorm(h, W("model.norm.weight")); logits = hn @ W("lm_head.weight").T
    np.save(os.path.join(a.out, "logits.npy"), logits.numpy()); am = logits.argmax(-1).numpy(); np.save(os.path.join(a.out, "argmax.npy"), am)
    log(f"mean routing overlap over layers: {np.mean(ov):.3f}  (min {np.min(ov):.3f} at layer {int(np.argmin(ov))})")
    log("argmax next tokens (positions 0..15):", tok.decode(am[:16].tolist()))
    log("last position: input tail ...", repr(tok.decode(ids[-8:].tolist())), " -> predicted next:", repr(tok.decode([int(am[-1])])))
    json.dump(dict(prompt=a.prompt, T=T, overlap=ov, argmax_last=int(am[-1])), open(os.path.join(a.out, "summary.json"), "w"))
log("done")
