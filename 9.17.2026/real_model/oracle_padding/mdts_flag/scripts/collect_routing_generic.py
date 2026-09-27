#!/usr/bin/env python3
"""Real routing for the flexibility study: FP32 CPU forward of all 48 layers (same math as tools/moe_e2e_cpu_ref.py) on
chat prompts built from consecutive GSM8K questions until they reach T tokens (truncated to T), plus the natural
single-question prompts that reach 128 tokens. Saves per prompt: routing top-8 indices and normalized weights for every
layer, and the layer-2 MoE input. Usage: collect_routing_generic.py <out dir> <threads> <prompts.npz (names + one int64 array per name)>"""
import sys, os, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); import moe_layer0_router_oncard as R
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
out, threads = sys.argv[1], int(sys.argv[2]); os.makedirs(out, exist_ok=True); torch.set_num_threads(threads)
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
HF = R.HF; cfg = AutoConfig.from_pretrained(HF); H, nh, nkv, hd, E, K, eps = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.num_experts, cfg.num_experts_per_tok, cfg.rms_norm_eps
wmap = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]; _open = {}
def W(key):
    f = wmap[key]
    if f not in _open: _open[f] = safe_open(os.path.join(HF, f), framework="pt")
    return _open[f].get_tensor(key).float()
tok = AutoTokenizer.from_pretrained(HF)
specs_npz = np.load(sys.argv[3], allow_pickle=True); specs = [(str(n), np.asarray(specs_npz[str(n)], np.int64)) for n in specs_npz['names']]
def rmsnorm(x, w): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
rot = Qwen3MoeRotaryEmbedding(cfg)
def run(ids):
    T = len(ids); cos, sin = rot(torch.zeros(1, T, H), torch.arange(T)[None]); cos, sin = cos.float(), sin.float()
    def rope(x): half = x.shape[-1] // 2; rx = torch.cat((-x[..., half:], x[..., :half]), -1); return x * cos + rx * sin
    mask = torch.full((T, T), float("-inf")).triu(1)
    h = W("model.embed_tokens.weight")[torch.from_numpy(ids)]; ridx = np.zeros((48, T, K), np.int16); rw = np.zeros((48, T, K), np.float32); x2_l2 = None
    for l in range(48):
        p = f"model.layers.{l}."
        x = rmsnorm(h, W(p + "input_layernorm.weight"))
        q = rmsnorm((x @ W(p + "self_attn.q_proj.weight").T).view(T, nh, hd), W(p + "self_attn.q_norm.weight")).transpose(0, 1)
        k = rmsnorm((x @ W(p + "self_attn.k_proj.weight").T).view(T, nkv, hd), W(p + "self_attn.k_norm.weight")).transpose(0, 1)
        v = (x @ W(p + "self_attn.v_proj.weight").T).view(T, nkv, hd).transpose(0, 1)
        q, k = rope(q), rope(k); rep = nh // nkv; k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
        att = F.softmax((q @ k.transpose(-1, -2)) * hd ** -0.5 + mask, -1)
        o = (att @ v).transpose(0, 1).reshape(T, nh * hd) @ W(p + "self_attn.o_proj.weight").T; h = h + o
        x2 = rmsnorm(h, W(p + "post_attention_layernorm.weight"))
        logits = x2 @ W(p + "mlp.gate.weight").T; probs = F.softmax(logits, -1); top_p, top_i = torch.topk(probs, K, -1); top_p = top_p / top_p.sum(-1, keepdim=True)
        ridx[l] = top_i.numpy(); rw[l] = top_p.numpy()
        if l == 2: x2_l2 = x2.numpy().astype(np.float16)
        moe = torch.zeros_like(x2)
        for e in range(E):
            sel = (top_i == e); rows = sel.any(1).nonzero().squeeze(1)
            if len(rows) == 0: continue
            w = (top_p * sel)[rows].sum(1, keepdim=True); xs = x2[rows]
            g = xs @ W(p + f"mlp.experts.{e}.gate_proj.weight").T; u = xs @ W(p + f"mlp.experts.{e}.up_proj.weight").T
            moe[rows] += w * ((F.silu(g) * u) @ W(p + f"mlp.experts.{e}.down_proj.weight").T)
        h = h + moe
        for f in list(_open):
            if not any(wmap[k_] == f for k_ in (f"model.layers.{l+1}.input_layernorm.weight",) if k_ in wmap): _open.pop(f)
    return ridx, rw, x2_l2
with torch.no_grad():
    for name, ids in specs:
        f = f"{out}/{name}.npz"
        if os.path.exists(f): continue
        ridx, rw, x2 = run(np.asarray(ids, np.int64)); np.savez(f, ids=np.asarray(ids, np.int64), routing_idx=ridx, routing_w=rw, x2_L2=x2)
        c2 = np.bincount(ridx[2].ravel(), minlength=E); log(f"{name}: T={len(ids)} layer2 active {int((c2>0).sum())} max {c2.max()} empty {int((c2==0).sum())}")
log("done")
