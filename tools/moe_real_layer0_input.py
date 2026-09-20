#!/usr/bin/env python3
"""Real layer-0 MoE-block input for one GSM8K prompt: x = post_attention_layernorm(embed + attn) [T, H] fp16,
computed on CPU with the real checkpoint (same module as moe_layer0_router_oncard.py), truncated/padded to T.
Also prints the CPU top-8 token counts per expert for that x, to cross-check against the tier graph's router.
  python3 moe_real_layer0_input.py --prompt 41 --T 128 --out DIR/x_real_p41.npy
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_layer0_router_oncard as R

ap = argparse.ArgumentParser(); ap.add_argument("--prompt", type=int, default=41); ap.add_argument("--T", type=int, default=128)
ap.add_argument("--out", required=True); a = ap.parse_args()
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file
tok = AutoTokenizer.from_pretrained(R.HF); tok.pad_token = tok.pad_token or tok.eos_token
ids = R.prompts(tok, a.prompt + 1, 4096)[a.prompt]
n = min(len(ids), a.T); x_ids = np.full((1, a.T), tok.pad_token_id, np.int32); x_ids[0, :n] = ids[:n]
cfg = AutoConfig.from_pretrained(R.HF)
sd = load_file(os.path.join(R.HF, "model-00001-of-00016.safetensors"))
sd = {k: v for k, v in sd.items() if "experts" not in k and (k.startswith("model.layers.0.") or k.startswith("model.embed"))}
m = R.Layer0Router(cfg, sd, a.T).eval()
with torch.no_grad():
    m.return_h = True; h = m(torch.from_numpy(x_ids))
    m.return_h = False; ti, _ = m(torch.from_numpy(x_ids))
np.save(a.out, h.half().numpy())
cnt = np.bincount(ti[:n].numpy().ravel(), minlength=128)
json.dump({"prompt": a.prompt, "real_tokens": int(n), "T": a.T, "cpu_top8_counts": cnt.tolist()}, open(a.out.replace(".npy", ".json"), "w"))
print(f"prompt {a.prompt}: {len(ids)} tokens -> using first {n} of T={a.T}; x saved {a.out} {tuple(h.shape)}")
print("CPU top-8 counts: max", cnt.max(), "experts used", int((cnt > 0).sum()), "top:", [(int(e), int(cnt[e])) for e in np.argsort(-cnt)[:6]])
