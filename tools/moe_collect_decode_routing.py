#!/usr/bin/env python3
"""Collect per-step decode routing for a BATCH of prompts (CPU, HF model, greedy).
B prompts are left-padded, prefilled once, then decoded N steps together. At every decode step the
router hook records, for each layer and each sequence in the batch, the top-k expert indices.
Output npz: dec [N, L, B, K] int16, gen_ids [B, N], prompt_len [B], meta.
Usage: python3 moe_collect_decode_routing.py --bench gsm8k -B 32 -N 32 --out routing_decode_gsm8k_B32.npz"""
import argparse, json, time, os, sys, numpy as np, torch
BENCH = {"gsm8k": ("/home/chihao/models/bench/gsm8k_test.jsonl", "question"),
         "humaneval": ("/home/chihao/models/bench/HumanEval.jsonl", "prompt")}
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/chihao/models/qwen3_30b_a3b/hf"); ap.add_argument("--bench", default="gsm8k")
ap.add_argument("-B", type=int, default=32); ap.add_argument("-N", type=int, default=32); ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--dtype", default="bfloat16")
a = ap.parse_args(); t0 = time.time()
def log(*x): print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
path, field = BENCH[a.bench]; items = [json.loads(l) for l in open(path)][a.skip:a.skip + a.B]
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
cfg = model.config; L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
log(f"loaded; B={a.B} N={a.N} L={L} E={E} K={K}")
texts = []
for it in items:
    msgs = [{"role": "user", "content": it[field]}]
    try: s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    texts.append(s)
enc = tok(texts, return_tensors="pt", padding=True)
ids, am = enc["input_ids"], enc["attention_mask"]; plen = am.sum(1).numpy()
log(f"prompt lens {plen.min()}..{plen.max()}, padded T={ids.shape[1]}")
captured = {}
def mk(i):
    def hook(mod, inp, out): captured[i] = out[2].detach().to(torch.int16).cpu().numpy()
    return hook
for i, layer in enumerate(model.model.layers): layer.mlp.gate.register_forward_hook(mk(i))
dec = np.zeros((a.N, L, a.B, K), np.int16); gen = np.zeros((a.B, a.N), np.int64)
with torch.no_grad():
    t1 = time.time()
    out = model(input_ids=ids, attention_mask=am, use_cache=True)
    log(f"prefill done {time.time()-t1:.1f}s ({ids.numel()/(time.time()-t1):.1f} tok/s incl. pad)")
    # keep the prefill routing of the LAST real token of each sequence (the position that produces decode token 0)
    pre_last = np.stack([captured[i].reshape(a.B, ids.shape[1], K)[:, -1, :] for i in range(L)])   # [L,B,K]
    past = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
    pos = am.sum(1)                                   # next position id per sequence
    for s in range(a.N):
        gen[:, s] = nxt.numpy(); captured.clear(); t1 = time.time()
        am = torch.cat([am, torch.ones(a.B, 1, dtype=am.dtype)], 1)
        out = model(input_ids=nxt.view(a.B, 1), attention_mask=am, position_ids=pos.view(a.B, 1), past_key_values=past, use_cache=True)
        past = out.past_key_values; pos = pos + 1
        for i in range(L): dec[s, i] = captured[i].reshape(a.B, K)
        nxt = out.logits[:, -1].argmax(-1)
        if s % 4 == 0: log(f"  step {s+1}/{a.N}  {time.time()-t1:.2f}s")
np.savez_compressed(a.out, dec=dec, pre_last=pre_last, gen_ids=gen, prompt_len=plen,
    meta=np.array(json.dumps({"bench": a.bench, "B": a.B, "N": a.N, "skip": a.skip, "L": L, "E": E, "K": K, "model": a.model}), dtype=object))
log(f"WROTE {a.out}  dec shape {dec.shape}; sample decoded text[0]: {tok.decode(gen[0][:24])!r}")
