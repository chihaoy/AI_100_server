#!/usr/bin/env python3
"""Chat prompts of exactly T tokens from a {"text": ...} jsonl: consecutive items joined until T tokens, then truncated.
Usage: make_prompts_text.py <jsonl> <name> <out dir> <n> [T]"""
import sys, os, json, numpy as np
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); import moe_layer0_router_oncard as R
from transformers import AutoTokenizer
src, name, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); T = int(sys.argv[5]) if len(sys.argv) > 5 else 128
tok = AutoTokenizer.from_pretrained(R.HF); items = [json.loads(l)['text'] for l in open(src)]
def ids_of(text): return tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="np")["input_ids"][0]
specs, names, i = {}, [], 0
for k in range(n):
    buf = []
    while True:
        buf.append(items[i]); i += 1; ids = ids_of("\n\n".join(buf))
        if len(ids) >= T: break
    nm = f'T{T}_{name}{k}'; specs[nm] = ids[:T]; names.append(nm)
np.savez(f'{out}/prompts_{name}.npz', names=np.array(names), **specs); print(name, len(names), 'prompts, items consumed', i)
