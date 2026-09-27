#!/usr/bin/env python3
"""Prompt sets for the cross-workload routing study: chat prompts truncated to T tokens.
mmlu: consecutive questions (shuffled across subjects, seed 0) with their four choices; swe: SWE-bench Lite problem statements
(issue text), one per prompt; humaneval: consecutive HumanEval function stubs. 50 prompts at T=128, 4 at T=512 each.
Usage: make_prompts.py <data dir> <out dir>"""
import sys, os, json, numpy as np
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); import moe_layer0_router_oncard as R
from transformers import AutoTokenizer
data, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); tok = AutoTokenizer.from_pretrained(R.HF); rng = np.random.default_rng(0)
def chat(text): return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def ids_of(text): return tok(chat(text), return_tensors="np")["input_ids"][0]
def concat_until(pieces, T, start, sep):
    i = start; buf = []
    while True:
        buf.append(pieces[i]); i += 1; ids = ids_of(sep.join(buf))
        if len(ids) >= T: return ids[:T], i
def build(name, pieces, sep, n128=50, n512=4):
    specs = {}; i = 0; names = []
    for k in range(n128): ids, i = concat_until(pieces, 128, i, sep); specs[f'T128_{name}{k}'] = ids; names.append(f'T128_{name}{k}')
    for k in range(n512): ids, i = concat_until(pieces, 512, i, sep); specs[f'T512_{name}{k}'] = ids; names.append(f'T512_{name}{k}')
    np.savez(f'{out}/prompts_{name}.npz', names=np.array(names), **specs); print(name, len(names), 'prompts, pieces consumed', i, flush=True)
mm = [json.loads(l) for l in open(f'{data}/mmlu_test.jsonl')]; order = rng.permutation(len(mm))
mm_pieces = [f"Question: {mm[j]['question']}\n" + '\n'.join(f"{'ABCD'[c]}. {ch}" for c, ch in enumerate(mm[j]['choices'])) + "\nAnswer:" for j in order]
build('mmlu', mm_pieces, '\n\n')
sw = [json.loads(l) for l in open(f'{data}/swebench_lite_test.jsonl')]
sw_pieces = [r['problem_statement'] for r in sw]
build('swe', sw_pieces, '\n\n')     # one statement is usually >= 128 tokens; the 512-token prompts concatenate a few
he = [json.loads(l) for l in open('/home/chihao/models/bench/HumanEval.jsonl')]
he_pieces = ["Complete the following Python function.\n\n" + r['prompt'] for r in he]
build('humaneval', he_pieces, '\n\n')
