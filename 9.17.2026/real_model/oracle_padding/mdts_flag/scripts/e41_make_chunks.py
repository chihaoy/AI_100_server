#!/usr/bin/env python3
"""Held-out routing chunks for experiment 4b (E41). Each workload's items (documents) are split into a calibration half and
an evaluation half by item index BEFORE chunking; within a half, consecutive items are joined into one chat turn until the
turn reaches T tokens and truncated to T (the make_prompts_text.py rule), each item used once. Writes
full_model/e41/chunks_T<T>_<workload>_<cal|eval>.bin (int64, n x T) and chunks_index.json (item ranges per chunk).
Usage: e41_make_chunks.py <mdts_flag dir> [T ...]"""
import sys, os, json, numpy as np
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); import moe_layer0_router_oncard as R
from transformers import AutoTokenizer
M = sys.argv[1]; Ts = [int(v) for v in sys.argv[2:]] or [128, 256, 512]; O = f'{M}/full_model/e41'; os.makedirs(O, exist_ok=True)
S = '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'
def texts(path, key=None):
    rows = [json.loads(l) for l in open(path)]
    return [r[key] if key else r['text'] for r in rows]
W = {'gsm8k': texts('/home/chihao/models/bench/gsm8k_test.jsonl', 'question'),
     'humaneval': texts('/home/chihao/models/bench/HumanEval.jsonl', 'prompt'),
     'mmlu': [f"Question: {r['question']}\n" + '\n'.join(f"{'ABCD'[c]}. {ch}" for c, ch in enumerate(r['choices'])) + "\nAnswer:"
              for r in (json.loads(l) for l in open(f'{S}/e21/data/mmlu_test.jsonl'))],     # as make_prompts.py, in file order (subjects grouped)
     'swe': texts(f'{S}/e21/data/swebench_lite_test.jsonl', 'problem_statement'),
     'alpaca': texts(f'{S}/e24/data/alpaca.jsonl'), 'news': texts(f'{S}/e24/data/news.jsonl'),
     'chinese': texts(f'{S}/e24/data/chinese.jsonl'), 'code_js': texts(f'{S}/e24/data/code_js.jsonl')}
tok = AutoTokenizer.from_pretrained(R.HF)
def ids_of(text): return tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="np")["input_ids"][0]
index = json.load(open(f'{O}/chunks_index.json')) if os.path.exists(f'{O}/chunks_index.json') else {}
for T in Ts:
    for w, items in W.items():
        half = len(items) // 2
        for split, part, base in (('cal', items[:half], 0), ('eval', items[half:], half)):
            chunks, ranges, i = [], [], 0
            while i < len(part):
                buf, j = [], i
                while j < len(part):
                    buf.append(part[j]); j += 1; ids = ids_of("\n\n".join(buf))
                    if len(ids) >= T: break
                if len(ids) < T: break                      # leftover items do not fill a chunk
                chunks.append(ids[:T].astype(np.int64)); ranges.append([base + i, base + j]); i = j
            np.array(chunks, np.int64).reshape(-1, T).tofile(f'{O}/chunks_T{T}_{w}_{split}.bin')
            index[f'T{T}_{w}_{split}'] = ranges
            print(f'T={T} {w:9s} {split:4s}: {len(items):5d} items, {len(chunks):4d} chunks', flush=True)
json.dump(index, open(f'{O}/chunks_index.json', 'w'))
