#!/usr/bin/env python3
"""Phase 2: collect per-token expert routing from a MoE model on real benchmarks (CPU).

Why CPU and not the cards: Phase 1 measured that on AI 100 the compiled graph is
index-driven with fixed shapes, so cycle counts are input-independent (real token vs
random input differed by <=0.64%). Which experts get picked is therefore only
observable from the torch model, never from the device trace.

Captures, per prompt, per layer, per token: the top-k expert indices chosen by the
router -- i.e. the output[2] of Qwen3MoeTopKRouter.forward.

Only a PREFILL forward is run (no generation): that is what the expert-coverage
question needs, and it is ~100x cheaper on CPU than generating.

Usage:
  python3 moe_collect_routing.py --bench gsm8k --n 50 --out routing_gsm8k.npz
"""
import argparse, json, time, os, sys, gzip
import numpy as np
import torch

BENCH = {
    "gsm8k":     ("/home/chihao/models/bench/gsm8k_test.jsonl", "question"),
    "humaneval": ("/home/chihao/models/bench/HumanEval.jsonl",  "prompt"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/chihao/models/qwen3_30b_a3b/hf")
    ap.add_argument("--bench", required=True, choices=list(BENCH))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=1024, help="truncate prompts longer than this")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--chat-template", action="store_true", default=True)
    ap.add_argument("--raw", dest="chat_template", action="store_false",
                    help="feed the raw benchmark text instead of the chat template")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    if a.threads: torch.set_num_threads(a.threads)
    t0 = time.time()
    def log(*x): print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)

    path, field = BENCH[a.bench]
    items = [json.loads(l) for l in open(path)][: a.n]
    log(f"{a.bench}: {len(items)} items from {path}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    log(f"loading model ({a.dtype}) on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map=None,
    ).eval()
    cfg = model.config
    L, E, K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
    log(f"loaded: {L} layers, {E} experts, top-{K}")

    # ---- hook every router; output[2] is router_indices [n_tok, K] ----
    captured = {}
    def mk(idx):
        def hook(mod, inp, out):
            captured[idx] = out[2].detach().to(torch.int16).cpu().numpy()
        return hook
    handles = []
    for i, layer in enumerate(model.model.layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is None:
            sys.exit(f"layer {i} has no mlp.gate — not a MoE layer?")
        handles.append(gate.register_forward_hook(mk(i)))
    log(f"hooked {len(handles)} routers")

    out, meta = {}, []
    for n, it in enumerate(items):
        text = it[field]
        if a.chat_template:
            msgs = [{"role": "user", "content": text}]
            try:
                s = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            s = text
        ids = tok(s, return_tensors="pt", truncation=True, max_length=a.max_len)
        T = ids["input_ids"].shape[1]
        captured.clear()
        t1 = time.time()
        with torch.no_grad():
            model(**ids, use_cache=False)
        dt = time.time() - t1
        if len(captured) != L:
            sys.exit(f"prompt {n}: captured {len(captured)} layers, expected {L}")
        arr = np.stack([captured[i] for i in range(L)])        # [L, T, K]
        assert arr.shape == (L, T, K), (arr.shape, (L, T, K))
        out[f"p{n}"] = arr
        meta.append({"i": n, "tokens": T, "sec": round(dt, 2)})
        log(f"  [{n+1}/{len(items)}] T={T:4d}  {dt:6.2f}s  ({T/dt:5.1f} tok/s)")

    for h in handles: h.remove()
    np.savez_compressed(
        a.out,
        meta=np.array(json.dumps({"bench": a.bench, "model": a.model, "layers": L,
                                  "experts": E, "top_k": K,
                                  "chat_template": a.chat_template,
                                  "prompts": meta}), dtype=object),
        **out,
    )
    tot = sum(m["tokens"] for m in meta)
    log(f"WROTE {a.out}  ({len(out)} prompts, {tot} tokens, "
        f"{tot*L*K:,} routing decisions, {os.path.getsize(a.out)/2**20:.1f} MB)")

if __name__ == "__main__":
    main()
