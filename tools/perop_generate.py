#!/usr/bin/env python3
# Generate the model's actual RESPONSE for a prompt on the AI 100 — a correctness check to run
# alongside the per-op profiling. Uses QEfficient's KV-cache generation loop on the retained-state
# generation QPC (NOT the profiling QPC, which only does a single forward pass).
#
#   python3 perop_generate.py --prompt "what is the capital of china" --gen-len 20
import os, sys, argparse
os.environ.setdefault("HF_HOME", "/home/chihao/models/hf")

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--qpc", default="/home/chihao/models/qeff_native_cache/Qwen3ForCausalLM/"
                                  "Qwen3ForCausalLM-d165ffa1b02f89bc/qpc-b3c8aad261505a46/qpc",
                help="retained-state generation QPC (has prefill+decode specializations)")
ap.add_argument("--device", default="0", help="comma-separated card ids")
ap.add_argument("--gen-len", type=int, default=20)
ap.add_argument("--out", help="also write the response text here")
args = ap.parse_args()

from QEfficient.generation.text_generation_inference import cloud_ai_100_exec_kv
from QEfficient.utils import load_hf_tokenizer

tok = load_hf_tokenizer(pretrained_model_name_or_path=args.model)
devs = [int(x) for x in args.device.split(",")]
print(f">>> generating on card(s) {devs}  qpc={args.qpc}")
print(f">>> prompt: {args.prompt!r}\n")

res = cloud_ai_100_exec_kv(
    tokenizer=tok, qpc_path=args.qpc, device_id=devs,
    prompt=args.prompt, generation_len=args.gen_len,
)

# cloud_ai_100_exec_kv streams the text to stdout; also try to capture it for --out
def to_text(r):
    g = getattr(r, "generated_texts", r)
    while isinstance(g, (list, tuple)) and g:   # unwrap nested list(s)
        g = g[0]
    return g if isinstance(g, str) else None
text = to_text(res)
if args.out and text:
    open(args.out, "w").write(text.strip() + "\n")
    print(f"\n>>> response written to {args.out}")
