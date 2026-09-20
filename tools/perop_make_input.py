#!/usr/bin/env python3
# Build a real-input bindings.json (+ bind/*.bin) for a Qwen3 profiling QPC so qaic-runner
# feeds YOUR prompt instead of random data. Drop the output into the QPC dir; qaic-runner -t
# picks it up automatically. Buffer shapes are computed from the spec (NO qaic-runner call,
# so it can't hang on a busy card).
#
#   python3 perop_make_input.py --qpc <qpc_dir> --prompt "Hello world"
#   python3 perop_make_input.py --qpc <qpc_dir> --token-ids 9707,1879,0
#
# NOTE: per-op *cycle* counts are shape-determined, so real inputs give the SAME timing as
# random. This only matters if you want real output logits or the run driven by your tokens.
import argparse, json, os, sys
from array import array   # stdlib; 'q' = int64 (8 bytes)

ap = argparse.ArgumentParser()
ap.add_argument("--qpc", required=True, help="QPC dir containing programqpc.bin")
ap.add_argument("--prompt", help="text prompt (tokenized with --tokenizer)")
ap.add_argument("--token-ids", help="comma-separated token ids (skips tokenizer)")
ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
ap.add_argument("--pad-id", type=int, default=0)
# shape: default read from <qpc>/../spec.json; Qwen3-8B model dims as defaults
ap.add_argument("--seqlen", type=int, help="override seq_len (else from spec.json)")
ap.add_argument("--ctxlen", type=int, help="override ctx_len (else from spec.json)")
ap.add_argument("--layers", type=int, default=36)      # Qwen3-8B
ap.add_argument("--kv-heads", type=int, default=8)
ap.add_argument("--head-dim", type=int, default=128)
args = ap.parse_args()

# 1. shape (seq_len / ctx_len) from spec.json next to the qpc dir, unless overridden
seqlen, ctxlen = args.seqlen, args.ctxlen
if seqlen is None or ctxlen is None:
    spec_path = os.path.join(args.qpc, "..", "spec.json")
    try:
        s = json.load(open(spec_path))["specializations"][0]
        seqlen = seqlen or int(s["seq_len"]);  ctxlen = ctxlen or int(s["ctx_len"])
    except Exception as e:
        sys.exit(f"could not read shape from {spec_path} ({e}); pass --seqlen and --ctxlen")
print(f"seq_len={seqlen}  ctx_len={ctxlen}  layers={args.layers} kv_heads={args.kv_heads} head_dim={args.head_dim}")

# 2. buffer list (names + byte sizes) — deterministic for Qwen3, no qaic-runner needed
kv_bytes = args.kv_heads * ctxlen * args.head_dim * 2         # fp16
bufs = [("input_ids", seqlen * 8), ("position_ids", seqlen * 8)]
for i in range(args.layers): bufs.append((f"past_key.{i}",   kv_bytes))
for i in range(args.layers): bufs.append((f"past_value.{i}", kv_bytes))

# 3. resolve token ids
if args.token_ids:
    ids = [int(x) for x in args.token_ids.split(",")]
elif args.prompt:
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(args.tokenizer)(args.prompt)["input_ids"]
else:
    sys.exit("provide --prompt or --token-ids")
ids = (ids + [args.pad_id] * seqlen)[:seqlen]                 # pad/truncate to seq_len
print(f"input_ids[:8]={ids[:8]}{' ...' if seqlen > 8 else ''}")

# 4. write one .bin per buffer (real input_ids/position_ids, zeros for KV) + bindings.json
binddir = os.path.join(args.qpc, "bind"); os.makedirs(binddir, exist_ok=True)
userBuffers = []
for num, (name, size) in enumerate(bufs):
    if name == "input_ids":       data = array('q', ids).tobytes()
    elif name == "position_ids":  data = array('q', range(seqlen)).tobytes()
    else:                         data = b"\x00" * size          # KV cache = zeros
    assert len(data) == size, f"{name}: {len(data)} != {size}"
    open(os.path.join(binddir, f"in_{num}.bin"), "wb").write(data)
    userBuffers.append({"num": num, "buffSize": size, "name": name, "inputFile": f"bind/in_{num}.bin"})

json.dump({"userBuffers": userBuffers}, open(os.path.join(args.qpc, "bindings.json"), "w"))
print(f"wrote {args.qpc}/bindings.json + {len(bufs)} files in bind/")
print("qaic-runner -t <this qpc> will now use YOUR input (delete bindings.json to revert to random).")
