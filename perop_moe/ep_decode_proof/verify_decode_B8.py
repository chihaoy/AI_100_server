"""Feed 8 real tokens (batch 8, seq_len 1, position 0) to the decode-only EP QPC and compare logits with PyTorch."""
import os, sys, json, subprocess, numpy as np, torch
from array import array
S = os.path.dirname(os.path.abspath(__file__)); Q = open(f"{S}/Q").read().strip(); QPC = f"{Q}/qpc_decode_B8"
B, CTX, L, KVH, HD = 8, 64, 2, 2, 64
ids = [17, 250, 511, 777, 3, 999, 64, 128]
bufs = [("input_ids", array('q', ids).tobytes()), ("position_ids", array('q', [0]*B).tobytes())]
for l in range(L):
    for kv in ("key", "value"):
        bufs.append((f"past_{kv}.{l}", b"\x00" * (B * KVH * CTX * HD * 2)))
os.makedirs(f"{QPC}/bind", exist_ok=True); ub = []
for n, (name, data) in enumerate(bufs):
    open(f"{QPC}/bind/in_{n}.bin", "wb").write(data)
    ub.append({"num": n, "buffSize": len(data), "name": name, "inputFile": f"bind/in_{n}.bin"})
json.dump({"userBuffers": ub}, open(f"{QPC}/bindings.json", "w"))
out = f"{Q}/outdir_B8"; subprocess.run(["rm", "-rf", out])
r = subprocess.run(["qaic-runner", "-t", QPC, "-D", "0:1:2:3", "--num-iter", "1", "--write-output-dir", out], capture_output=True, text=True)
print("qaic-runner rc", r.returncode); print([l for l in r.stdout.splitlines() if "logits" in l or "Error" in l][:3])
f = [x for x in os.listdir(out) if x.startswith("logits")][0]
qpc_logits = np.fromfile(f"{out}/{f}", dtype=np.float32).reshape(B, -1)[:, :1024]
from transformers import Qwen3MoeForCausalLM
m = Qwen3MoeForCausalLM.from_pretrained(f"{S}/hf", torch_dtype=torch.float32).eval()
with torch.no_grad():
    ref = m(input_ids=torch.tensor(ids).view(B, 1), attention_mask=torch.ones(B, 1, dtype=torch.long)).logits[:, 0, :].numpy()
print("output file", f, "bytes", os.path.getsize(f"{out}/{f}"))
for b in range(B):
    c = np.corrcoef(qpc_logits[b], ref[b])[0, 1]
    print(f"token {ids[b]:4d}: corr={c:.4f}  top1 qpc={qpc_logits[b].argmax()} ref={ref[b].argmax()}  max|ref|={np.abs(ref[b]).max():.2f} max|err|={np.abs(qpc_logits[b]-ref[b]).max():.3f}")
# also: does the batch of 8 rows really route to different experts? (proves the EP path had >1 lane busy)
with torch.no_grad():
    h = m.model.embed_tokens(torch.tensor(ids).view(B, 1)); 
    x = m.model.layers[0].input_layernorm(h)  # not exact (attention skipped) but shows router spread
    rl = m.model.layers[0].mlp.gate(x.view(B, -1))
    rl = rl[0] if isinstance(rl, tuple) else rl
    print("layer0 router top-2 experts per token (pre-attn approx):", torch.topk(rl, 2, dim=-1).indices.tolist())
