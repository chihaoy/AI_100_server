#!/usr/bin/env python3
"""Final norm + lm_head on the CPU for a residual stream produced on the card (h_final_f32.bin [T,H] fp32), then compare
with the CPU fp32 reference logits (all positions) and with the naive QPC's last-token logits.
  python3 tools/moe_e2e_logits.py --h 9.17.2026/e2e/chain/h_final_f32.bin [--naive 9.17.2026/e2e/naive/logits_naive.bin]
"""
import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
from transformers import AutoTokenizer
HF = "/home/chihao/models/qwen3_30b_a3b/hf"; E = "/home/chihao/mllm/9.17.2026/e2e"
ap = argparse.ArgumentParser(); ap.add_argument("--h", required=True); ap.add_argument("--naive", default=f"{E}/naive/logits_naive.bin"); ap.add_argument("--name", default="path-2 chain"); a = ap.parse_args()
wmap = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
W = lambda k: safe_open(f"{HF}/{wmap[k]}", framework="pt").get_tensor(k).float()
h = torch.from_numpy(np.fromfile(a.h, np.float32).reshape(128, 2048))
hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * W("model.norm.weight"); logits = (hn @ W("lm_head.weight").T).numpy()
ref = np.load(f"{E}/ref/logits.npy"); href = np.load(f"{E}/ref/h_L48.npy")
tok = AutoTokenizer.from_pretrained(HF); ids = np.load(f"{E}/ref/input_ids.npy")
am, ar = logits.argmax(-1), ref.argmax(-1)
print(f"[{a.name}] final residual stream vs CPU ref: max|h| {np.abs(href).max():.1f}, max|diff| {np.abs(h.numpy()-href).max():.2f}, cos {float((h.numpy()*href).sum()/np.linalg.norm(h.numpy())/np.linalg.norm(href)):.6f}")
print(f"[{a.name}] argmax agreement with CPU ref over 128 positions: {(am==ar).sum()}/128;  top-5 of last position: {np.argsort(-logits[-1])[:5].tolist()} vs ref {np.argsort(-ref[-1])[:5].tolist()}")
print(f"[{a.name}] last-position logits corr vs CPU ref: {np.corrcoef(logits[-1], ref[-1])[0,1]:.4f}; predicted next token: {tok.decode([int(am[-1])])!r} (ref {tok.decode([int(ar[-1])])!r})")
if os.path.exists(a.naive):
    ln = np.fromfile(a.naive, np.float32)
    print(f"[{a.name}] vs naive QPC (last position): argmax equal {am[-1]==ln.argmax()}, corr {np.corrcoef(logits[-1], ln)[0,1]:.4f}, top-5 naive {np.argsort(-ln)[:5].tolist()};  naive vs CPU ref: argmax equal {ln.argmax()==ar[-1]}, corr {np.corrcoef(ln, ref[-1])[0,1]:.4f}, argmax agreement n/a (naive outputs last token only)")
mism = np.where(am != ar)[0]
if len(mism): print("  mismatching positions:", mism.tolist()[:20], " e.g.", [(int(t), tok.decode([int(am[t])]), tok.decode([int(ar[t])])) for t in mism[:5]])
np.save(a.h.replace(".bin", "_logits.npy"), logits)
