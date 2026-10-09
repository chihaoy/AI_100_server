#!/usr/bin/env python3
"""Export one Qwen3-30B-A3B MoE layer and its routing traces for the TensorRT/H100 experiments.
Weights (BF16 in the checkpoint) are written as FP16: w_gate [E, d, f], w_up [E, d, f], w_down [E, f, d] (x @ W layout) and
the router w_router [E, d]. Traces come from the FP32 CPU references of the AI100 study (held-out chunks at T = 128/256/512):
x = the layer's MoE input (x2_L*), ids = top-8 experts in descending router probability, w = renormalized top-8 probabilities
(norm_topk_prob), recomputed here in FP32 and checked against the captured routing.
Usage: export_layer.py --layer 20 --out data/L20"""
import argparse, json, os, numpy as np, torch
from safetensors import safe_open
HF = '/home/chihao/models/qwen3_30b_a3b/hf'
REF = {128: 'ref', 256: 'ref256', 512: 'ref512'}
FM = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'full_model')
ap = argparse.ArgumentParser(); ap.add_argument('--layer', type=int, default=20); ap.add_argument('--out', required=True)
ap.add_argument('--no-weights', action='store_true'); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); L = a.layer
wm = json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
def load(name):
    with safe_open(f'{HF}/{wm[name]}', framework='pt') as f: return f.get_tensor(name)
p = f'model.layers.{L}.mlp'
router = load(f'{p}.gate.weight').float()                                   # [E, d]
E, d = router.shape
if not a.no_weights:
    for kind, tr in (('gate', True), ('up', True), ('down', True)):
        W = torch.stack([load(f'{p}.experts.{e}.{kind}_proj.weight').t().contiguous() for e in range(E)])
        np.save(f'{a.out}/w_{kind}.npy', W.to(torch.float16).numpy()); print(kind, tuple(W.shape))
    np.save(f'{a.out}/w_router.npy', router.to(torch.float16).numpy())
meta = dict(layer=L, E=E, d=d, k=8, traces={})
for T, r in REF.items():
    x = torch.from_numpy(np.load(f'{FM}/{r}/x2_L{L}.npy')).float()            # [T, d]
    cap = np.load(f'{FM}/{r}/routing.npy')[L].astype(np.int64)                  # [T, 8]
    prob = torch.softmax(x @ router.t(), dim=-1)
    w, ids = torch.topk(prob, 8, dim=-1); w = w / w.sum(-1, keepdim=True)
    same = np.mean([set(cap[t]) == set(ids[t].tolist()) for t in range(T)])
    counts = np.bincount(ids.numpy().ravel(), minlength=E)
    np.save(f'{a.out}/x_T{T}.npy', x.to(torch.float16).numpy()); np.save(f'{a.out}/x32_T{T}.npy', x.numpy())
    np.save(f'{a.out}/ids_T{T}.npy', ids.numpy().astype(np.int32)); np.save(f'{a.out}/w_T{T}.npy', w.numpy().astype(np.float32))
    meta['traces'][T] = dict(source=r, captured_topk_match=float(same), max_count=int(counts.max()), empty_experts=int((counts == 0).sum()))
    print(f'T={T}: top-8 sets match the capture for {100 * same:.1f}% of tokens; largest expert count {counts.max()}, empty experts {(counts == 0).sum()}')
json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
