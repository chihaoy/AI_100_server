#!/usr/bin/env python3
"""FP32 CPU reference of the layer-2 MoE block output for a collected prompt (x2_L2 + routing). Usage: ref_moe_l2.py <npz> <out.bin>"""
import sys, os, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools'); import moe_layer0_router_oncard as R
from safetensors import safe_open
z = np.load(sys.argv[1]); x2 = torch.from_numpy(z['x2_L2'].astype(np.float32)); idx = torch.from_numpy(z['routing_idx'][2].astype(np.int64)); w = torch.from_numpy(z['routing_w'][2])
wmap = json.load(open(os.path.join(R.HF, 'model.safetensors.index.json')))['weight_map']; _open = {}
def W(key):
    f = wmap[key]
    if f not in _open: _open[f] = safe_open(os.path.join(R.HF, f), framework='pt')
    return _open[f].get_tensor(key).float()
y = torch.zeros_like(x2); p = 'model.layers.2.'
with torch.no_grad():
    for e in range(128):
        sel = (idx == e); rows = sel.any(1).nonzero().squeeze(1)
        if len(rows) == 0: continue
        we = (w * sel)[rows].sum(1, keepdim=True); xs = x2[rows]
        g = xs @ W(p + f'mlp.experts.{e}.gate_proj.weight').T; u = xs @ W(p + f'mlp.experts.{e}.up_proj.weight').T
        y[rows] += we * ((F.silu(g) * u) @ W(p + f'mlp.experts.{e}.down_proj.weight').T)
y.numpy().astype(np.float32).tofile(sys.argv[2]); print(sys.argv[2], 'written', tuple(y.shape))
