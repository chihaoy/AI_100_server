#!/usr/bin/env python3
"""Rebuild a full-model graph directory whose non-expert weights lived in the deleted QEfficient export.
Usage: rebuild_full_graph.py <src model.onnx> <out dir> <shared weights dir>
- expert banks: repointed to oracle_padding/weights_fp16/weights.bin (offsets from its index.json, shapes checked)
- non-expert tensors: written as fp16 files under <out>/weights/<name> from the HF checkpoint (bf16 -> fp16, MatMul weights transposed
  to the [in, out] layout of torch.onnx), rotary sin/cos caches recomputed the QEfficient way (fp32 -> fp16, 40960 x 128)."""
import sys, os, json, numpy as np, torch, onnx
from safetensors import safe_open
O='/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding'
src, out, shared = sys.argv[1], sys.argv[2], sys.argv[3]; BANKS = sys.argv[4] if len(sys.argv) > 4 else f'{O}/weights_fp16'; os.makedirs(out, exist_ok=True); os.makedirs(shared, exist_ok=True)
O='/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding'; HF='/home/chihao/models/qwen3_30b_a3b/hf'
cfg=json.load(open(f'{HF}/config.json')); wm=json.load(open(f'{HF}/model.safetensors.index.json'))['weight_map']
banks=json.load(open(f'{BANKS}/index.json'))['tensors']; BANKNAME=os.path.basename(BANKS.rstrip('/'))
m=onnx.load(src, load_external_data=False); g=m.graph
cons={}
for n in g.node:
    for i in n.input: cons.setdefault(i, n.name)
handles={}
def hf(name):
    f=wm[name]
    if f not in handles: handles[f]=safe_open(f'{HF}/{f}', framework='pt')
    return handles[f].get_tensor(name)
def rotary():
    dim=cfg['head_dim']; base=cfg['rope_theta']; L=cfg['max_position_embeddings']
    inv=1.0/(base**(torch.arange(0, dim, 2, dtype=torch.int64).float()/dim)); t=torch.arange(L, dtype=torch.int64).type_as(inv)
    emb=torch.cat([torch.outer(t, inv)]*2, dim=-1); return emb.cos().to(torch.float16), emb.sin().to(torch.float16)
cos, sin = rotary(); written=0; repointed=0; total=0; regrouped=False; skipped=0
for t in g.initializer:
    ext={e.key:e.value for e in t.external_data}; loc=ext.get('location')
    if loc and loc.startswith('regrouped/'): regrouped=True; continue
    if not loc or not loc.startswith('weights/'): continue
    fname=loc[len('weights/'):]
    name=t.name; dims=list(t.dims); assert t.data_type==onnx.TensorProto.FLOAT16, (name, t.data_type)
    if name in banks:
        b=banks[name]; assert dims==b['shape'], (name, dims, b['shape'])
        t.ClearField('external_data')
        for k,v in (('location',f'{BANKNAME}/weights.bin'),('offset',str(b['offset'])),('length',str(b['length']))): e=t.external_data.add(); e.key=k; e.value=v
        repointed+=1; continue
    if name=='model.cos_cached': arr=cos
    elif name=='model.sin_cached': arr=sin
    elif name.startswith('onnx::MatMul'):
        node=cons[name]; assert node.endswith('/MatMul'), node; key=node[1:-len('/MatMul')].replace('/','.')+'.weight'
        arr=hf(key).to(torch.float16).t().contiguous()
    else: arr=hf(name).to(torch.float16)
    assert list(arr.shape)==dims, (name, list(arr.shape), dims)
    raw=arr.numpy().tobytes(); dst=f'{shared}/{fname}'
    if os.path.exists(dst) and os.path.getsize(dst)==len(raw): skipped+=1
    else: open(dst,'wb').write(raw); written+=1; total+=len(raw)
    assert len(raw)==int(np.prod(dims))*2
if repointed and not os.path.islink(f'{out}/{BANKNAME}'): os.symlink(os.path.relpath(BANKS, out), f'{out}/{BANKNAME}')
if regrouped and not os.path.islink(f'{out}/regrouped'): os.symlink(os.path.realpath(os.path.join(os.path.dirname(src), 'regrouped')), f'{out}/regrouped')
if not os.path.islink(f'{out}/weights'): os.symlink(os.path.relpath(shared, out), f'{out}/weights')
onnx.save(m, f'{out}/model.onnx'); _cwd=os.getcwd(); os.chdir(out); onnx.checker.check_model('model.onnx'); os.chdir(_cwd)
print(f'written {written} non-expert tensors ({total/2**30:.2f} GiB), reused {skipped}, repointed {repointed} expert banks, regrouped banks {regrouped}; model saved to {out}/model.onnx')
