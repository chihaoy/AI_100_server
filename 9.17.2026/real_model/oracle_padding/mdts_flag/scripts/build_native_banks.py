#!/usr/bin/env python3
"""Native-order expert banks from the sorted-layout bank file. For every layer, the six bank tensors (gate/up/down x stage 0/1)
are read from oracle_padding/weights_fp16/weights.bin (layout = sorted128_fp16 plan), inverse-permuted with that plan's per-layer
order (position p -> expert order[p], stage p//64, lane p%64) and written to <out>/weights.bin at the same offsets, plus index.json.
Usage: build_native_banks.py <out dir> [layers, e.g. 2 for a check]"""
import sys, os, json, numpy as np, onnx
O='/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding'; out=sys.argv[1]; only=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else list(range(48))
os.makedirs(out, exist_ok=True)
idx=json.load(open(f'{O}/weights_fp16/index.json')); banks=idx['tensors']; plan=json.load(open(f'{O}/sorted128_fp16_materialized_graph/plan.json'))['layers']
g=onnx.load(f'{O}/diagnostic_graph/model.onnx', load_external_data=False).graph
names={}   # layer -> {'MatMul': name, ..., 'MatMul_5': name}
for n in g.node:
    if n.op_type=='MatMul' and n.name.startswith('/model/layers.') and '/mlp/MatMul' in n.name:
        L=int(n.name.split('.')[1].split('/')[0]); k=n.name.split('/')[-1]; w=[i for i in n.input if i in banks]; names.setdefault(L,{})[k]=w[0]
total=sum(b['length'] for b in banks.values()); src=np.memmap(f'{O}/weights_fp16/weights.bin', np.float16, 'r')
mode='r+' if os.path.exists(f'{out}/weights.bin') else 'w+'; dst=np.memmap(f'{out}/weights.bin', np.float16, mode, shape=(total//2,))
for L in only:
    order=np.array(plan[str(L)]['order']); assert sorted(order.tolist())==list(range(128)); inv=np.argsort(order)   # inv[e] = position of expert e
    for kind,(k0,k1) in {'gate':('MatMul','MatMul_3'),'up':('MatMul_1','MatMul_4'),'down':('MatMul_2','MatMul_5')}.items():
        b0,b1=banks[names[L][k0]],banks[names[L][k1]]; shp=tuple(b0['shape']); n=int(np.prod(shp))
        s0=src[b0['offset']//2:b0['offset']//2+n].reshape(shp); s1=src[b1['offset']//2:b1['offset']//2+n].reshape(shp)
        sorted_all=np.concatenate([s0,s1],0)             # position p -> bank row p (stage p//64, lane p%64)
        native=sorted_all[inv]                            # expert e -> row at position inv[e]
        dst[b0['offset']//2:b0['offset']//2+n]=native[:64].reshape(-1); dst[b1['offset']//2:b1['offset']//2+n]=native[64:].reshape(-1)
    print('layer', L, 'done', flush=True)
dst.flush()
if len(only)==48:
    json.dump(dict(layout='native expert order (inverse of sorted128_fp16 plan)', source_index=f'{O}/weights_fp16/index.json', tensors=banks), open(f'{out}/index.json','w'))
    print('index written')
