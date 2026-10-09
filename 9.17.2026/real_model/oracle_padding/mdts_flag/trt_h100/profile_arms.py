#!/usr/bin/env python3
"""Mechanism profile of the TensorRT arms (plan section 9, Figure 2), separate from the latency runs: CUPTI kernel traces
(torch.profiler) of eager executions, each kernel mapped to its engine layer through the engine inspector's tactic names,
and the layers classified by what they do: routing metadata, weight gather, token gather, gate/up GEMM, activation, down
GEMM (with the routing-weight multiply), combine. Reported per arm: summed kernel time per class (kernels on parallel
streams overlap, so the classes need not add up to the latency), the number of kernels, and bytes of the largest
intermediates. Usage: profile_arms.py --engines engines/L20 --data data/L20 --T 512 --tags ... --out results/L20_T512"""
import argparse, collections, json, os, re, numpy as np, torch, tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument('--engines', required=True); ap.add_argument('--data', required=True); ap.add_argument('--T', type=int, required=True)
ap.add_argument('--tags', required=True); ap.add_argument('--out', required=True); ap.add_argument('--iters', type=int, default=20)
a = ap.parse_args(); T, dev = a.T, 'cuda'
rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
TD = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
x = torch.from_numpy(np.load(f'{a.data}/x_T{T}.npy')).to(dev); ids = torch.from_numpy(np.load(f'{a.data}/ids_T{T}.npy')).to(dev).int()
w = torch.from_numpy(np.load(f'{a.data}/w_T{T}.npy')).to(dev).half()


def classify(layer):
    n = layer.get('Name', ''); tac = str(layer.get('TacticName', '')); t = layer.get('LayerType', '')
    dims = [o.get('Dimensions') or [] for o in layer.get('Outputs', [])]
    if any(len(d) == 3 and tuple(d[1:]) in ((2048, 1536), (768, 2048)) for d in dims): return 'weight gather'
    if t in ('gemm', 'fusion') or 'gemm' in tac:
        return 'gate/up GEMM' if any(d and d[-1] == 1536 for d in dims) else 'down GEMM'
    if 'Silu' in n: return 'activation'
    if re.search(r'Scat|Move', n) and any(len(d) == 3 and d[1] == T + 1 for d in dims): return 'combine'
    if re.search(r'Sum', n) and any(d[-1:] == [2048] and (len(d) == 2 or (len(d) >= 2 and d[-2] in (1, T))) for d in dims): return 'combine'
    if 'Gath' in n and any(len(d) == 3 and d[-1] == 2048 for d in dims): return 'token gather'
    if t in ('signal', 'wait'): return 'sync'
    return 'metadata'


res = {}
for tag in a.tags.split(','):
    eng = rt.deserialize_cuda_engine(open(f'{a.engines}/{tag}.engine', 'rb').read()); ctx = eng.create_execution_context()
    info = json.load(open(f'{a.engines}/{tag}.layers.json'))['Layers']
    tacmap = {}
    for l in info:
        if isinstance(l, dict) and l.get('TacticName'): tacmap[str(l['TacticName'])[:40]] = classify(l)
    outs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            outs[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=TD[eng.get_tensor_dtype(n)], device=dev); ctx.set_tensor_address(n, outs[n].data_ptr())
    for n, v in (('x', x), ('ids', ids), ('w', w)): ctx.set_tensor_address(n, v.data_ptr())
    s = torch.cuda.Stream()
    for _ in range(10): ctx.execute_async_v3(s.cuda_stream)
    s.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.iters): ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
    cls, unk, nk = collections.defaultdict(float), collections.defaultdict(float), 0
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA: continue
        dur = e.device_time / a.iters if hasattr(e, 'device_time') else e.cuda_time / a.iters
        key = next((v for k, v in tacmap.items() if e.name.startswith(k[:30])), None)
        if key is None and re.search(r'gemm|xmma|cutlass', e.name): key = 'GEMM (unmapped)'
        if key is None: unk[e.name[:60]] += dur; key = 'other'
        cls[key] += dur; nk += 1
    res[tag] = dict(us_per_call={k: round(v, 1) for k, v in sorted(cls.items(), key=lambda kv: -kv[1])}, kernels_per_call=nk / a.iters,
                    unmapped={k: round(v, 1) for k, v in sorted(unk.items(), key=lambda kv: -kv[1])[:6]})
    print(f'{tag:26s} {nk / a.iters:5.1f} kernels/call  ' + ', '.join(f'{k} {v:.0f}' for k, v in res[tag]['us_per_call'].items()))
    if unk: print('   unmapped:', res[tag]['unmapped'])
json.dump(res, open(f'{a.out}/profile_classes.json', 'w'), indent=1)
