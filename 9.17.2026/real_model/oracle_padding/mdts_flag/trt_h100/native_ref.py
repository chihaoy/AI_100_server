#!/usr/bin/env python3
"""Native optimized MoE reference (plan section 7) on the same inputs as the TensorRT arms: vLLM's fused_experts, a Triton
grouped-GEMM MoE (token alignment by expert, gate/up GEMM with fused SiLU-and-mul, down GEMM, weighted top-k sum), FP16.
Same weights (gate and up stacked as w13 [E, 2f, d], down [E, d, f]), hidden states, selected IDs and routing weights; the
router is excluded as in the TensorRT arms. Correctness against the untiered FP32 reference and the timing protocol of
run_trt.py (rounds, warm-up, CUDA events, eager and CUDA graph). Records the vLLM version and the kernels it launched.
Usage: native_ref.py --data data/L20 --T 512 --out results/L20_T512 [--rounds 6 --warmup 50 --iters 200]"""
import argparse, csv, json, os, numpy as np, torch
from moe_variants import reference

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--T', type=int, required=True); ap.add_argument('--out', required=True)
ap.add_argument('--rounds', type=int, default=6); ap.add_argument('--warmup', type=int, default=50); ap.add_argument('--iters', type=int, default=200)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T, dev = a.T, 'cuda'
import vllm
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
D = a.data
wg, wu, wd = (torch.from_numpy(np.load(f'{D}/w_{n}.npy')).to(dev) for n in ('gate', 'up', 'down'))
x = torch.from_numpy(np.load(f'{D}/x_T{T}.npy')).to(dev)
ids = torch.from_numpy(np.load(f'{D}/ids_T{T}.npy')).to(dev).int()
w = torch.from_numpy(np.load(f'{D}/w_T{T}.npy')).to(dev).half()
ref = reference(x, ids.long(), w, wg, wu, wd)
w13 = torch.cat([wg, wu], -1).transpose(1, 2).contiguous()           # [E, 2f, d]
w2 = wd.transpose(1, 2).contiguous()                                  # [E, d, f]
del wg, wu, wd; torch.cuda.empty_cache()
tw = w.float()
fn = lambda: fused_experts(x, w13, w2, tw, ids)
outs = [fn().clone() for _ in range(3)]; torch.cuda.synchronize()
y = outs[0].float(); rel = ((y - ref).norm() / ref.norm()).item(); mx = (y - ref).abs().max().item()
det = all(torch.equal(outs[0], o) for o in outs[1:])
print(f'vllm {vllm.__version__} fused_experts: rel L2 {rel:.2e}, max abs {mx:.2e}, deterministic {det}')
s = torch.cuda.Stream(); res = {}
with torch.cuda.stream(s):
    g = torch.cuda.CUDAGraph(); fn(); torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s): fn()
for mode, f in (('eager', fn), ('graph', g.replay)):
    meds = []
    for r in range(a.rounds):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
        with torch.cuda.stream(s):
            for _ in range(a.warmup): f()
            for st, en in ev: st.record(s); f(); en.record(s)
        s.synchronize(); meds.append(float(np.median([st.elapsed_time(en) * 1000 for st, en in ev])))
    res[mode] = meds; print(f'{mode}: median {np.median(meds):.1f} us (batch medians {min(meds):.1f}-{max(meds):.1f})')
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    fn(); torch.cuda.synchronize()
kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
json.dump(dict(vllm=vllm.__version__, torch=torch.__version__, kernels=kernels, relative_l2=rel, max_abs_error=mx, deterministic=det,
               latency_us={m: v for m, v in res.items()}), open(f'{a.out}/native_vllm.json', 'w'), indent=1)
with open(f'{a.out}/native_latency.csv', 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['backend', 'T', 'launch_mode', 'latency_us', 'batch_median_min_us', 'batch_median_max_us', 'relative_l2', 'max_abs_error'])
    for m, v in res.items(): wr.writerow([f'vllm {vllm.__version__} fused_experts', T, m, np.median(v), min(v), max(v), rel, mx])
print('kernels:', kernels)
