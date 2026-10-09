#!/usr/bin/env python3
"""Correctness and latency of TensorRT MoE-block engines on one GPU (plan sections 8-9).

Correctness, before any timing: each engine's y against the untiered FP32 reference (relative L2, max abs error), the
overflow flag, and bitwise determinism over three runs. Threshold (fixed before timing): relative L2 <= 2e-3, i.e. FP16
rounding of the intermediates with FP32 GEMM accumulation. TensorRT 11 picks FP16-accumulating GEMM kernels for some fused
GEMMs (no builder control), which emulation shows gives 1.05e-3 (down GEMM) to 2.26e-3 (both GEMMs); engines within 1e-2
and without overflow are therefore timed, and each row records whether it met the 2e-3 threshold.
Timing: inputs, outputs and weights resident; one engine at a time; rounds alternate the engine order (forward, reverse);
each (round, engine) runs warm-up iterations, then timed iterations with CUDA events on the engine's stream around each
execute (eager) or each CUDA-graph replay (graph). Reported: median of the batch medians, and the range of batch medians.
Usage: run_trt.py --engines engines/L20 --onnx onnx/L20 --data data/L20 --T 512 --tags G00_T512,G01_T512,... --out results/L20_T512
       [--rounds 6 --warmup 50 --iters 200 --modes eager,graph --trace real|balanced|skewed|overflow]"""
import argparse, json, os, subprocess, time, numpy as np, torch, tensorrt as trt
from moe_variants import reference

ap = argparse.ArgumentParser()
ap.add_argument('--engines', required=True); ap.add_argument('--onnx', required=True); ap.add_argument('--data', required=True); ap.add_argument('--T', type=int, required=True)
ap.add_argument('--tags', required=True); ap.add_argument('--out', required=True); ap.add_argument('--rounds', type=int, default=6)
ap.add_argument('--warmup', type=int, default=50); ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--modes', default='eager,graph'); ap.add_argument('--trace', default='real'); ap.add_argument('--tol', type=float, default=2e-3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T, k, E, dev = a.T, 8, 128, 'cuda'
logger = trt.Logger(trt.Logger.WARNING); rt = trt.Runtime(logger)
TD = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}


def trace(kind):
    D = a.data
    x = torch.from_numpy(np.load(f'{D}/x_T{T}.npy')); w = torch.from_numpy(np.load(f'{D}/w_T{T}.npy'))
    ids = torch.from_numpy(np.load(f'{D}/ids_T{T}.npy')).long()
    if kind == 'balanced': ids = (torch.arange(T).unsqueeze(1) * k + torch.arange(k)) % E           # T/16 tokens per expert
    elif kind == 'skewed': ids = torch.arange(k).repeat(T, 1)                                        # 8 experts take all, 120 empty
    elif kind == 'overflow': ids = (torch.arange(T).unsqueeze(1) * k + torch.arange(k)) % 40         # 40 experts x 8T/40 tokens
    return x, ids.int(), w


class Engine:
    def __init__(self, tag):
        self.tag = tag
        self.engine = rt.deserialize_cuda_engine(open(f'{a.engines}/{tag}.engine', 'rb').read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]

    def bind(self, inputs):
        self.inputs, self.outs = inputs, {}
        for n in self.names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.outs[n] = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=TD[self.engine.get_tensor_dtype(n)], device=dev)
                self.ctx.set_tensor_address(n, self.outs[n].data_ptr())
            else:
                self.ctx.set_tensor_address(n, inputs[n].data_ptr())

    def run(self):
        assert self.ctx.execute_async_v3(self.stream.cuda_stream)

    def capture(self):
        self.run(); self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.stream):
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.run()
        self.stream.synchronize()

    def time(self, mode, warmup, iters):
        fn = self.graph.replay if mode == 'graph' else self.run
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        with torch.cuda.stream(self.stream):
            for _ in range(warmup): fn()
            for s, e in ev: s.record(self.stream); fn(); e.record(self.stream)
        self.stream.synchronize()
        return np.array([s.elapsed_time(e) * 1000 for s, e in ev])        # microseconds


def env():
    q = 'name,pci.bus_id,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.mem,mig.mode.current,clocks_throttle_reasons.active'
    smi = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    return dict(nvidia_smi=smi, torch=torch.__version__, cuda=torch.version.cuda, tensorrt=trt.__version__,
                device=torch.cuda.get_device_name(), host=os.uname().nodename, time=time.strftime('%F %T'))


x, ids, w = trace(a.trace)
wg, wu, wd = (torch.from_numpy(np.load(f'{a.data}/w_{n}.npy')).to(dev) for n in ('gate', 'up', 'down'))
inputs = dict(x=x.half().to(dev), ids=ids.to(dev), w=w.half().to(dev))
ref = reference(inputs['x'], inputs['ids'].long(), inputs['w'], wg, wu, wd)        # FP32, from the FP16 inputs
del wg, wu, wd; torch.cuda.empty_cache()
E_ = env(); json.dump(E_, open(f'{a.out}/env.json', 'w'), indent=1); print(E_)
engines, rows, check, allg = [], [], {}, {}
for tag in a.tags.split(','):
    eg = Engine(tag); eg.bind(inputs)
    outs = []
    for _ in range(3):
        eg.run(); eg.stream.synchronize(); outs.append({n: t.clone() for n, t in eg.outs.items()})
    y = outs[0]['y'].float(); rel = ((y - ref).norm() / ref.norm()).item(); mx = (y - ref).abs().max().item()
    det = all(torch.equal(outs[0][n], o[n]) for o in outs[1:] for n in o)
    ovf = float(outs[0]['ovf'].float().max())
    status = 'overflow flagged' if ovf > 0 else 'within threshold' if rel <= a.tol else 'ABOVE THRESHOLD' if rel <= 1e-2 else 'WRONG'
    check[tag] = dict(relative_l2=rel, max_abs_error=mx, deterministic=det, overflow=ovf, within_threshold=bool(rel <= a.tol), status=status)
    print(f'{tag:28s} rel L2 {rel:.2e}  max abs {mx:.2e}  deterministic {det}  overflow {ovf:.0f}  {status}')
    allg[tag] = eg
    if rel <= 1e-2 and ovf == 0: engines.append(eg)      # timed; the threshold status is reported with the latency


def host_path(eg, partner, iters=200):
    """Host-observed latency of one chunk: run, copy the overflow flag to the host, and on overflow rerun on the uniform
    engine of the same combine (the recovery of plan section 8). Returns (latencies us, overflows, output rel L2)."""
    flag = torch.empty(1, dtype=eg.outs['ovf'].dtype, pin_memory=True); ts, n = [], 0
    for _ in range(iters + 20):
        t0 = time.perf_counter(); eg.run()
        with torch.cuda.stream(eg.stream): flag.copy_(eg.outs['ovf'], non_blocking=True)
        eg.stream.synchronize(); out = eg.outs['y']
        if float(flag[0]) > 0: partner.run(); partner.stream.synchronize(); out = partner.outs['y']; n += 1
        ts.append((time.perf_counter() - t0) * 1e6)
    y = out.float(); return np.array(ts[20:]), n, ((y - ref).norm() / ref.norm()).item()


for tag, eg in allg.items():                                                         # fast path + check, and recovery
    v = tag.split('_')[0]
    if v not in ('G10', 'G11'): continue
    ptag = f"G0{v[2]}_T{T}"
    if ptag not in allg: continue
    plain = []
    for _ in range(220): t0 = time.perf_counter(); eg.run(); eg.stream.synchronize(); plain.append((time.perf_counter() - t0) * 1e6)
    ts, n, rel = host_path(eg, allg[ptag])
    check[tag]['host_us'] = dict(run_and_sync=float(np.median(plain[20:])), with_check=float(np.median(ts)), overflows=n,
                                 recovered_relative_l2=rel, partner=ptag)
    print(f'{tag:28s} host-observed: run+sync {np.median(plain[20:]):.1f} us, with overflow check {np.median(ts):.1f} us, '
          f'overflow reruns {n}/{len(ts) + 20}, final output rel L2 {rel:.2e}')
json.dump(check, open(f'{a.out}/check_{a.trace}.json', 'w'), indent=1)
if a.trace != 'real': raise SystemExit                                               # stress traces: correctness only
lat = {(eg.tag, m): [] for eg in engines for m in a.modes.split(',')}
for m in a.modes.split(','):
    if m == 'graph':
        for eg in engines:
            try: eg.capture()
            except Exception as ex: print('capture failed', eg.tag, ex); lat.pop((eg.tag, m), None)
    for r in range(a.rounds):
        order = engines if r % 2 == 0 else engines[::-1]
        for eg in order:
            if (eg.tag, m) not in lat: continue
            t = eg.time(m, a.warmup, a.iters)
            lat[(eg.tag, m)].append(dict(median=float(np.median(t)), p10=float(np.percentile(t, 10)), p90=float(np.percentile(t, 90))))
        print(f'round {r} {m}: ' + ', '.join(f'{eg.tag} {lat[(eg.tag, m)][-1]["median"]:.1f}' for eg in engines if (eg.tag, m) in lat))
for (tag, m), v in lat.items():
    meds = [b['median'] for b in v]; bj = f'{a.engines}/{tag}.build.json'; meta = json.load(open(bj)) if os.path.exists(bj) else {}
    info = json.load(open(f'{a.onnx}/{tag}.json')) if os.path.exists(f'{a.onnx}/{tag}.json') else {}
    rows.append(dict(gpu=E_['device'], software_versions=f'trt {E_["tensorrt"]}; torch {E_["torch"]}; cuda {E_["cuda"]}', backend='tensorrt',
                     layer=os.path.basename(a.data.rstrip('/')), trace_id=a.trace, T=T, E=E, k=k, dtype='fp16', variant=tag.split('_')[0],
                     tier_config=json.dumps(info.get('tiers')), launch_mode=m, latency_us=float(np.median(meds)),
                     batch_median_min_us=float(min(meds)), batch_median_max_us=float(max(meds)), rounds=len(meds), iters=a.iters,
                     metadata_us='', gemm_us='', combine_us='', provisioned_rows=info.get('rows', ''),
                     workspace_bytes=meta.get('device_memory', ''), observed_dram_bytes='', overflow=check[tag]['overflow'],
                     relative_l2=check[tag]['relative_l2'], max_abs_error=check[tag]['max_abs_error'],
                     within_threshold=check[tag]['within_threshold'], tag=tag))
import csv
with open(f'{a.out}/latency.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
for r_ in rows: print(f'{r_["tag"]:28s} {r_["launch_mode"]:6s} {r_["latency_us"]:9.1f} us  (batch medians {r_["batch_median_min_us"]:.1f}-{r_["batch_median_max_us"]:.1f})')
