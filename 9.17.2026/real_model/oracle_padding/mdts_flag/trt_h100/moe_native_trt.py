"""Probe TensorRT's native IMoELayer (add_moe) on this GPU with the layer-20 weights: does it build, is it correct, how fast?"""
import sys, numpy as np, torch, tensorrt as trt
sys.path.insert(0, '.')
from moe_variants import reference
T = int(sys.argv[1]) if len(sys.argv) > 1 else 512; D = 'data/L20'; dev = 'cuda'
lg = trt.Logger(trt.Logger.WARNING); rt = trt.Runtime(lg)
wg, wu, wd = (np.load(f'{D}/w_{n}.npy') for n in ('gate', 'up', 'down'))      # [E, d, f], [E, d, f], [E, f, d]
x = torch.from_numpy(np.load(f'{D}/x_T{T}.npy')).to(dev); ids = torch.from_numpy(np.load(f'{D}/ids_T{T}.npy')).to(dev).int()
w = torch.from_numpy(np.load(f'{D}/w_T{T}.npy')).to(dev).half()
ref = reference(x, ids.long(), w, *(torch.from_numpy(a).to(dev) for a in (wg, wu, wd)))
for layout in ('in_out', 'out_in'):
    g, u, dn = (wg, wu, wd) if layout == 'in_out' else (wg.transpose(0, 2, 1).copy(), wu.transpose(0, 2, 1).copy(), wd.transpose(0, 2, 1).copy())
    b = trt.Builder(lg); net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    xi = net.add_input('x', trt.float16, (1, T, 2048)); ii = net.add_input('ids', trt.int32, (1, T, 8)); wi = net.add_input('w', trt.float16, (1, T, 8))
    cg, cu, cd = (net.add_constant(a.shape, trt.Weights(a)).get_output(0) for a in (g, u, dn))
    try:
        moe = net.add_moe(xi, ii, wi)
        if moe is None: print(layout, 'add_moe returned None'); continue
        moe.set_gated_weights(cg, cu, cd, trt.MoEActType.SILU)
        net.mark_output(moe.get_output(0)); moe.get_output(0).name = 'y'
        cfg = b.create_builder_config(); cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        ser = b.build_serialized_network(net, cfg)
    except Exception as ex:
        print(layout, 'exception:', str(ex)[:300]); continue
    if ser is None: print(layout, 'build failed (see TensorRT errors above)'); continue
    eng = rt.deserialize_cuda_engine(ser); ctx = eng.create_execution_context()
    y = torch.empty(1, T, 2048, dtype=torch.float16, device=dev); s = torch.cuda.Stream()
    for n, t in (('x', x), ('ids', ids), ('w', w), ('y', y)): ctx.set_tensor_address(n, t.data_ptr())
    ctx.execute_async_v3(s.cuda_stream); s.synchronize()
    rel = ((y[0].float() - ref).norm() / ref.norm()).item()
    for _ in range(20): ctx.execute_async_v3(s.cuda_stream)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); e0.record(s)
    for _ in range(200): ctx.execute_async_v3(s.cuda_stream)
    e1.record(s); s.synchronize()
    import json
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    kern = [str(l.get('TacticName', l.get('Name', '')))[:70] for l in info['Layers'] if isinstance(l, dict)]
    print(f'{layout}: built, rel L2 vs FP32 reference {rel:.2e}, {e0.elapsed_time(e1) / 200 * 1000:.1f} us per call, layers {kern}')
