#!/usr/bin/env python3
"""Build TensorRT engines from the exported ONNX variants (export_onnx.py) with identical settings for every arm:
strongly typed network (the graph's FP16/FP32/INT tensor types are kept as written, so no arm gets a different precision),
the same workspace limit, builder optimization level and a shared timing cache; detailed profiling verbosity so the engine
inspector can list layers, fusions, reformat/copy layers and the selected tactics. Writes <tag>.engine, <tag>.layers.json
(engine inspector) and <tag>.build.json (versions, build time, sizes). Usage: build_engines.py <onnx dir> <engine dir> [tag ...]"""
import sys, os, json, time, glob
import tensorrt as trt

ONNX, OUT = sys.argv[1], sys.argv[2]; TAGS = sys.argv[3:]
WORKSPACE = 8 << 30; OPT_LEVEL = 3
os.makedirs(OUT, exist_ok=True)
logger = trt.Logger(trt.Logger.WARNING)
cache_path = f'{OUT}/timing.cache'
for p in sorted(glob.glob(f'{ONNX}/*.onnx')):
    tag = os.path.basename(p)[:-5]
    if TAGS and tag not in TAGS: continue
    if os.path.exists(f'{OUT}/{tag}.engine'): print('exists', tag); continue
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(p):
        print('PARSE FAILED', tag, *[str(parser.get_error(i)) for i in range(parser.num_errors)], sep='\n  '); continue
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE)
    cfg.builder_optimization_level = OPT_LEVEL
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cache = cfg.create_timing_cache(open(cache_path, 'rb').read() if os.path.exists(cache_path) else b'')
    cfg.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.time(); ser = builder.build_serialized_network(net, cfg); dt = time.time() - t0
    if ser is None: print('BUILD FAILED', tag); continue
    open(f'{OUT}/{tag}.engine', 'wb').write(ser)
    open(cache_path, 'wb').write(cfg.get_timing_cache().serialize())
    eng = trt.Runtime(logger).deserialize_cuda_engine(ser)
    insp = eng.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    json.dump(info, open(f'{OUT}/{tag}.layers.json', 'w'), indent=1)
    layers = info.get('Layers', [])
    json.dump(dict(tag=tag, tensorrt=trt.__version__, build_s=dt, engine_bytes=ser.nbytes, n_layers=len(layers),
                   device_memory=eng.device_memory_size_v2 if hasattr(eng, 'device_memory_size_v2') else eng.device_memory_size,
                   workspace=WORKSPACE, opt_level=OPT_LEVEL, strongly_typed=True), open(f'{OUT}/{tag}.build.json', 'w'), indent=1)
    print(f"built {tag}: {dt:.0f} s, {ser.nbytes / 2**30:.2f} GiB, {len(layers)} layers")
