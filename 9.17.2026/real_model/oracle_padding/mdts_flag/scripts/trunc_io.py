#!/usr/bin/env python3
"""Emit a qaic-runner io.json for a (possibly retained-state) truncated-model QPC: input_ids/position_ids from the FP32 reference,
any exposed KV inputs as zero files, outputs as placeholders. Usage: trunc_io.py <qpc dir> <work dir> [<reference dir>]"""
import sys, os, json, numpy as np
sys.path.extend(['/opt/qti-aic/dev/lib/x86_64', '/opt/qti-aic/dev/python']); import qaicrt, QAicApi_pb2 as api
qpc, work = sys.argv[1], sys.argv[2]; os.makedirs(work, exist_ok=True)
REF = sys.argv[3] if len(sys.argv) > 3 else '/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad/kv/ref'
q = qaicrt.Qpc(qpc); st, data = q.getIoDescriptor(); d = api.IoDesc(); d.ParseFromString(bytes(data))
esize = {api.FLOAT_TYPE: 4, api.FLOAT_16_TYPE: 2, api.INT64_I_TYPE: 8, api.INT32_I_TYPE: 4, api.INT8_Q_TYPE: 1, api.UINT8_Q_TYPE: 1}
files = []
for b in d.selected_set.bindings:
    dims = list(b.dims); es = esize.get(b.type, 2); n = int(np.prod(dims)) * es
    if b.dir == api.BUFFER_IO_TYPE_INPUT:
        if b.name in ('input_ids', 'position_ids'): path = f'{REF}/{b.name}_i64.bin'
        else:
            path = f'{work}/{b.name}.bin'
            with open(path, 'wb') as f: f.write(b'\0' * n)
        files.append(dict(path=path, dims=dims, **{'elem-size': es, 'io-direction': 'in', 'map-to': b.name}))
    else:
        path = f'{work}/{b.name}_out.bin'
        with open(path, 'wb') as f: f.write(b'\0' * n)
        files.append(dict(path=path, dims=dims, **{'elem-size': es, 'io-direction': 'out', 'map-to': b.name}))
json.dump({'IO-files': [files]}, open(f'{work}/io.json', 'w'), indent=1)
print(f'{len(files)} IO files; types', sorted({b.type for b in d.selected_set.bindings}))
