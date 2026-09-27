#!/usr/bin/env python3
"""Per-prompt replay inputs for the dynamic-split graph: [T, 2048+128] fp16 = layer-2 MoE input + routing weights in native
expert order; expected counts in sorted (position) order; FP32 reference of the MoE output as fp16. Usage: make_inputs_dyn.py <routing dir> <out dir> <T>"""
import sys, os, glob, subprocess, numpy as np
d, out, T = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f'{d}/T{T}_*.npz')):
    name = os.path.basename(f)[:-4]; z = np.load(f); idx = z['routing_idx'][2]; w = z['routing_w'][2]
    routes = np.zeros((T, 128), np.float16)
    for t in range(T): routes[t, idx[t]] = w[t].astype(np.float16)
    xin = np.zeros((T, 2176), np.float16); xin[:, :2048] = z['x2_L2']; xin[:, 2048:] = routes; xin.tofile(f'{out}/{name}_input_f16.bin')
    counts = np.bincount(idx.ravel(), minlength=128); np.sort(counts)[::-1].astype(np.int32).tofile(f'{out}/{name}_counts_sorted_i32.bin'); counts.astype(np.int32).tofile(f'{out}/{name}_counts_native_i32.bin')
    if not os.path.exists(f'{out}/{name}_y_ref_f16.bin'):
        subprocess.run(['/home/chihao/qeff-venv/bin/python', os.path.dirname(os.path.abspath(__file__)) + '/ref_moe_l2.py', f, f'{out}/{name}_y_ref_f32.bin'], check=True, capture_output=True)
        np.fromfile(f'{out}/{name}_y_ref_f32.bin', np.float32).astype(np.float16).tofile(f'{out}/{name}_y_ref_f16.bin')
    print(name, 'max count', counts.max(), '65th', np.sort(counts)[::-1][64], 'active', (counts > 0).sum(), flush=True)
