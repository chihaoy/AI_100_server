#!/usr/bin/env python3
"""Per-layer check of the end-to-end path-2 QPCs on the card: feed the CPU-reference residual stream h_L (x.bin) to the
layer's hot QPC (cards 0,1) and cold QPC (cards 2,3), add the outputs on the host, compare with the reference h_{L+1}.
Also compares each QPC's own output with the exporter's torch fp16 forward (y_torch.npy) to separate graph errors
from MXFP6 quantisation.   python3 tools/moe_e2e_verify_layers.py [--layers 0-47] [--e2e DIR]
"""
import argparse, os, subprocess, sys, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="0-47"); ap.add_argument("--e2e", default="/home/chihao/mllm/9.17.2026/e2e"); a = ap.parse_args()
lo, hi = [int(v) for v in a.layers.split("-")]; E = a.e2e; R = f"{E}/ref"; QR = "/opt/qti-aic/exec/qaic-runner"
def run(d, devs):
    od = f"{d}/outdir"; subprocess.run(["rm", "-rf", od]); os.makedirs(od)
    r = subprocess.run([QR, "-t", f"{d}/qpc", "-D", devs, "-i", f"{d}/x.bin", "--write-output-dir", od, "--num-iter", "1"], capture_output=True, text=True)
    fs = [f for f in os.listdir(od) if f.endswith(".bin")]
    if not fs: print(r.stdout[-500:], r.stderr[-500:]); sys.exit(f"{d}: no output")
    return np.fromfile(f"{od}/{fs[0]}", np.float16).astype(np.float32).reshape(128, 2048)
print(f"{'L':>2} {'|h_ref|max':>10} {'max|err| (hot+cold vs ref)':>26} {'rel-to-max':>10} {'median rel':>10} {'cos':>8} | hot vs torch  cold vs torch")
for l in range(lo, hi + 1):
    d = f"{E}/L{l}"
    if not (os.path.exists(f"{d}/hot/qpc/programqpc.bin") and os.path.exists(f"{d}/cold/qpc/programqpc.bin")): print(f"{l:2d} (not built yet)"); continue
    yh, yc = run(f"{d}/hot", "0:1"), run(f"{d}/cold", "2:3"); y = yh + yc
    ref = np.load(f"{R}/h_L{l+1}.npy").astype(np.float32); th, tc = np.load(f"{d}/hot/y_torch.npy"), np.load(f"{d}/cold/y_torch.npy")
    err = np.abs(y - ref); scale = np.abs(ref).max(); rel = err / (np.abs(ref) + 1e-2)
    cos = float((y * ref).sum() / (np.linalg.norm(y) * np.linalg.norm(ref)))
    eh = np.abs(yh - th).max() / max(np.abs(th).max(), 1e-6); ec = np.abs(yc - tc).max() / max(np.abs(tc).max(), 1e-6)
    # drop detector: a dropped (token, expert) removes that expert's whole contribution from the token's layer delta;
    # compare each token's error with the norm of its reference layer delta (attention + MoE contribution)
    hl = np.load(f"{R}/h_L{l}.npy").astype(np.float32); delta = np.linalg.norm(ref - hl, axis=1); et = np.linalg.norm(y - ref, axis=1)
    rt = et / np.maximum(delta, 1e-3); sus = int((rt > 0.15).sum())
    print(f"{l:2d} {scale:10.2f} {err.max():26.4f} {err.max()/scale:10.4f} {np.median(rel):10.4f} {cos:8.5f} | {eh:.4f}        {ec:.4f}   | per-token err/|delta|: median {np.median(rt):.3f} max {rt.max():.3f}  tokens>0.15: {sus}")
