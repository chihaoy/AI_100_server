#!/usr/bin/env python3
"""Export the TensorRT experiment's MoE-block variants to ONNX (opset 17), FP16 weights and activations, one graph per
(variant, T). Weights go to one external-data file per graph. Variants:
  S00/S01  static fixed-order uniform, dense / token-owned combine (no ranking, constant weights)
  G00/G01  ranked uniform (one tier of 128 lanes at T), dense / token-owned
  G10/G11  ranked tiers from tiers.json (--tiers), dense / token-owned
Usage: export_onnx.py --data data/L20 --T 512 --variants G00,G01,G10,G11,S01 --tiers 3t_32_32_64 --out onnx/L20
       export_onnx.py --selftest   (export small random graphs and check onnxruntime against PyTorch and the reference)"""
import argparse, json, os, numpy as np, torch
from moe_variants import MoEBlock, reference

SPEC = {'S00': ('uniform', 'dense', False), 'S01': ('uniform', 'token', False), 'G00': ('uniform', 'dense', True),
        'G01': ('uniform', 'token', True), 'G10': ('tiered', 'dense', True), 'G11': ('tiered', 'token', True),
        'U00': ('uniform-staged', 'dense', True), 'U01': ('uniform-staged', 'token', True)}
# U00/U01 (control): capacity T in the tiered arms' stage structure (same lanes per stage), separating the stage organization
# (and with it TensorRT's kernel choices for the weight gather) from the capacity reduction.


def build(variant, wgu, wd, T, tiers):
    cap, comb, ranked = SPEC[variant]
    tz = [(wgu.shape[0], T)] if cap == 'uniform' else [(l, T) for l, _ in tiers] if cap == 'uniform-staged' else tiers
    return MoEBlock(wgu, wd, T, tz, comb, ranked).eval()


def export(model, T, dtype, path, external=True):
    d, k = model.d, model.k; dev = model.wgu.device                 # tracing runs the model once: use the GPU when there is one
    args = (torch.zeros(T, d, dtype=dtype, device=dev), torch.zeros(T, k, dtype=torch.int32, device=dev), torch.zeros(T, k, dtype=dtype, device=dev))
    torch.onnx.export(model, args, path, input_names=['x', 'ids', 'w'], output_names=['y', 'ovf'], opset_version=17,
                      do_constant_folding=False, dynamo=False)
    if external:
        import onnx
        m = onnx.load(path)
        onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(path) + '.data',
                  size_threshold=1024)


def selftest():
    import onnxruntime as ort, tempfile
    torch.manual_seed(0); E, d, f, T, k = 128, 64, 32, 128, 8
    wg, wu, wd = torch.randn(E, d, f) / 8, torch.randn(E, d, f) / 8, torch.randn(E, f, d) / 6
    wgu = torch.cat([wg, wu], -1)
    tiers = [(32, 128), (32, 32), (64, 16)]
    cases = {'real-like': torch.stack([torch.randperm(E)[:k] for _ in range(T)]),
             'skewed (8 experts take everything)': torch.arange(k).repeat(T, 1),
             'balanced': (torch.arange(T).unsqueeze(1) * k + torch.arange(k)) % E}
    x = torch.randn(T, d); wts = torch.softmax(torch.randn(T, k), -1)
    tmp = tempfile.mkdtemp()
    for v in SPEC:
        m = build(v, wgu, wd, T, tiers); p = f'{tmp}/{v}.onnx'; export(m, T, torch.float32, p, external=False)
        sess = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
        for cname, ids in cases.items():
            ref = reference(x, ids, wts, wg, wu, wd)
            y, ovf = sess.run(None, {'x': x.numpy(), 'ids': ids.int().numpy(), 'w': wts.numpy()})
            with torch.no_grad(): yt, ot = m(x, ids.int(), wts)
            rel = np.linalg.norm(y - ref.numpy()) / np.linalg.norm(ref.numpy())
            print(f'{v} {cname:36s} ort vs ref {rel:.1e}  ort vs torch {np.abs(y - yt.numpy()).max():.1e}  ovf {ovf[0]:.0f}/{ot.item():.0f}')
    # forced overflow: small tiers (8 lanes at T, 24 at 16, 96 at 4) and 40 experts with ~26 tokens each, so ranks 9-32 overflow;
    # the flag must be raised, and the caller's rerun on the uniform graph must reproduce the reference
    small = [(8, T), (24, 16), (96, 4)]; ids = (torch.arange(T).unsqueeze(1) * k + torch.arange(k)) % 40
    ref = reference(x, ids, wts, wg, wu, wd)
    for v in ('G10', 'G11'):
        m = build(v, wgu, wd, T, small)
        with torch.no_grad(): y, o = m(x, ids.int(), wts); yu, ou = build(v.replace('1', '0', 1), wgu, wd, T, small)(x, ids.int(), wts)
        print(f'forced overflow {v}: flag {o.item():.0f}, tiered output off by {((y - ref).norm() / ref.norm()).item():.2f}; '
              f'uniform rerun flag {ou.item():.0f}, rel L2 {((yu - ref).norm() / ref.norm()).item():.1e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--data'); ap.add_argument('--T', type=int)
    ap.add_argument('--variants', default='S01,G00,G01,G10,G11'); ap.add_argument('--tiers', default='3t_32_32_64')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.selftest: selftest(); raise SystemExit
    os.makedirs(a.out, exist_ok=True)
    wg, wu, wd = (torch.from_numpy(np.load(f'{a.data}/w_{n}.npy')) for n in ('gate', 'up', 'down'))
    wgu = torch.cat([wg, wu], -1); del wg, wu
    st = json.load(open(os.path.join(os.path.dirname(a.data.rstrip('/')), 'tiers.json')))[str(a.T)]['structures'][a.tiers]
    tiers = list(zip(st['lanes'], st['caps']))
    for v in a.variants.split(','):
        tag = f'{v}_T{a.T}' + (f'_{a.tiers}' if SPEC[v][0] in ('tiered', 'uniform-staged') else '')
        m = build(v, wgu, wd, a.T, tiers).to('cuda' if torch.cuda.is_available() else 'cpu'); p = f'{a.out}/{tag}.onnx'
        export(m, a.T, torch.float16, p)
        json.dump(dict(variant=v, T=a.T, tiers=m.tiers, spec=SPEC[v], rows=m.R),
                  open(f'{a.out}/{tag}.json', 'w'))
        print('exported', p, f'{os.path.getsize(p + ".data") / 2**30:.2f} GiB weights')
