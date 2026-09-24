#!/usr/bin/env python3
"""Fixed-token-count MoE capacity specialization: export, compile, run and audit.

Synthetic weights, real router/top-k/packing/expert/reduction graph. Two int32
row vectors [C0], [C1] control canvas widths. A zero-valued tag [P] uniquely
identifies each compiled profile by shape. A joint [C0,C1] selector is an
alternative; the ambiguous separate-input case is retained to reproduce failure.
No runtime expert permutation, KV cache, attention or trained model weights.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
import torch
import torch.nn.functional as F

from moe_tier_bench_export import (MoELayerEP, CtxGatherFunc3DGeneralized,
                                 CtxScatterFunc3DGeneralized)
from QEfficient.base.onnx_transforms import CustomOpTransform

PROFILES = [(128, 128), (64, 32), (32, 64), (32, 32)]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def command(args, log):
    print('RUN', ' '.join(map(str, args)), flush=True)
    with log.open('w') as f:
        f.write(repr(list(map(str, args))) + '\n')
        f.flush()
        subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True)


class CapacityMoE(MoELayerEP):
    def cap(self, tier, s):
        return self.rows[s].shape[0]

    def forward(self, x, rows0, rows1):
        self.rows = (rows0, rows1)
        counts = (self.route(x) > 0).sum(0).to(torch.int32)
        return super().forward(x), counts

    def stage_update(self, x, T2Ei, gi, s, rw_unsq, expert_out, cap, x_unsq=None):
        # Match the existing expert path, retaining row values as actual inputs.
        n, T = T2Ei.shape
        midx = self.matched_idx(T2Ei, cap)
        valid_rows = T2Ei.to(torch.int32).sum(1, keepdim=True)
        x_exp = (x.unsqueeze(0) if x_unsq is None else x_unsq).expand(n, -1, -1)
        x_chunk = CtxGatherFunc3DGeneralized.apply(x_exp, midx)
        gate = x_chunk @ self.Wg[gi][:, s]
        up = x_chunk @ self.Wu[gi][:, s]
        down = (up * F.silu(gate)) @ self.Wd[gi][:, s]
        rw_chunk = CtxGatherFunc3DGeneralized.apply(rw_unsq, midx)
        old = CtxGatherFunc3DGeneralized.apply(expert_out, midx)
        upd = old + down * rw_chunk
        # midx has exactly cap columns; valid_rows need not be clipped here.
        upd = torch.where((self.rows[s].unsqueeze(0) < valid_rows).unsqueeze(-1),
                          upd, torch.zeros_like(upd))
        return CtxScatterFunc3DGeneralized.apply(expert_out, midx, upd)


class JointCapacityMoE(CapacityMoE):
    def forward(self, x, grid):
        return super().forward(x, grid[:, 0].to(torch.int32), grid[0, :].to(torch.int32))


class TaggedCapacityMoE(CapacityMoE):
    def forward(self, x, rows0, rows1, tag):
        # tag is a zero vector. Its length uniquely identifies the profile;
        # its values participate in masking so the input remains in the graph.
        return super().forward(x, rows0 + tag.sum().to(torch.int32), rows1)


def reference(model, x):
    """Independent FP32 expert loop, using the model's FP16 router decisions."""
    with torch.no_grad():
        rw = model.route(x)
        out = torch.zeros_like(x, dtype=torch.float32)
        for e in range(model.E):
            ids = torch.nonzero(rw[:, e] > 0).flatten()
            if ids.numel() == 0:
                continue
            lane, stage = e % model.P, e // model.P
            wg, wu, wd = (w.float() for w in model.weights(0, lane, stage))
            z = x[ids].float()
            out[ids] += ((F.silu(z @ wg) * (z @ wu)) @ wd) * rw[ids, e, None].float()
        return out.numpy(), (rw > 0).sum(0).numpy().astype(np.int32)


def export(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cls = {'joint':JointCapacityMoE,'separate':CapacityMoE,'tagged':TaggedCapacityMoE}[a.selector]
    model = cls(T=128, H=a.hidden, I=a.intermediate, K=8, S=2,
                       groups=[(128, 64)], num_devices=a.devices, seed=1701).eval()
    x = torch.randn(128, a.hidden, generator=torch.Generator().manual_seed(1702)).half()
    if a.routing == 'margin':
        if a.hidden < 128:
            raise ValueError('Margin-controlled router requires H >= 128')
        # One-hot token identity selects a predetermined, uneven set of top-8
        # experts. A large 8th/9th logit margin isolates capacity correctness
        # from CPU/device rounding differences in top-k membership.
        x[:, :128] = torch.eye(128, dtype=torch.float16)
        scores = torch.randn(128,128,generator=torch.Generator().manual_seed(1703))
        selected = scores.topk(8,dim=0).indices
        with torch.no_grad():
            model.router.zero_()
            model.router[:, :128] = -4
            for rank in range(8):
                model.router[selected[rank],torch.arange(128)] = 4 - rank/8
    # The stress input intentionally routes all 128 tokens to the same eight experts.
    stress = x[0:1].expand_as(x).contiguous()
    x.numpy().tofile(a.out / 'x.bin')
    stress.numpy().tofile(a.out / 'stress_x.bin')
    ref, counts = reference(model, x)
    np.save(a.out / 'reference.npy', ref)
    np.save(a.out / 'reference_counts.npy', counts)
    if counts.max() > 32:
        raise RuntimeError(f'Synthetic input unexpectedly overflows: {counts.max()}')
    slots = torch.arange(128, dtype=torch.int32)
    grid = (slots[:,None]+slots[None,:]).to(torch.uint8 if a.grid_dtype=='uint8' else torch.int32)
    def args(c0,c1):
        if a.selector=='joint': return (x,grid[:c0,:c1].contiguous())
        if a.selector=='tagged':
            return (x,slots[:c0],slots[:c1],torch.zeros(PROFILES.index((c0,c1))+1,dtype=torch.int32))
        return (x,slots[:c0],slots[:c1])
    cpu = {}
    with torch.no_grad():
        for c0, c1 in PROFILES:
            y, ns = model(*args(c0,c1))
            cpu[f'{c0}_{c1}'] = y.numpy()
            assert np.array_equal(ns.numpy(), counts)
        for key, y in cpu.items():
            if not np.array_equal(y, cpu['128_128']):
                raise RuntimeError(f'CPU capacity mismatch: {key}')
        input_names=['x','grid'] if a.selector=='joint' else ['x','rows0','rows1']
        dynamic_axes=({'grid': {0:'C0',1:'C1'}} if a.selector=='joint' else
                      {'rows0': {0:'C0'},'rows1': {0:'C1'}})
        if a.selector=='tagged':
            input_names.append('tag'); dynamic_axes['tag']={0:'P'}
        torch.onnx.export(model, args(128,128), str(a.out / 'model.onnx'),
                          input_names=input_names,
                          output_names=['y', 'counts'],
                          dynamic_axes=dynamic_axes,
                          dynamo=False, opset_version=17, do_constant_folding=True)
    graph = onnx.load(a.out / 'model.onnx')
    CustomOpTransform.apply(graph)
    onnx.save_model(graph, a.out / 'model.onnx', save_as_external_data=True,
                    all_tensors_to_one_file=True, location='weights.bin', size_threshold=1024)
    onnx.checker.check_model(str(a.out / 'model.onnx'))
    specs=[{'C0':str(c0),'C1':str(c1)} for c0,c1 in PROFILES]
    if a.selector=='tagged':
        for i,spec in enumerate(specs): spec['P']=str(i+1)
    dump(a.out / 'specializations.json', {'specializations': specs})
    dump(a.out / 'single_specialization.json', {'specializations': specs[:1]})
    dump(a.out / 'mdp.json', {'connections': [{'devices': list(range(a.devices)), 'type': 'p2p'}],
        'partitions': [{'name': 'Partition0', 'devices': [
            {'deviceId': i, 'numCores': 16} for i in range(a.devices)]}]})
    dump(a.out / 'info.json', {'T': 128, 'H': a.hidden, 'I': a.intermediate, 'E': 128,
         'K': 8, 'devices': a.devices, 'profiles': PROFILES, 'synthetic': True, 'selector':a.selector,
         'grid_dtype':a.grid_dtype,
         'routing':a.routing,
         'cpu_counts_max_per_stage': counts.reshape(2,64).max(1).tolist(),
         'cpu_profiles_exact': True, 'input_bytes': (a.out/'x.bin').stat().st_size})
    print('EXPORT_OK', a.out, flush=True)


def compile_graph(a):
    info = json.loads((a.out / 'info.json').read_text())
    dest = a.out / a.precision
    dest.mkdir(exist_ok=True)
    for name, config in [('multi', 'specializations.json'), ('single', 'single_specialization.json')]:
        if (dest / name).exists():
            raise RuntimeError(f'Will not overwrite compiled directory {dest/name}')
        args = ['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100',
                f'-m={a.out / "model.onnx"}', '-convert-to-fp16', '-aic-num-cores=16',
                '-mos=1', '-aic-enable-depth-first', '-stats-level=70', '-compile-only',
                f'-network-specialization-config={a.out/config}', f'-aic-binary-dir={dest/name}']
        if info['devices'] > 1:
            args.append(f'-mdp-load-partition-config={a.out / "mdp.json"}')
        if a.precision == 'mxfp6':
            args.append('-mxfp6-matmul')
        command(args, dest / f'{name}_compile.log')


def analyze(a, run):
    info = json.loads((a.out / 'info.json').read_text())
    meta = json.loads((run / 'metadata.json').read_text())
    samples = np.genfromtxt(run / 'samples.csv', delimiter=',', names=True, encoding=None,
                           dtype=[('round','i4'),('mode','U8'),('profile','U16'),
                                  ('total_ms','f8'),('set_ms','f8'),('exec_ms','f8')])
    meta['qpc_info_scope']='MDP wrapper; zero constants/reqMem fields do not measure device memory'
    report = {'scope': 'synthetic single MoE, fixed T=128; one load and activation',
              'metadata': meta, 'precision': a.precision, 'profiles': {}}
    ref = np.load(a.out / 'reference.npy')
    cpu_counts = np.load(a.out / 'reference_counts.npy')
    baseline = np.fromfile(run/'128_128_y.bin', np.float16).reshape(ref.shape)
    for c0,c1 in PROFILES:
        key = f'{c0}_{c1}'
        y = np.fromfile(run/f'{key}_y.bin', np.float16).reshape(ref.shape)
        counts = np.fromfile(run/f'{key}_counts.bin', np.int32)
        if counts.size != 128 or np.any(counts < 0) or np.any(counts > 128):
            raise RuntimeError('Malformed routing counts')
        overflow = int(np.maximum(counts - np.repeat([c0,c1],64), 0).sum())
        rel = float(np.linalg.norm(y.astype(np.float64)-ref) / np.linalg.norm(ref))
        item = {'exact_vs_128_128': bool(np.array_equal(y,baseline)), 'overflow': overflow,
                'assignments': int(counts.sum()), 'stage_max': counts.reshape(2,64).max(1).tolist(),
                'cpu_counts_equal': bool(np.array_equal(counts,cpu_counts)), 'relative_l2_vs_cpu':rel,
                'latency': {}}
        for mode in ('solo','cycle'):
            s = samples[(samples['mode']==mode) & (samples['profile']==key)]
            item['latency'][mode] = {f'{field}_median':float(np.median(s[field]))
                                     for field in ('total_ms','set_ms','exec_ms')}
            item['latency'][mode]['p10_ms'] = float(np.percentile(s['total_ms'],10))
            item['latency'][mode]['p90_ms'] = float(np.percentile(s['total_ms'],90))
            item['latency'][mode]['samples'] = len(s)
        item['cycle_minus_solo_ms'] = (item['latency']['cycle']['total_ms_median'] -
                                       item['latency']['solo']['total_ms_median'])
        report['profiles'][key] = item
    stress = np.fromfile(run/'stress_counts.bin',np.int32)
    stress_y = np.fromfile(run/'stress_y.bin',np.float16)
    stress_full = np.fromfile(run/'stress_full_y.bin',np.float16)
    report['stress'] = {'counts_max':int(stress.max()), 'assignments':int(stress.sum()),
                       'overflow_at_32':int(np.maximum(stress-32,0).sum()),
                       'differs_from_full_capacity':bool(not np.array_equal(stress_y,stress_full)),
                       'scope':'intentional truncation; detector test, not a recovery implementation'}
    if (run/'unlisted_y.bin').exists():
        uy=np.fromfile(run/'unlisted_y.bin',np.float16).reshape(ref.shape)
        report['unlisted_48_48']={'nonzero_tokens':int(np.any(uy!=0,axis=1).sum()),
                                  'exact_vs_stress_128_128':bool(np.array_equal(uy.ravel(),stress_full)),
                                  'note':'binding success alone does not establish which compiled shape runs'}
    report['reference_accuracy_threshold']=0.05
    report['reference_accuracy_pass']=bool(all(p['relative_l2_vs_cpu'] < 0.05
                                             for p in report['profiles'].values()))
    report['capacity_equivalence_pass'] = bool(all(
        p['exact_vs_128_128'] and p['overflow']==0 and p['assignments']==1024 and
        (p['cpu_counts_equal'] or info.get('routing','random')!='margin')
        for p in report['profiles'].values()) and report['stress']['overflow_at_32'] == 768
        and report['stress']['assignments']==1024 and report['stress']['differs_from_full_capacity'])
    report['validation_pass']=report['reference_accuracy_pass'] and report['capacity_equivalence_pass']
    report['equivalence_only']=a.equivalence_only
    report['timing_scope']=('capacity equivalence and CPU reference gates passed' if report['validation_pass'] else
                            'capacity equivalence only; absolute accuracy is not established')
    dump(run/'result.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if not (report['validation_pass'] or (a.equivalence_only and report['capacity_equivalence_pass'])):
        raise RuntimeError('Capacity probe validation failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['export','compile','run','analyze','all'], default='all')
    p.add_argument('--devices',type=int,choices=[1,2,4],default=4)
    p.add_argument('--selector',choices=['joint','separate','tagged'],default='tagged')
    p.add_argument('--grid-dtype',choices=['int32','uint8'],default='int32')
    p.add_argument('--routing',choices=['margin','random'],default='margin')
    p.add_argument('--device-ids',default=None)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--intermediate',type=int,default=128)
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16')
    p.add_argument('--equivalence-only',action='store_true',
                   help='Accept exact capacity equivalence while preserving any failed reference gate')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--run-name',default='run')
    a=p.parse_args(); a.out=a.out.resolve()
    if a.iterations<1 or a.hidden<1 or a.intermediate<1: p.error('Dimensions/iterations must be positive')
    if a.stage in ['export','all']: export(a)
    if a.stage in ['compile','all']: compile_graph(a)
    run = a.out/a.precision/a.run_name
    if a.stage in ['run','all']:
        info=json.loads((a.out/'info.json').read_text())
        devices=a.device_ids or ','.join(map(str,range(info['devices'])))
        ids=[int(i) for i in devices.split(',')]
        if len(ids)!=info['devices'] or len(set(ids))!=len(ids): p.error('Invalid device mapping')
        host=a.out/'host'
        command(['g++','-O2','-std=c++17','-Wall','-Wextra','-I/opt/qti-aic/dev/inc',
                 Path(__file__).with_name('moe_capacity_probe_host.cpp'),'-o',host,
                 '-L/opt/qti-aic/dev/lib/x86_64','-lQAic','-Wl,-rpath,/opt/qti-aic/dev/lib/x86_64'],
                a.out/'host_build.log')
        command([host,a.out/a.precision/'multi',a.out,run,devices,info['H'],a.iterations],
                a.out/a.precision/f'{a.run_name}_host.log')
    if a.stage in ['run','analyze','all']: analyze(a,run)


if __name__=='__main__': main()
