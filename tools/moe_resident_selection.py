#!/usr/bin/env python3
"""Probe runtime selection from resident expert weights on a single AI 100 card.

Global 128-expert/top-8 router; this graph returns only the contribution of a
32-expert resident bank. Two groups of 16 use capacities 64/32. Normal ID lists
permute the bank; a duplicate-half diagnostic proves the indices affect results.
"""
import argparse
import hashlib
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

CASES = ['identity', 'swap', 'shuffle', 'duplicate_control']
VARIANTS = ['dynamic', 'static_identity', 'static_swap', 'static_shuffle']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def command(args, log):
    print('RUN',' '.join(map(str,args)),flush=True)
    with log.open('w') as f:
        f.write(repr(list(map(str,args)))+'\n');f.flush()
        return subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT).returncode


class ResidentMoE(torch.nn.Module):
    def __init__(self, H, I, group_size=16, capacities=None):
        super().__init__()
        self.H,self.I,self.K,self.E_router=H,I,8,128
        self.group_size=group_size
        self.capacities=list(capacities if capacities is not None else [64,32])
        if group_size<1 or group_size*len(self.capacities)!=32 or any(c<1 or c>128 for c in self.capacities):
            raise ValueError('Groups must cover the 32-expert bank with capacities in [1,128]')
        self.capacity_check='oob'
        generator=torch.Generator().manual_seed(2701)
        self.router=torch.nn.Parameter(torch.zeros(128,H,dtype=torch.float16),requires_grad=False)
        self.Wg=torch.nn.Parameter((torch.randn(32,H,I,generator=generator)*0.02).half(),requires_grad=False)
        self.Wu=torch.nn.Parameter((torch.randn(32,H,I,generator=generator)*0.02).half(),requires_grad=False)
        self.Wd=torch.nn.Parameter((torch.randn(32,I,H,generator=generator)*0.02).half(),requires_grad=False)
        self.register_buffer('fixed_order',torch.arange(32,dtype=torch.int32))
        self.dynamic=True

    def row_indices(self,stage,capacity):
        return torch.arange(capacity,dtype=torch.int32)

    def forward(self,x,expert_ids=None):
        order=expert_ids if self.dynamic else self.fixed_order
        rw=MoELayerEP.route(self,x)
        counts=(rw>0).sum(0).to(torch.int32)[:32]
        expert_out=x.new_zeros((self.group_size,128,self.H))
        for stage,capacity in enumerate(self.capacities):
            ids=order[stage*self.group_size:(stage+1)*self.group_size]
            routes=torch.index_select(rw.transpose(0,1),0,ids)
            active=routes>0
            matched=MoELayerEP.matched_idx(self,active,capacity)
            packed=CtxGatherFunc3DGeneralized.apply(x.unsqueeze(0).expand(self.group_size,-1,-1),matched)
            wg=torch.index_select(self.Wg,0,ids)
            wu=torch.index_select(self.Wu,0,ids)
            wd=torch.index_select(self.Wd,0,ids)
            z=(F.silu(packed@wg)*(packed@wu))@wd
            weights=CtxGatherFunc3DGeneralized.apply(routes.unsqueeze(-1),matched)
            old=CtxGatherFunc3DGeneralized.apply(expert_out,matched)
            update=old+z*weights
            valid=(self.row_indices(stage,capacity)[None,:] < active.sum(1,keepdim=True))
            update=torch.where(valid.unsqueeze(-1),update,torch.zeros_like(update))
            expert_out=CtxScatterFunc3DGeneralized.apply(expert_out,matched,update)
        return torch.einsum('nth->th',expert_out),counts


def export(a):
    root=a.out;root.mkdir(parents=True,exist_ok=False);(root/'orders').mkdir();(root/'reference').mkdir()
    torch.set_num_threads(8)
    if a.hidden<128: raise ValueError('H must be at least 128 for controlled routing')
    model=ResidentMoE(a.hidden,a.intermediate).eval()
    x=torch.randn(128,a.hidden,generator=torch.Generator().manual_seed(2702)).half()
    x[:,:128]=torch.eye(128,dtype=torch.float16)
    selected=torch.randn(128,128,generator=torch.Generator().manual_seed(2703)).topk(8,dim=0).indices
    with torch.no_grad():
        model.router[:,:128]=-4
        for rank in range(8):model.router[selected[rank],torch.arange(128)]=2
    # Selected logits are equal and well separated from all others. Router
    # weights and normalized 1/8 contributions are exactly representable.
    orders={'identity':np.arange(32,dtype=np.int32),
            'swap':np.roll(np.arange(32,dtype=np.int32),16),
            'shuffle':np.random.default_rng(2704).permutation(32).astype(np.int32),
            'duplicate_control':np.tile(np.arange(16,dtype=np.int32),2)}
    x.numpy().tofile(root/'x.bin')
    for name,ids in orders.items():ids.tofile(root/'orders'/f'{name}.bin')
    with torch.no_grad():
        rw=MoELayerEP.route(model,x)
        counts=(rw>0).sum(0).to(torch.int32)[:32].numpy()
        if counts.max()>32:raise RuntimeError('Normal input exceeds smallest capacity')
        per_expert=torch.zeros(32,128,a.hidden,dtype=torch.float32)
        for e in range(32):
            tokens=torch.nonzero(rw[:,e]>0).flatten()
            xs=x[tokens].float()
            y=(F.silu(xs@model.Wg[e].float())*(xs@model.Wu[e].float()))@model.Wd[e].float()
            per_expert[e,tokens]=y*rw[tokens,e,None].float()
        for name,ids in orders.items():
            ref=per_expert[torch.from_numpy(ids).long()].sum(0).numpy()
            np.save(root/'reference'/f'{name}.npy',ref)
    np.save(root/'reference/counts.npy',counts)
    static_audits={}
    for variant in VARIANTS:
        directory=root/variant;directory.mkdir()
        model.dynamic=variant=='dynamic'
        if not model.dynamic:model.fixed_order.copy_(torch.from_numpy(orders[variant.removeprefix('static_')]))
        args=(x,torch.from_numpy(orders['identity'])) if model.dynamic else (x,)
        path=directory/'model.onnx'
        with torch.no_grad():
            torch.onnx.export(model,args,str(path),input_names=['x','expert_ids'] if model.dynamic else ['x'],
                              output_names=['y','counts'],dynamo=False,opset_version=17,do_constant_folding=True)
        graph=onnx.load(path);CustomOpTransform.apply(graph)
        initializers={t.name:list(t.dims) for t in graph.graph.initializer}
        weight_gathers=[{'name':n.name,'bank':n.input[0],'bank_shape':initializers[n.input[0]],
                         'indices':n.input[1]} for n in graph.graph.node
                        if n.op_type=='Gather' and n.input[0] in initializers and len(initializers[n.input[0]])==3]
        matmuls=[{'name':n.name,'rhs':n.input[1],'rhs_is_initializer':n.input[1] in initializers}
                 for n in graph.graph.node if n.op_type=='MatMul']
        static_audits[variant]={'weight_gathers':weight_gathers,'matmuls':matmuls}
        onnx.save_model(graph,path,save_as_external_data=True,all_tensors_to_one_file=True,
                        location='weights.bin',size_threshold=1024)
        onnx.checker.check_model(str(path))
        print('EXPORT_OK',variant,flush=True)
    dump(root/'graph_audit.json',static_audits)
    hashes={name:hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()
            for name,t in [('gate',model.Wg),('up',model.Wu),('down',model.Wd)]}
    dump(root/'info.json',{'T':128,'H':a.hidden,'I':a.intermediate,'global_experts':128,'resident_experts':32,
         'group_size':16,'capacities':[64,32],'top_k':8,'normal_cases':CASES[:3],
         'diagnostic_case':CASES[3],'resident_assignments':int(counts.sum()),'max_count':int(counts.max()),
         'bank_bytes':32*3*a.hidden*a.intermediate*2,'bank_sha256':hashes,
         'scope':'single-card partial MoE; synthetic resident weights; no attention/KV/inter-card transfers'})


def compile_graph(a):
    base=a.out/a.precision;base.mkdir(exist_ok=True);results={}
    for variant in VARIANTS:
        dest=base/variant
        if dest.exists():raise RuntimeError(f'Will not overwrite {dest}')
        args=['/opt/qti-aic/exec/qaic-compile','-aic-hw','-aic-hw-version=ai100',
              f'-m={a.out/variant/"model.onnx"}','-convert-to-fp16','-aic-num-cores=16',
              '-mos=1','-aic-enable-depth-first','-stats-level=70','-compile-only',f'-aic-binary-dir={dest}']
        if a.precision=='mxfp6':args.append('-mxfp6-matmul')
        status=command(args,base/f'{variant}_compile.log')
        results[variant]={'exit_code':status,'qpc_exists':(dest/'programqpc.bin').exists()}
        dump(base/'compilation.json',results)
    if not all(v['exit_code']==0 and v['qpc_exists'] for v in results.values()):
        raise RuntimeError('Some compile probes failed; see compilation.json and logs')


def compare(y,reference):
    y=y.astype(np.float64);reference=reference.astype(np.float64)
    if y.shape!=reference.shape or not np.isfinite(y).all():raise RuntimeError('Invalid output')
    return {'exact':bool(np.array_equal(y,reference)),
            'relative_l2':float(np.linalg.norm(y-reference)/max(np.linalg.norm(reference),1e-12)),
            'max_abs':float(np.max(np.abs(y-reference)))}


def analyze(a):
    info=json.loads((a.out/'info.json').read_text());base=a.out/a.precision/a.run_name
    report={'scope':info['scope'],'precision_flag':a.precision,'variants':{},'dynamic_vs_static':{}}
    counts_ref=np.load(a.out/'reference/counts.npy');outputs={}
    for variant in VARIANTS:
        run=base/variant
        rows=np.genfromtxt(run/'samples.csv',delimiter=',',names=True,encoding=None,
                          dtype=[('round','i4'),('mode','U8'),('case','U32'),('total_ms','f8')])
        rows=np.atleast_1d(rows);vr={'cases':{},'metadata':json.loads((run/'metadata.json').read_text())}
        cases=CASES if variant=='dynamic' else [variant.removeprefix('static_')]
        outputs[variant]={}
        for case in cases:
            y=np.fromfile(run/f'{case}_y.bin',np.float16).reshape(128,info['H']);outputs[variant][case]=y
            counts=np.fromfile(run/f'{case}_counts.bin',np.int32)
            p={'vs_cpu':compare(y,np.load(a.out/'reference'/f'{case}.npy')),
               'counts_match':bool(np.array_equal(counts,counts_ref)),'timing':{}}
            for mode in ['solo','cycle']:
                samples=rows[(rows['mode']==mode)&(rows['case']==case)]['total_ms']
                if len(samples):p['timing'][mode]={'median_ms':float(np.median(samples)),
                     'p10_ms':float(np.percentile(samples,10)),'p90_ms':float(np.percentile(samples,90)),
                     'samples':len(samples)}
            vr['cases'][case]=p
        report['variants'][variant]=vr
    for case in CASES[:3]:
        d=compare(outputs['dynamic'][case],outputs[f'static_{case}'][case])
        dt=report['variants']['dynamic']['cases'][case]['timing']['solo']['median_ms']
        st=report['variants'][f'static_{case}']['cases'][case]['timing']['solo']['median_ms']
        d.update(dynamic_ms=dt,static_ms=st,dynamic_over_static=dt/st)
        report['dynamic_vs_static'][case]=d
    report['selection_control_vs_identity']=compare(outputs['dynamic'][CASES[3]],outputs['dynamic']['identity'])
    report['dynamic_reference_pass']=all(p['counts_match'] and p['vs_cpu']['relative_l2']<0.05
                                         for p in report['variants']['dynamic']['cases'].values())
    report['static_reference_pass']=all(v['cases'][name.removeprefix('static_')]['counts_match'] and
                                       v['cases'][name.removeprefix('static_')]['vs_cpu']['relative_l2']<0.05
                                       for name,v in report['variants'].items() if name!='dynamic')
    report['selection_control_changes_output']=report['selection_control_vs_identity']['relative_l2']>0.1
    report['pair_equivalence_pass']=all(v['relative_l2']<0.005 for v in report['dynamic_vs_static'].values())
    report['selection_equivalence_pass']=bool(report['pair_equivalence_pass'] and
        report['selection_control_changes_output'] and all(p['counts_match']
            for v in report['variants'].values() for p in v['cases'].values()))
    report['reference_accuracy_pass']=report['dynamic_reference_pass'] and report['static_reference_pass']
    report['validation_pass']=bool(report['reference_accuracy_pass'] and report['selection_equivalence_pass'])
    report['equivalence_only']=a.equivalence_only
    dump(base/'result.json',report);print(json.dumps(report,indent=2),flush=True)
    if not report['selection_equivalence_pass'] or (not a.equivalence_only and not report['validation_pass']):
        raise RuntimeError('Selection probe validation failed; see separate equivalence and accuracy fields')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['export','compile','run','analyze','all'],default='all')
    p.add_argument('--hidden',type=int,default=256);p.add_argument('--intermediate',type=int,default=128)
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16')
    p.add_argument('--device',type=int,default=0);p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--equivalence-only',action='store_true',
                   help='Allow shared quantization accuracy failure; validation_pass remains false')
    p.add_argument('--run-name',default='run');a=p.parse_args();a.out=a.out.resolve()
    if a.hidden<128 or a.intermediate<1 or a.iterations<1:p.error('Invalid dimensions/iterations')
    if a.stage in ['export','all']:export(a)
    if a.stage in ['compile','all']:compile_graph(a)
    if a.stage in ['run','all']:
        info=json.loads((a.out/'info.json').read_text());base=a.out/a.precision/a.run_name;base.mkdir(exist_ok=False)
        host=base/'host'
        args=['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I/opt/qti-aic/dev/inc',
              Path(__file__).with_name('moe_resident_selection_host.cpp'),'-o',host,
              '-L/opt/qti-aic/dev/lib/x86_64','-lQAic','-Wl,-rpath,/opt/qti-aic/dev/lib/x86_64']
        if command(args,base/'host_build.log'):raise RuntimeError('Host build failed')
        for variant in VARIANTS:
            if command([host,a.out/a.precision/variant,a.out,base/variant,a.device,info['H'],a.iterations,variant],
                       base/f'{variant}_host.log'):raise RuntimeError(f'Host failed: {variant}')
    if a.stage in ['run','analyze','all']:analyze(a)


if __name__=='__main__':main()
