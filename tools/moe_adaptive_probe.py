#!/usr/bin/env python3
"""Combine resident expert IDs and eight-group capacity specialization on AI 100."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.utils import Extractor
import torch
import torch.nn.functional as F
from QEfficient.base.onnx_transforms import CustomOpTransform

from moe_resident_selection import ResidentMoE, CASES, command, compare, dump
from moe_tier_bench_export import MoELayerEP
from moe_capacity_audit import CtxScatter3DInt, CtxGather3D
from moe_resident_audit import trace_summary

SDK=Path('/opt/qti-aic')
PROFILES=[('full128',[128]*8),('uniform32',[32]*8),('uniform16',[16]*8),
          ('alternating',[32,16]*4),('mixed',[64,32,16,16,32,16,16,16])]


class AdaptiveMoE(ResidentMoE):
    def __init__(self,H,I):
        super().__init__(H,I,4,[128]*8)

    def forward(self,x,expert_ids,rows0,rows1,rows2,rows3,rows4,rows5,rows6,rows7,tag):
        # Values stay live, while the input shapes select compiled capacities.
        self.selector_rows=(rows0+tag.sum().to(torch.int32),rows1,rows2,rows3,rows4,rows5,rows6,rows7)
        self.capacities=[r.shape[0] for r in self.selector_rows]
        return super().forward(x,expert_ids)

    def row_indices(self,stage,capacity):
        return self.selector_rows[stage]


def export(a):
    root=a.out; source=a.source.resolve(); original=json.loads((source/'info.json').read_text())
    root.mkdir(parents=True,exist_ok=False)
    for name in ['orders','reference']: shutil.copytree(source/name,root/name)
    shutil.copyfile(source/'x.bin',root/'x.bin'); torch.set_num_threads(8)
    model=AdaptiveMoE(original['H'],original['I']).eval()
    hashes={n:hashlib.sha256(w.detach().numpy().tobytes()).hexdigest()
            for n,w in [('gate',model.Wg),('up',model.Wu),('down',model.Wd)]}
    if hashes!=original['bank_sha256']: raise RuntimeError('Source weight hash mismatch')
    graph=onnx.load(source/'dynamic/model.onnx',load_external_data=False)
    rn=next(n.input[1] for n in graph.graph.node if n.name=='/MatMul')
    router=next(w for w in graph.graph.initializer if w.name==rn)
    with torch.no_grad():
        model.router.copy_(torch.from_numpy(onnx.numpy_helper.to_array(router,base_dir=str(source/'dynamic')).copy()).T)
    x=torch.from_numpy(np.fromfile(root/'x.bin',np.float16).reshape(128,original['H']))
    ids=torch.from_numpy(np.fromfile(root/'orders/identity.bin',np.int32))
    counts=np.load(root/'reference/counts.npy')
    if counts.max()>16: raise RuntimeError('Normal input exceeds minimum capacity')
    # Repeat a token with maximal resident-bank participation to force overflow.
    with torch.no_grad():
        rw=MoELayerEP.route(model,x)
        token=int((rw[:,:32]>0).sum(1).argmax())
        stress=x[token:token+1].expand_as(x).contiguous(); srw=MoELayerEP.route(model,stress)
        scounts=(srw>0).sum(0).to(torch.int32)[:32].numpy()
        ref=torch.zeros_like(stress,dtype=torch.float32)
        for e in range(32):
            if scounts[e]:
                z=stress.float()
                ref+=((F.silu(z@model.Wg[e].float())*(z@model.Wu[e].float()))@model.Wd[e].float())*srw[:,e,None].float()
    stress.numpy().tofile(root/'stress_x.bin'); np.save(root/'reference/stress.npy',ref.numpy())
    np.save(root/'reference/stress_counts.npy',scounts)
    axes={f'rows{i}':{0:f'C{i}'} for i in range(8)}; axes['tag']={0:'P'}
    input_names=['x','expert_ids']+[f'rows{i}' for i in range(8)]+['tag']
    args=(x,ids,*[torch.arange(128,dtype=torch.int32) for _ in range(8)],torch.zeros(1,dtype=torch.int32))
    with torch.no_grad():
        torch.onnx.export(model,args,str(root/'model.onnx'),input_names=input_names,
                          output_names=['y','counts'],dynamic_axes=axes,dynamo=False,
                          opset_version=17,do_constant_folding=True)
    graph=onnx.load(root/'model.onnx'); CustomOpTransform.apply(graph)
    constants={w.name:list(w.dims) for w in graph.graph.initializer}
    audit={'weight_gathers':[n.name for n in graph.graph.node if n.op_type=='Gather'
                            and n.input[0] in constants and len(constants[n.input[0]])==3],
           'expert_matmuls':[n.name for n in graph.graph.node if n.op_type=='MatMul' and n.name!='/MatMul']}
    assert len(audit['weight_gathers'])==len(audit['expert_matmuls'])==24
    onnx.save_model(graph,root/'model.onnx',save_as_external_data=True,all_tensors_to_one_file=True,
                    location='weights.bin',size_threshold=1024)
    onnx.checker.check_model(str(root/'model.onnx')); dump(root/'graph_audit.json',audit)
    profiles=[{'name':name,'capacities':cs,'tag':i+1,'padded_expert_rows':4*sum(cs)} for i,(name,cs) in enumerate(PROFILES)]
    specs=[{**{f'C{g}':str(c) for g,c in enumerate(p['capacities'])},'P':str(p['tag'])} for p in profiles]
    for name,indices in [('multi',range(len(specs))),('single128',[0]),('single32',[1])]:
        dump(root/f'{name}_specializations.json',{'specializations':[specs[i] for i in indices]})
    (root/'profiles.txt').write_text(''.join(p['name']+' '+str(p['tag'])+' '+' '.join(map(str,p['capacities']))+'\n' for p in profiles))
    dump(root/'info.json',{**original,'source':str(source),'group_size':4,'groups':8,'profiles':profiles,
         'capacities':None,'stress_token':token,'stress_resident_assignments':int(scounts.sum()),
         'bank_sha256':hashes,'scope':'Single-card synthetic partial MoE; combined runtime IDs and shape-specialized capacities; no KV/inter-card transfers'})
    print('EXPORT_OK',root,flush=True)


def compile_graph(a):
    base=a.out/a.precision; base.mkdir(exist_ok=True); report={}
    for variant in ['multi','single128','single32']:
        dest=base/variant
        if dest.exists(): raise RuntimeError(f'Will not overwrite {dest}')
        args=[SDK/'exec/qaic-compile','-aic-hw','-aic-hw-version=ai100',f'-m={a.out/"model.onnx"}',
              '-convert-to-fp16','-aic-num-cores=16','-mos=1','-aic-enable-depth-first','-stats-level=70',
              '-compile-only',f'-network-specialization-config={a.out/f"{variant}_specializations.json"}',
              f'-aic-binary-dir={dest}']
        if a.precision=='mxfp6': args.append('-mxfp6-matmul')
        status=command(args,base/f'{variant}_compile.log')
        report[variant]={'exit_code':status,'qpc_exists':(dest/'programqpc.bin').exists()}; dump(base/'compilation.json',report)
    if not all(v['exit_code']==0 and v['qpc_exists'] for v in report.values()): raise RuntimeError('Compilation failure; see logs')


def run(a,info):
    base=a.out/a.precision/a.run_name; base.mkdir(exist_ok=False); host=base/'host'
    if command(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',f'-I{SDK}/dev/inc',
                Path(__file__).with_name('moe_adaptive_host.cpp'),'-o',host,
                f'-L{SDK}/dev/lib/x86_64','-lQAic',f'-Wl,-rpath,{SDK}/dev/lib/x86_64'],base/'host_build.log'):
        raise RuntimeError('Host build failed')
    dump(base/'run_config.json',{'iterations':a.iterations,'device':a.device})
    for variant in ['multi','single32']:
        if command([host,a.out/a.precision/variant,a.out,base/variant,a.device,info['H'],a.iterations,variant],
                   base/f'{variant}_host.log'): raise RuntimeError(f'Host failed: {variant}')


def sharing(a):
    """Compile-only subsets isolate the storage cost of adding capacity families."""
    base=a.out/a.precision
    specs=json.loads((a.out/'multi_specializations.json').read_text())['specializations']
    result={}
    for variant,indices in [('compact3',[1,2,3]),('safe4',[0,1,2,3])]:
        config=base/f'{variant}_specializations.json';dump(config,{'specializations':[specs[i] for i in indices]})
        dest=base/variant
        if dest.exists():raise RuntimeError(f'Will not overwrite {dest}')
        args=[SDK/'exec/qaic-compile','-aic-hw','-aic-hw-version=ai100',f'-m={a.out/"model.onnx"}',
              '-convert-to-fp16','-aic-num-cores=16','-mos=1','-aic-enable-depth-first','-stats-level=70',
              '-compile-only',f'-network-specialization-config={config}',f'-aic-binary-dir={dest}']
        if a.precision=='mxfp6':args.append('-mxfp6-matmul')
        status=command(args,base/f'{variant}_compile.log')
        result[variant]={'profile_indices':indices,'exit_code':status,'qpc_exists':(dest/'programqpc.bin').exists()}
        dump(base/'sharing_compilation.json',result)
    if not all(v['exit_code']==0 and v['qpc_exists'] for v in result.values()):raise RuntimeError('Sharing subset compile failed')


def latency(rows):
    return {key:{'median_ms':float(np.median([float(r[key]) for r in rows])),
                 'p10_ms':float(np.percentile([float(r[key]) for r in rows],10)),
                 'p90_ms':float(np.percentile([float(r[key]) for r in rows],90))}
            for key in ['total_ms','set_ms','exec_ms']} | {'samples':len(rows)}


def analyze(a,info):
    base=a.out/a.precision/a.run_name
    read_y=lambda path: np.fromfile(path,np.float16).reshape(128,info['H'])
    baseline={c:read_y(base/'multi'/f'full128_{c}_y.bin') for c in CASES}
    counts_ref=np.load(a.out/'reference/counts.npy'); orders={c:np.fromfile(a.out/'orders'/f'{c}.bin',np.int32) for c in CASES}
    report={'scope':info['scope'],'precision':a.precision,'variants':{},'equivalence_only':a.equivalence_only}
    for variant in ['multi','single32']:
        root=base/variant
        with (root/'samples.csv').open() as f: rows=list(csv.DictReader(f))
        vr={'metadata':json.loads((root/'metadata.json').read_text()),'profiles':{}}
        with (root/'resources.csv').open() as f: resources=list(csv.DictReader(f))
        vr['residency']={'activation_dram_delta_mib':(int(resources[0]['dram_free_kib'])-int(resources[1]['dram_free_kib']))/1024,
                         'stable_after_activation':len({r['dram_free_kib'] for r in resources[1:]})==1}
        for p in info['profiles']:
            name=p['name']
            if variant=='single32' and name!='uniform32': continue
            value={'capacities':p['capacities'],'padded_expert_rows':p['padded_expert_rows'],'cases':{},'latency':{}}
            for c in CASES:
                y=read_y(root/f'{name}_{c}_y.bin'); counts=np.fromfile(root/f'{name}_{c}_counts.bin',np.int32)
                capacity=np.repeat(p['capacities'],4)
                value['cases'][c]={'vs_cpu':compare(y,np.load(a.out/'reference'/f'{c}.npy')),
                    'vs_full128':compare(y,baseline[c]),'counts_match':bool(np.array_equal(counts,counts_ref)),
                    'overflow':int(np.maximum(counts[orders[c]]-capacity,0).sum())}
            for mode in ['held','both','ids_only','shape_only']:
                selected=[r for r in rows if r['profile']==name and r['mode']==mode]
                value['latency'][mode]=latency(selected)
            value['diagnostic_changes_output']=compare(read_y(root/f'{name}_duplicate_control_y.bin'),
                                                       read_y(root/f'{name}_identity_y.bin'))['relative_l2']>0.1
            vr['profiles'][name]=value
        report['variants'][variant]=vr
    report['single32_vs_multi32']={c:compare(read_y(base/'single32'/f'uniform32_{c}_y.bin'),
        read_y(base/'multi'/f'uniform32_{c}_y.bin')) for c in CASES}
    stress_counts=np.load(a.out/'reference/stress_counts.npy'); stress_reference=np.load(a.out/'reference/stress.npy')
    with (base/'multi/recovery.csv').open() as f: recovery=list(csv.DictReader(f))
    stress={'scope':'Stateless reject-and-replay in one program; no KV rollback','cases':{},'recovery_profiles':{}}
    recovery_ok=True
    for c in CASES[:3]:
        full=read_y(base/'multi'/f'stress_{c}_full_y.bin')
        counts=np.fromfile(base/'multi'/f'stress_{c}_counts.bin',np.int32)
        stress['cases'][c]={'vs_cpu':compare(full,stress_reference),'counts_match':bool(np.array_equal(counts,stress_counts))}
        recovery_ok &= stress['cases'][c]['counts_match']
        for name in ['uniform16','mixed']:
            cs=next(p['capacities'] for p in info['profiles'] if p['name']==name)
            expected=int(np.maximum(stress_counts[orders[c]]-np.repeat(cs,4),0).sum())
            selected=[r for r in recovery if r['case']==c and r['profile']==name]
            assert len(selected)==20
            good=all(int(r['overflow'])==expected and int(r['truncated_diff'])==1 and int(r['recovered_exact'])==1 for r in selected)
            y=read_y(base/'multi'/f'stress_{c}_{name}_recovered_y.bin')
            good &= bool(np.array_equal(y,full)) and expected>0
            recovery_ok &= good
            stress['cases'][c][name]={'expected_overflow':expected,'recovered_exact':good}
    for name in ['uniform16','mixed']:
        selected=[r for r in recovery if r['profile']==name]
        stress['recovery_profiles'][name]={k:float(np.median([float(r[k]) for r in selected]))
            for k in ['low_ms','replay_ms','total_ms']} | {'samples':len(selected)}
    stress['recovery_pass']=bool(recovery_ok); report['stress']=stress
    all_profiles=[p for v in report['variants'].values() for p in v['profiles'].values()]
    report['combined_equivalence_pass']=bool(recovery_ok and all(p['diagnostic_changes_output'] and all(
        c['counts_match'] and c['overflow']==0 and c['vs_full128']['relative_l2']<0.005 for c in p['cases'].values())
        for p in all_profiles) and all(v['relative_l2']<0.005 for v in report['single32_vs_multi32'].values()))
    report['reference_accuracy_pass']=bool(all(c['vs_cpu']['relative_l2']<0.05 for p in all_profiles for c in p['cases'].values())
        and all(c['vs_cpu']['relative_l2']<0.05 for c in stress['cases'].values()))
    report['validation_pass']=report['combined_equivalence_pass'] and report['reference_accuracy_pass']
    dump(base/'result.json',report)
    for name,p in report['variants']['multi']['profiles'].items():
        print(name,{m:round(v['total_ms']['median_ms'],4) for m,v in p['latency'].items()},flush=True)
    print('EQUIVALENCE',report['combined_equivalence_pass'],'ACCURACY',report['reference_accuracy_pass'],flush=True)
    if not report['combined_equivalence_pass'] or (not a.equivalence_only and not report['validation_pass']):
        raise RuntimeError('Validation failed; see separate mechanism and accuracy gates')


def audit(a,info):
    base=a.out/a.precision; graph=onnx.load(a.out/'model.onnx')
    graph_info=json.loads((a.out/'graph_audit.json').read_text())
    first=set(graph_info['expert_matmuls'][::3])
    outputs=[n.input[0] for n in graph.graph.node if n.name in first]
    sub=Extractor(graph).extract_model([v.name for v in graph.graph.input],outputs)
    evaluator=ReferenceEvaluator(sub,new_ops=[CtxScatter3DInt,CtxGather3D])
    x=np.fromfile(a.out/'x.bin',np.float16).reshape(128,info['H'])
    ids=np.fromfile(a.out/'orders/identity.bin',np.int32)
    report={'scope':'Logical packing shapes, packed constant storage and device traces; not PCIe DMA accounting',
            'profiles':{},'packages':{}}
    for p in info['profiles']:
        inputs={'x':x,'expert_ids':ids,'tag':np.zeros(p['tag'],np.int32)}
        inputs.update({f'rows{g}':np.arange(c,dtype=np.int32) for g,c in enumerate(p['capacities'])})
        shapes=[list(v.shape) for v in evaluator.run(None,inputs)]
        assert shapes==[[4,c,info['H']] for c in p['capacities']],shapes
        report['profiles'][p['name']]={'logical_packed_shapes':shapes,'padded_expert_rows':sum(s[0]*s[1] for s in shapes)}
    packages=['multi','single128','single32']+[v for v in ['compact3','safe4'] if (base/v/'programqpc.bin').exists()]
    for variant in packages:
        qpc=base/variant/'programqpc.bin'; dest=base/f'{variant}_segments'
        if not dest.exists():
            if command([SDK/'tools/qaic-qpc','extract','--qpc',qpc,'--output-dir',dest,
                        '-s','*StaticConstants.constants.bin','-s','*networkdesc.bin_dir/networkdesc.json'],base/f'{variant}_extract.log'):
                raise RuntimeError('QPC extraction failed')
        constants=sum(p.stat().st_size for p in dest.rglob('StaticConstants.constants.bin'))
        if not constants:raise RuntimeError('No static constants found')
        descriptions=[json.loads(p.read_text()) for p in dest.rglob('networkdesc.json')]
        report['packages'][variant]={'static_constants_bytes':constants,'qpc_bytes':qpc.stat().st_size,
            'network_shapes':[{'name':d['network_name'],'allowed_shapes':d.get('allowed_shapes'),
                               'input_names':[v['name'] for v in d['inputs']]} for d in descriptions]}
    report['multi_over_single128_constants_ratio']=report['packages']['multi']['static_constants_bytes']/report['packages']['single128']['static_constants_bytes']
    report['multi_over_single32_constants_ratio']=report['packages']['multi']['static_constants_bytes']/report['packages']['single32']['static_constants_bytes']
    dump(base/'package_audit.json',report)
    if a.profile:
        dest=base/'profiling';dest.mkdir(exist_ok=False)
        for p in info['profiles']:
            name=p['name'];subdir=dest/name;subdir.mkdir()
            ios=[{'path':str(a.out/'x.bin'),'dims':[128,info['H']],'elem-size':2,'io-direction':'in','map-to':'x'},
                 {'path':str(a.out/'orders/identity.bin'),'dims':[32],'elem-size':4,'io-direction':'in','map-to':'expert_ids'}]
            values={f'rows{g}':np.arange(c,dtype=np.int32) for g,c in enumerate(p['capacities'])}
            values['tag']=np.zeros(p['tag'],np.int32)
            for key,v in values.items():
                path=subdir/f'{key}.bin';v.tofile(path)
                ios.append({'path':str(path),'dims':list(v.shape),'elem-size':4,'io-direction':'in','map-to':key})
            for key,shape,size in [('y',[128,info['H']],2),('counts',[32],4)]:
                ios.append({'path':str(base/a.run_name/'multi'/f'{name}_identity_{key}.bin'),
                            'dims':shape,'elem-size':size,'io-direction':'out','map-to':key})
            config=subdir/'io.json';dump(config,{'IO-files':[ios]});stats=subdir/'stats';stats.mkdir()
            if command([SDK/'exec/qaic-runner','-t',base/'multi','-d',a.device,'--aic-batch-json-input',config,
                        '-n','3','-S','1','-T','1','-c','--aic-profiling-type','raw_device_stats',
                        '--aic-profiling-num-samples','1','--aic-profiling-out-dir',stats],subdir/'runner.log'):
                raise RuntimeError('Device profiling failed')
            traces=subdir/'trace';traces.mkdir()
            if command([SDK/'exec/qaic-opstats','--qpc',base/'multi/programqpc.bin','--input-dir',stats,
                        '--output-dir',traces,'--summary','--trace','--merge-mq-traces','true','--flow-events','none'],subdir/'opstats.log'):
                raise RuntimeError('Trace conversion failed')
            print('PROFILE_OK',name,flush=True)
    for p in info['profiles']:
        traces=base/'profiling'/p['name']/'trace'
        if traces.exists():report['profiles'][p['name']]['trace']=trace_summary(traces,graph_info['weight_gathers'])
    dump(base/'package_audit.json',report)
    print('PACKED_CONSTANTS_MIB',{k:round(v['static_constants_bytes']/2**20,4) for k,v in report['packages'].items()},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--stage',choices=['export','compile','sharing','run','analyze','audit','all'],default='all')
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16')
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--device',type=int,default=0)
    p.add_argument('--run-name',default='run');p.add_argument('--equivalence-only',action='store_true')
    p.add_argument('--profile',action='store_true');a=p.parse_args();a.out=a.out.resolve()
    if a.iterations<1:p.error('Iterations must be positive')
    if a.stage in ['export','all']:
        if a.source is None:p.error('--source resident-selection artifact root is required')
        export(a)
    info=json.loads((a.out/'info.json').read_text())
    if a.stage in ['compile','all']:compile_graph(a)
    if a.stage=='sharing':sharing(a)
    if a.stage in ['run','all']:run(a,info)
    if a.stage in ['run','analyze','all']:analyze(a,info)
    if a.stage=='audit' or (a.stage=='all' and a.profile):audit(a,info)


if __name__=='__main__':main()
