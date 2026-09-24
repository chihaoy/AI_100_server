#!/usr/bin/env python3
"""Sweep resident expert group width at fixed weights, capacities and padded work.

Uses the resident-selection probe's source inputs/references and C++ host.
Each width is a separate QPC, not runtime group-count specialization.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import onnx
import torch
from QEfficient.base.onnx_transforms import CustomOpTransform

from moe_resident_selection import ResidentMoE, CASES, command, compare, dump
from moe_resident_audit import trace_summary
from moe_capacity_audit import CtxScatter3DInt, CtxGather3D
from onnx.reference import ReferenceEvaluator
from onnx.utils import Extractor

SDK = Path('/opt/qti-aic')


def export(a):
    source=a.source.resolve(); original=json.loads((source/'info.json').read_text())
    root=a.out; root.mkdir(parents=True,exist_ok=False)
    for name in ['orders','reference']: shutil.copytree(source/name,root/name)
    shutil.copyfile(source/'x.bin',root/'x.bin')
    torch.set_num_threads(8)
    model=ResidentMoE(original['H'],original['I']).eval()
    hashes={name:hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()
            for name,t in [('gate',model.Wg),('up',model.Wu),('down',model.Wd)]}
    if hashes!=original['bank_sha256']: raise RuntimeError('Source weight bank differs from reconstruction')
    # Load just the small router initializer, without loading the external banks.
    source_graph=onnx.load(source/'dynamic/model.onnx',load_external_data=False)
    router_name=next(n.input[1] for n in source_graph.graph.node if n.name=='/MatMul')
    router=next(t for t in source_graph.graph.initializer if t.name==router_name)
    with torch.no_grad():
        model.router.copy_(torch.from_numpy(onnx.numpy_helper.to_array(router,base_dir=str(source/'dynamic')).copy()).T)
    counts=np.load(root/'reference/counts.npy')
    if counts.max()>a.capacity: raise RuntimeError('Capacity would drop assignments')
    x=torch.from_numpy(np.fromfile(root/'x.bin',np.float16).reshape(128,original['H']))
    ids=torch.from_numpy(np.fromfile(root/'orders/identity.bin',np.int32))
    info={**original,'source':str(source),'widths':a.widths,'capacity':a.capacity,
          'padded_expert_rows':32*a.capacity,'bank_sha256':hashes,
          'group_size':None,'capacities':None,
          'scope':'Single-card synthetic partial MoE; runtime IDs; static group topology; no KV or inter-card transfers'}
    dump(root/'info.json',info); audit={}
    for width in a.widths:
        variant=f'w{width}'; dest=root/variant; dest.mkdir()
        model.group_size=width; model.capacities=[a.capacity]*(32//width)
        path=dest/'model.onnx'
        with torch.no_grad():
            torch.onnx.export(model,(x,ids),str(path),input_names=['x','expert_ids'],
                              output_names=['y','counts'],dynamo=False,opset_version=17,do_constant_folding=True)
        graph=onnx.load(path); CustomOpTransform.apply(graph)
        initializers={t.name:list(t.dims) for t in graph.graph.initializer}
        gathers=[{'name':n.name,'bank':n.input[0]} for n in graph.graph.node
                 if n.op_type=='Gather' and n.input[0] in initializers and len(initializers[n.input[0]])==3]
        matmuls=[n.name for n in graph.graph.node if n.op_type=='MatMul' and n.name!='/MatMul']
        assert len(gathers)==len(matmuls)==3*32//width
        audit[variant]={'width':width,'groups':32//width,'capacity':a.capacity,
                        'logical_packed_shape':[width,a.capacity,original['H']],
                        'padded_expert_rows':32*a.capacity,'weight_gathers':gathers,'expert_matmuls':matmuls}
        onnx.save_model(graph,path,save_as_external_data=True,all_tensors_to_one_file=True,
                        location='weights.bin',size_threshold=1024)
        onnx.checker.check_model(str(path)); dump(root/'graph_audit.json',audit)
        print('EXPORT_OK',variant,flush=True)


def compile_graph(a,info):
    base=a.out/a.precision; base.mkdir(exist_ok=True); result={}
    for width in info['widths']:
        variant=f'w{width}'; dest=base/variant
        if dest.exists(): raise RuntimeError(f'Will not overwrite {dest}')
        args=[SDK/'exec/qaic-compile','-aic-hw','-aic-hw-version=ai100',
              f'-m={a.out/variant/"model.onnx"}','-convert-to-fp16','-aic-num-cores=16',
              '-mos=1','-aic-enable-depth-first','-stats-level=70','-compile-only',f'-aic-binary-dir={dest}']
        if a.precision=='mxfp6': args.append('-mxfp6-matmul')
        status=command(args,base/f'{variant}_compile.log')
        result[variant]={'exit_code':status,'qpc_exists':(dest/'programqpc.bin').exists()}
        dump(base/'compilation.json',result)


def available(a,info):
    results=json.loads((a.out/a.precision/'compilation.json').read_text())
    return [f'w{w}' for w in info['widths'] if results.get(f'w{w}',{}).get('exit_code')==0
            and results[f'w{w}']['qpc_exists']]


def run(a,info):
    base=a.out/a.precision/a.run_name; base.mkdir(exist_ok=False); host=base/'host'
    if command(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',f'-I{SDK}/dev/inc',
                Path(__file__).with_name('moe_resident_selection_host.cpp'),'-o',host,
                f'-L{SDK}/dev/lib/x86_64','-lQAic',f'-Wl,-rpath,{SDK}/dev/lib/x86_64'],base/'host_build.log'):
        raise RuntimeError('Host build failed')
    variants=available(a,info)
    if a.reverse: variants.reverse()
    dump(base/'run_config.json',{'order':variants,'iterations':a.iterations,'device':a.device})
    for variant in variants:
        if command([host,a.out/a.precision/variant,a.out,base/variant,a.device,info['H'],a.iterations,'dynamic'],
                   base/f'{variant}_host.log'): raise RuntimeError(f'Host failed: {variant}')


def analyze(a,info):
    base=a.out/a.precision/a.run_name; variants=available(a,info)
    if 'w16' not in variants: raise RuntimeError('Width 16 control is unavailable')
    reference_counts=np.load(a.out/'reference/counts.npy')
    baseline={c:np.fromfile(base/'w16'/f'{c}_y.bin',np.float16).reshape(128,info['H']) for c in CASES}
    report={'scope':info['scope'],'precision_flag':a.precision,'capacity':info['capacity'],
            'padded_expert_rows':info['padded_expert_rows'],'variants':{},'equivalence_only':a.equivalence_only}
    report['compilation']=json.loads((a.out/a.precision/'compilation.json').read_text())
    for variant in variants:
        directory=base/variant
        rows=np.atleast_1d(np.genfromtxt(directory/'samples.csv',delimiter=',',names=True,encoding=None,
              dtype=[('round','i4'),('mode','U8'),('case','U32'),('total_ms','f8')]))
        result={'width':int(variant[1:]),'cases':{},
                'metadata':json.loads((directory/'metadata.json').read_text())}
        result['groups']=32//result['width']
        result['pooled_timing']={}
        for mode in ['solo','cycle']:
            samples=rows[rows['mode']==mode]['total_ms']
            if len(samples): result['pooled_timing'][mode]={'median_ms':float(np.median(samples)),
                'p10_ms':float(np.percentile(samples,10)),'p90_ms':float(np.percentile(samples,90)),
                'samples':len(samples)}
        for case in CASES:
            y=np.fromfile(directory/f'{case}_y.bin',np.float16).reshape(128,info['H'])
            counts=np.fromfile(directory/f'{case}_counts.bin',np.int32)
            item={'vs_cpu':compare(y,np.load(a.out/'reference'/f'{case}.npy')),
                  'vs_width16':compare(y,baseline[case]),'counts_match':bool(np.array_equal(counts,reference_counts)),
                  'overflow_assignments':int(np.maximum(counts-info['capacity'],0).sum()),'timing':{}}
            for mode in ['solo','cycle']:
                samples=rows[(rows['mode']==mode)&(rows['case']==case)]['total_ms']
                if len(samples): item['timing'][mode]={'median_ms':float(np.median(samples)),
                    'p10_ms':float(np.percentile(samples,10)),'p90_ms':float(np.percentile(samples,90)),
                    'samples':len(samples)}
            result['cases'][case]=item
        identity=np.fromfile(directory/'identity_y.bin',np.float16)
        duplicate=np.fromfile(directory/'duplicate_control_y.bin',np.float16)
        result['diagnostic_changes_output']=compare(duplicate,identity)['relative_l2']>0.1
        result['grouping_equivalence_pass']=result['diagnostic_changes_output'] and all(
            p['counts_match'] and p['overflow_assignments']==0 and p['vs_width16']['relative_l2']<0.005
            for p in result['cases'].values())
        result['reference_accuracy_pass']=all(p['vs_cpu']['relative_l2']<0.05 for p in result['cases'].values())
        report['variants'][variant]=result
    report['grouping_equivalence_pass']=all(v['grouping_equivalence_pass'] for v in report['variants'].values())
    report['reference_accuracy_pass']=all(v['reference_accuracy_pass'] for v in report['variants'].values())
    report['validation_pass']=report['grouping_equivalence_pass'] and report['reference_accuracy_pass']
    dump(base/'result.json',report)
    for name,v in report['variants'].items():
        print(name,{c:round(p['timing']['solo']['median_ms'],4) for c,p in v['cases'].items() if p['timing']},
              'equivalence',v['grouping_equivalence_pass'],'accuracy',v['reference_accuracy_pass'],flush=True)
    if not report['grouping_equivalence_pass'] or (not a.equivalence_only and not report['validation_pass']):
        raise RuntimeError('Validation failed; see separate grouping and reference-accuracy checks')


def union_length(intervals):
    total=0.; end=float('-inf')
    for lo,hi in sorted(intervals):
        total+=max(0.,hi-max(lo,end)); end=max(end,hi)
    return total


def utilization(directory,graph):
    files=list(directory.glob('*trace.json'))
    if len(files)!=1: raise RuntimeError('Expected one trace sample')
    events=json.loads(files[0].read_text())['traceEvents']
    threads={(e['pid'],e['tid']):e['args']['name'] for e in events if e.get('name')=='thread_name'}
    kernels=[e for e in events if e.get('ph')=='X' and e.get('name')==e.get('args',{}).get('opKind')]
    start=min(e['ts'] for e in kernels); end=max(e['ts']+e.get('dur',0) for e in kernels)
    busy=defaultdict(list); ops=defaultdict(lambda:defaultdict(list)); names=set(graph['expert_matmuls'])
    for e in kernels:
        if e['name']!='aicconvolutiond32': continue
        op='/'+e['args']['opName'].split('/')[1]
        if op not in names: continue
        thread=threads[(e['pid'],e['tid'])]; match=re.search(r'_Core_(\d+)_HMX$',thread)
        if not match: raise RuntimeError(f'Cannot identify HMX core: {thread}')
        core=int(match[1]); interval=(e['ts'],e['ts']+e['dur'])
        busy[core].append(interval); ops[op][core].append(interval)
    if not busy: raise RuntimeError('No expert HMX kernel events')
    stages=[]
    for i in range(graph['groups']):
        selected=graph['expert_matmuls'][i*3:i*3+3]
        active=sorted({core for n in selected for core in ops[n]})
        intervals=[t for n in selected for v in ops[n].values() for t in v]
        stages.append({'stage':i,'active_hmx_cores':active,
                       'projection_hmx_cores':{n:sorted(ops[n]) for n in selected},
                       'first_expert_hmx_us':min(t[0] for t in intervals)-start,
                       'last_expert_hmx_us':max(t[1] for t in intervals)-start})
    durations={str(c):union_length(v) for c,v in sorted(busy.items())}
    transitions=[]
    for core,intervals in busy.items():
        for lo,hi in intervals:
            if hi>lo: transitions.extend([(lo,1,core),(hi,-1,core)])
    core_depth=defaultdict(int); peak=0
    for _,delta,core in sorted(transitions):
        core_depth[core]+=delta
        peak=max(peak,sum(depth>0 for depth in core_depth.values()))
    return {'device_kernel_span_us':end-start,'expert_hmx_active_cores':sorted(busy),
            'expert_hmx_busy_us_per_core':durations,
            'peak_simultaneous_expert_hmx_cores':peak,
            'expert_hmx_kernel_occupancy':sum(durations.values())/(16*(end-start)),
            'stages':stages,
            'note':'Union of expert HMX kernel durations / (16 cores * device kernel span); not FLOP utilization. Host time excluded.'}


def audit(a,info):
    base=a.out/a.precision; graphs=json.loads((a.out/'graph_audit.json').read_text())
    report={'variants':{},'scope':'Packed constants; single-sample trace occupancy; not PCIe accounting'}
    for variant in available(a,info):
        graph=onnx.load(a.out/variant/'model.onnx')
        first_matmuls=set(graphs[variant]['expert_matmuls'][::3])
        packed_outputs=[n.input[0] for n in graph.graph.node if n.name in first_matmuls]
        subgraph=Extractor(graph).extract_model(['x','expert_ids'],packed_outputs)
        evaluator=ReferenceEvaluator(subgraph,new_ops=[CtxScatter3DInt,CtxGather3D])
        packed=evaluator.run(None,{'x':np.fromfile(a.out/'x.bin',np.float16).reshape(128,info['H']),
                                  'expert_ids':np.fromfile(a.out/'orders/identity.bin',np.int32)})
        shapes=[list(x.shape) for x in packed]
        assert shapes==[graphs[variant]['logical_packed_shape']]*graphs[variant]['groups'],shapes
        qpc=base/variant/'programqpc.bin'; segments=base/f'{variant}_segments'
        if not segments.exists():
            if command([SDK/'tools/qaic-qpc','extract','--qpc',qpc,'--output-dir',segments,
                        '-s','*StaticConstants.constants.bin'],base/f'{variant}_extract.log'):
                raise RuntimeError('QPC extraction failed')
        constants=sum(p.stat().st_size for p in segments.rglob('StaticConstants.constants.bin'))
        if not constants: raise RuntimeError('No packed static constants found')
        entry={'logical_packed_shapes':shapes,'static_constants_bytes':constants,'qpc_bytes':qpc.stat().st_size}
        sub=base/'profiling'/variant
        if a.profile:
            sub.mkdir(parents=True,exist_ok=False)
            ios=[{'path':str(a.out/'x.bin'),'dims':[128,info['H']],'elem-size':2,'io-direction':'in','map-to':'x'},
                 {'path':str(a.out/'orders/identity.bin'),'dims':[32],'elem-size':4,'io-direction':'in','map-to':'expert_ids'}]
            for name,shape,size in [('y',[128,info['H']],2),('counts',[32],4)]:
                ios.append({'path':str(base/a.run_name/variant/f'identity_{name}.bin'),
                            'dims':shape,'elem-size':size,'io-direction':'out','map-to':name})
            config=sub/'io.json'; dump(config,{'IO-files':[ios]}); stats=sub/'stats'; stats.mkdir()
            if command([SDK/'exec/qaic-runner','-t',base/variant,'-d',a.device,'--aic-batch-json-input',config,
                        '-n','3','-S','1','-T','1','-c','--aic-profiling-type','raw_device_stats',
                        '--aic-profiling-num-samples','1','--aic-profiling-out-dir',stats],sub/'runner.log'):
                raise RuntimeError('Device profiling failed')
            traces=sub/'trace'; traces.mkdir()
            if command([SDK/'exec/qaic-opstats','--qpc',qpc,'--input-dir',stats,'--output-dir',traces,
                        '--summary','--trace','--merge-mq-traces','true','--flow-events','none'],sub/'opstats.log'):
                raise RuntimeError('Trace conversion failed')
        if (sub/'trace').exists():
            entry['trace']=trace_summary(sub/'trace',[n['name'] for n in graphs[variant]['weight_gathers']])
            entry['utilization']=utilization(sub/'trace',graphs[variant])
        report['variants'][variant]=entry; dump(base/'package_audit.json',report)
        print('AUDIT_OK',variant,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--source',type=Path)
    p.add_argument('--stage',choices=['export','compile','run','analyze','audit','all'],default='all')
    p.add_argument('--widths',default='32,16,8,4,2,1'); p.add_argument('--capacity',type=int,default=32)
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16')
    p.add_argument('--iterations',type=int,default=50); p.add_argument('--device',type=int,default=0)
    p.add_argument('--run-name',default='run'); p.add_argument('--reverse',action='store_true')
    p.add_argument('--equivalence-only',action='store_true'); p.add_argument('--profile',action='store_true')
    a=p.parse_args(); a.out=a.out.resolve(); a.widths=[int(w) for w in a.widths.split(',')]
    if len(set(a.widths))!=len(a.widths) or 16 not in a.widths or any(w not in [1,2,4,8,16,32] for w in a.widths):
        p.error('Widths must be unique divisors of 32 and include 16')
    if a.capacity<1 or a.capacity>128 or a.iterations<1: p.error('Invalid capacity/iterations')
    if a.stage in ['export','all']:
        if a.source is None: p.error('--source resident-selection artifact root is required for export')
        export(a)
    info=json.loads((a.out/'info.json').read_text())
    if a.stage in ['compile','all']: compile_graph(a,info)
    if a.stage in ['run','all']: run(a,info)
    if a.stage in ['run','analyze','all']: analyze(a,info)
    if a.stage=='audit' or (a.stage=='all' and a.profile): audit(a,info)


if __name__=='__main__': main()
