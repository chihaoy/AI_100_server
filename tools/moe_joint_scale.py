#!/usr/bin/env python3
"""Joint MoE controls and bounded single-card scaling before multi-card tests."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
from onnx.external_data_helper import set_external_data
import torch
import torch.nn.functional as F
from QEfficient.base.onnx_transforms import CustomOpTransform

from moe_resident_selection import command, compare, dump
from moe_tier_bench_export import MoELayerEP, CtxGatherFunc3DGeneralized, CtxScatterFunc3DGeneralized

SDK = Path('/opt/qti-aic')
TOKENS = [64,128,256]
NAMES = ['full4','small4','medium4','small8','medium8','mixed']
H, I = 2048, 768


def layout(experts, name, tokens):
    if name == 'mixed':
        return [4,4,8]+[16]*((experts-16)//16), [tokens,tokens,tokens//4]+[16]*((experts-16)//16)
    width = 8 if name.endswith('8') else 4
    capacity = tokens if name == 'full4' else (16 if name.startswith('small') else 32)
    return [width]*(experts//width), [capacity]*(experts//width)


def tensor_info(name, dtype, shape):
    return h.make_tensor_value_info(name,dtype,shape)


def make_weights(root, experts, offset=0, layer=0):
    """Global seeds make all card-local banks exact slices of the 128-bank control."""
    path = root/f'weights_l{layer}.bin'
    if path.exists():
        raise RuntimeError(f'Will not overwrite {path}')
    weights, initializers, hashes = {}, [], {}
    with path.open('wb') as f:
        for projection,name,shape in [(0,'Wg',(experts,H,I)),(1,'Wu',(experts,H,I)),(2,'Wd',(experts,I,H))]:
            generator = torch.Generator().manual_seed(3710+100*layer+projection)
            full = (torch.randn(128,*shape[1:],generator=generator)*0.02).half()
            values = full[offset:offset+experts].contiguous().numpy().copy()
            position = f.tell()
            values.tofile(f)
            hashes[name] = hashlib.sha256(values.tobytes()).hexdigest()
            weights[name] = torch.from_numpy(values)
            init = onnx.TensorProto(name=name,data_type=TP.FLOAT16,dims=shape)
            init.raw_data = b'x'
            set_external_data(init,path.name,offset=position,length=values.nbytes)
            init.ClearField('raw_data')
            initializers.append(init)
        router = torch.zeros(128,H,dtype=torch.float16)
        router[:,:128] = -4
        perm = torch.randperm(128,generator=torch.Generator().manual_seed(3790+layer))
        for token in range(128):
            router[perm[(token+torch.arange(8)*17)%128],token] = 2
        values = router.numpy()
        position = f.tell()
        values.tofile(f)
        weights['router'] = router
        hashes['router'] = hashlib.sha256(values.tobytes()).hexdigest()
        init = onnx.TensorProto(name='router',data_type=TP.FLOAT16,dims=[128,H])
        init.raw_data = b'x'
        set_external_data(init,path.name,offset=position,length=values.nbytes)
        init.ClearField('raw_data')
        initializers.append(init)
    return weights,initializers,hashes


class Router(torch.nn.Module):
    def __init__(self,weights,experts,offset):
        super().__init__()
        self.router=torch.nn.Parameter(weights['router'],requires_grad=False)
        self.K,self.E_router,self.experts,self.offset=8,128,experts,offset

    def forward(self,x):
        rw=MoELayerEP.route(self,x)
        return rw,(rw>0).sum(0).to(torch.int32)[self.offset:self.offset+self.experts]


class ExportMatMul(torch.autograd.Function):
    """Emit an ordinary ONNX MatMul without computing dense padded GEMMs during tracing.

    Numerical references use the separate sparse FP32 implementation below.
    """
    @staticmethod
    def forward(ctx,a,b):
        return a.new_zeros((*a.shape[:-1],b.shape[-1]))

    @staticmethod
    def symbolic(g,a,b):
        return g.op('MatMul',a,b)


class Experts(torch.nn.Module):
    def __init__(self,weights,experts,offset,name):
        super().__init__()
        self.experts,self.offset,self.name=experts,offset,name
        self.capacity_check='oob'

    def forward(self,x,route_weights,expert_ids,Wg,Wu,Wd):
        tokens=x.shape[0]
        widths,capacities=layout(self.experts,self.name,tokens)
        lanes=x.new_zeros((max(widths),tokens,H))
        start=0
        for width,capacity in zip(widths,capacities):
            ids=expert_ids[start:start+width]
            start+=width
            routes=torch.index_select(route_weights.transpose(0,1),0,ids+self.offset)
            active=routes>0
            matched=MoELayerEP.matched_idx(self,active,capacity)
            packed=CtxGatherFunc3DGeneralized.apply(x.unsqueeze(0).expand(width,-1,-1),matched)
            wg,wu,wd=[torch.index_select(bank,0,ids) for bank in [Wg,Wu,Wd]]
            if torch.onnx.is_in_onnx_export():
                z=ExportMatMul.apply(F.silu(ExportMatMul.apply(packed,wg))*ExportMatMul.apply(packed,wu),wd)
            else:
                z=(F.silu(packed@wg)*(packed@wu))@wd
            rw=CtxGatherFunc3DGeneralized.apply(routes.unsqueeze(-1),matched)
            old=CtxGatherFunc3DGeneralized.apply(lanes[:width],matched)
            valid=torch.arange(capacity,dtype=torch.int32)[None,:]<active.sum(1,keepdim=True)
            update=torch.where(valid.unsqueeze(-1),old+z*rw,torch.zeros_like(z))
            updated=CtxScatterFunc3DGeneralized.apply(lanes[:width],matched,update)
            lanes=torch.cat([updated,lanes[width:]],dim=0) if width<max(widths) else updated
        return lanes.sum(0)


def export_torch(module,args,path,input_names,output_names,axes):
    if not path.exists():
        with torch.no_grad():
            torch.onnx.export(module.eval(),args,str(path),input_names=input_names,output_names=output_names,
                dynamic_axes=axes,export_params=False,do_constant_folding=False,dynamo=False,opset_version=17)
    model=onnx.load(path)
    CustomOpTransform.apply(model)
    return model


def rename_nodes(graph,prefix,replacements):
    mapping=dict(replacements)
    for node in graph.node:
        for name in node.output:
            mapping.setdefault(name,prefix+name)
    nodes=[]
    for node in graph.node:
        new=copy.deepcopy(node)
        new.name=prefix+node.name
        new.input[:]=[mapping.get(n,n) for n in node.input]
        new.output[:]=[mapping[n] for n in node.output]
        nodes.append(new)
    return nodes,mapping


def reference(x,weights,router):
    with torch.no_grad():
        rw,counts=router(x)
        y=torch.zeros_like(x,dtype=torch.float32)
        for e in range(router.experts):
            tokens=torch.nonzero(rw[:,e+router.offset]>0).flatten()
            if not len(tokens):
                continue
            z=x[tokens].float()
            z=(F.silu(z@weights['Wg'][e].float())*(z@weights['Wu'][e].float()))@weights['Wd'][e].float()
            y[tokens]+=z*rw[tokens,e+router.offset,None].float()
        return y.numpy(),counts.numpy(),rw.numpy()


def profiles(experts):
    result=[]
    for ti,tokens in enumerate(TOKENS):
        for ni,name in enumerate(NAMES):
            widths,caps=layout(experts,name,tokens)
            result.append(dict(name=name,tokens=tokens,tag=ti*len(NAMES)+ni+1,widths=widths,
                capacities=caps,padded_rows=sum(w*c for w,c in zip(widths,caps))))
    return result


def make_graphs(root,weights,inits,experts,offset):
    x=torch.zeros(128,H,dtype=torch.float16)
    x[torch.arange(128),torch.arange(128)]=1
    router=Router(weights,experts,offset)
    rw,_=router(x)
    rm=export_torch(router,(x,),root/'router_raw.onnx',['x'],['route_weights','counts'],
        {'x':{0:'T'},'route_weights':{0:'T'}})
    print('ROUTER_INPUTS',[v.name for v in rm.graph.input],flush=True)
    rn,_=rename_nodes(rm.graph,'routing/',{'x':'x','router':'router','route_weights':'route_weights','counts':'counts'})
    router_model=h.make_model(h.make_graph(rn,'router',[tensor_info('x',TP.FLOAT16,['T',H])],
        [tensor_info('route_weights',TP.FLOAT16,['T',128]),tensor_info('counts',TP.INT32,[experts])],
        [v for v in inits if v.name=='router']),opset_imports=list(rm.opset_import),ir_version=9)
    onnx.save(router_model,root/'router.onnx')
    branches={}
    functions={}
    opsets={v.domain:v.version for v in rm.opset_import}
    for name in NAMES:
        module=Experts(weights,experts,offset,name)
        # Explicit temporary graph inputs avoid feeding gigabytes of parameter values
        # into Torch's shape-inference passes. Parent graphs bind these to constants.
        m=export_torch(module,(x,rw,torch.arange(experts,dtype=torch.int32),weights['Wg'],weights['Wu'],weights['Wd']),root/f'raw_{name}.onnx',
            ['x','route_weights','expert_ids','Wg','Wu','Wd'],['y'],{'x':{0:'T'},'route_weights':{0:'T'},'y':{0:'T'}})
        print('BRANCH',name,'INPUTS',[v.name for v in m.graph.input],flush=True)
        nodes,mapping=rename_nodes(m.graph,name+'/',{'x':'x','route_weights':'route_weights',
            'expert_ids':'selected_ids','Wg':'Wg','Wu':'Wu','Wd':'Wd'})
        branches[name]=h.make_graph(nodes,name,[],[tensor_info(mapping['y'],TP.FLOAT16,['T',H])])
        for f in m.functions:
            functions[f.domain,f.name]=copy.deepcopy(f)
        opsets.update({v.domain:v.version for v in m.opset_import})
    constants=[nh.from_array(np.array(0,np.int64),'zero')]
    selector=[h.make_node('ReduceSum',['profile_tag'],['profile_sum'],keepdims=0),
        h.make_node('ReduceSum',['layout_tag'],['layout_sum'],keepdims=0),
        h.make_node('Add',['profile_sum','layout_sum'],['tag_sum']),
        h.make_node('Add',['expert_ids','tag_sum'],['selected_ids']),
        h.make_node('Shape',['layout_tag'],['tag_shape']),
        h.make_node('Gather',['tag_shape','zero'],['tag_length'],axis=0)]
    for ni,name in enumerate(NAMES[:-1]):
        key=f'layout_{ni+1}'
        constants.append(nh.from_array(np.array(ni+1,np.int64),key))
        selector.append(h.make_node('Equal',['tag_length',key],['is_'+name]))
    def branch(index):
        name=NAMES[index]
        if index==len(NAMES)-1:
            return branches[name]
        return h.make_graph([h.make_node('If',['is_'+name],['dispatch_'+name],name='dispatch_'+name,
            then_branch=branches[name],else_branch=branch(index+1))],name+'_dispatch',[],
            [tensor_info('dispatch_'+name,TP.FLOAT16,['T',H])])
    dispatch=branch(0).node[0]
    dispatch.output[:]=['y']
    inputs=[tensor_info('x',TP.FLOAT16,['T',H]),tensor_info('expert_ids',TP.INT32,[experts]),
        tensor_info('profile_tag',TP.INT32,['P']),tensor_info('layout_tag',TP.INT32,['W'])]
    outputs=[tensor_info('y',TP.FLOAT16,['T',H]),tensor_info('counts',TP.INT32,[experts])]
    count_init=[nh.from_array(np.array(0,np.float16),'count_zero'),nh.from_array(np.array([0],np.int64),'count_axis'),
        nh.from_array(np.array([offset],np.int64),'count_start'),nh.from_array(np.array([offset+experts],np.int64),'count_end')]
    count_nodes=[h.make_node('Greater',['route_weights','count_zero'],['count_active']),
        h.make_node('Cast',['count_active'],['count_int'],to=TP.INT32),
        h.make_node('ReduceSum',['count_int','count_axis'],['count_global'],keepdims=0),
        h.make_node('Slice',['count_global','count_start','count_end','count_axis'],['counts'])]
    for kind in ['fused','expert']:
        nodes=(rn if kind=='fused' else count_nodes)+selector+[dispatch]
        ins=inputs+([] if kind=='fused' else [tensor_info('route_weights',TP.FLOAT16,['T',128])])
        initializers=(inits if kind=='fused' else [v for v in inits if v.name!='router']+count_init)+constants
        model=h.make_model(h.make_graph(nodes,kind,ins,outputs,initializers),
            opset_imports=[h.make_opsetid(k,v) for k,v in opsets.items()],functions=list(functions.values()),ir_version=9)
        onnx.save(model,root/f'{kind}.onnx')
        onnx.checker.check_model(str(root/f'{kind}.onnx'))


def export(a):
    torch.set_num_threads(8)
    root=a.out.resolve()
    root.mkdir(parents=True,exist_ok=a.resume)
    (root/'inputs').mkdir(exist_ok=a.resume)
    (root/'reference').mkdir(exist_ok=a.resume)
    if a.resume:
        weights,inits,hashes={},[],{}
        position=0
        for key,shape in [('Wg',(a.experts,H,I)),('Wu',(a.experts,H,I)),('Wd',(a.experts,I,H)),('router',(128,H))]:
            values=np.memmap(root/'weights_l0.bin',dtype=np.float16,mode='c',offset=position,shape=shape)
            weights[key]=torch.from_numpy(values)
            hashes[key]=hashlib.sha256(values.tobytes()).hexdigest()
            init=onnx.TensorProto(name=key,data_type=TP.FLOAT16,dims=shape);init.raw_data=b'x'
            set_external_data(init,'weights_l0.bin',offset=position,length=values.nbytes);init.ClearField('raw_data');inits.append(init)
            position+=values.nbytes
    else:
        weights,inits,hashes=make_weights(root,a.experts,a.offset)
    make_graphs(root,weights,inits,a.experts,a.offset)
    ps=profiles(a.experts)
    dump(root/'specializations.json',{'specializations':[{'T':str(p['tokens']),'P':str(p['tag']),'W':str(NAMES.index(p['name'])+1)} for p in ps]})
    dump(root/'router_specializations.json',{'specializations':[{'T':str(t)} for t in TOKENS]})
    for variant,name in [('fixed','medium4'),('conservative','full4')]:
        p=next(p for p in ps if p['tokens']==128 and p['name']==name)
        dump(root/f'{variant}_specializations.json',{'specializations':[{'T':'128','P':str(p['tag']),'W':str(NAMES.index(p['name'])+1)}]})
    (root/'profiles.txt').write_text(''.join(f"{p['name']} {p['tokens']} {p['tag']} {len(p['widths'])} "+
        ' '.join(f'{w} {c}' for w,c in zip(p['widths'],p['capacities']))+'\n' for p in ps))
    router=Router(weights,a.experts,a.offset)
    identity=torch.zeros(128,H,dtype=torch.float16)
    identity[torch.arange(128),torch.arange(128)]=1
    routes,_=router(identity)
    hot=int((routes[:,a.offset:a.offset+a.experts]>0).sum(1).argmax())
    workloads=[]
    for tokens in TOKENS:
        for level,fraction in [('balanced',0),('warm',.125),('hot',.5),('concentrated',1)]:
            tokenids=torch.arange(tokens)%128
            tokenids[:int(tokens*fraction)]=hot
            x=(torch.randn(tokens,H,generator=torch.Generator().manual_seed(8800+tokens))*.05).half()
            x[:,:128]=0
            x[torch.arange(tokens),tokenids]=1
            name=f't{tokens}_{level}'
            y,counts,rw=reference(x,weights,router)
            ids=np.lexsort((np.arange(a.experts),-counts)).astype(np.int32)
            safe=[p for p in ps if p['tokens']==tokens and np.all(counts[ids]<=np.repeat(p['capacities'],p['widths']))]
            selected=min(safe,key=lambda p:(p['padded_rows'],p['tag']))
            x.numpy().tofile(root/'inputs'/f'{name}_x.bin')
            counts.tofile(root/'reference'/f'{name}_counts.bin')
            rw.tofile(root/'reference'/f'{name}_route_weights.bin')
            ids.tofile(root/'reference'/f'{name}_ids.bin')
            np.save(root/'reference'/f'{name}_y.npy',y)
            workloads.append(dict(name=name,tokens=tokens,level=level,peak=int(counts.max()),
                assignments=int(counts.sum()),selected_tag=selected['tag'],selected_name=selected['name']))
            print('REFERENCE',workloads[-1],flush=True)
    (root/'workloads.txt').write_text(''.join(f"{w['name']} {w['tokens']}\n" for w in workloads))
    dump(root/'info.json',dict(experts=a.experts,offset=a.offset,H=H,I=I,tokens=TOKENS,profiles=ps,
        workloads=workloads,bank_sha256=hashes,scope='Synthetic global top-8 MoE; runtime local permutation, widths and capacity; no attention or KV'))
    print('EXPORT_OK',root,flush=True)


def compile_graph(a):
    base=a.out/a.precision
    base.mkdir(exist_ok=True)
    for variant in a.variants:
        kind='router' if variant=='router1' else ('expert' if variant=='expert15' else 'fused')
        cores=1 if kind=='router' else (15 if variant.endswith('15') else 16)
        spec='router_' if kind=='router' else (variant+'_' if variant in ['fixed','conservative'] else '')
        dest=base/variant
        if dest.exists():
            raise RuntimeError(f'Will not overwrite {dest}')
        args=[SDK/'exec/qaic-compile','-aic-hw','-aic-hw-version=ai100',f'-m={a.out/f"{kind}.onnx"}',
            '-convert-to-fp16',f'-aic-num-cores={cores}','-mos=1','-aic-enable-depth-first','-stats-level=70',
            '-compile-only',f'-network-specialization-config={a.out/f"{spec}specializations.json"}',f'-aic-binary-dir={dest}']
        if a.precision=='mxfp6':
            args.append('-mxfp6-matmul')
        start=time.monotonic()
        status=command(args,base/f'{variant}_compile.log')
        dump(base/f'{variant}_compile.json',dict(exit_code=status,seconds=time.monotonic()-start,qpc_exists=(dest/'programqpc.bin').exists()))
        print('COMPILE',a.out,a.precision,variant,status,flush=True)
        if status:
            raise RuntimeError(f'Compilation failed: {variant}')


def run(a):
    base=a.out/a.precision/a.run_name
    base.mkdir(exist_ok=False)
    host=base/'host'
    if command(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',f'-I{SDK}/dev/inc',
        Path(__file__).with_name('moe_joint_host.cpp'),'-o',host,f'-L{SDK}/dev/lib/x86_64','-lQAic',
        f'-Wl,-rpath,{SDK}/dev/lib/x86_64'],base/'build.log'):
        raise RuntimeError('Host build failed')
    info=json.loads((a.out/'info.json').read_text())
    for mode in a.modes:
        if command([host,a.out,a.precision,base/mode,info['experts'],a.iterations,mode,info.get('layers',1)],base/f'{mode}.log'):
            raise RuntimeError(f'Host failed: {mode}')


def analyze(a):
    info=json.loads((a.out/'info.json').read_text())
    base=a.out/a.precision/a.run_name
    report={'scope':info['scope'],'precision':a.precision,'modes':{}}
    for mode in a.modes:
        dest=base/mode
        with (dest/'samples.csv').open() as f:
            rows=list(csv.DictReader(f))
        with (dest/'resources.csv').open() as f:
            resources=list(csv.DictReader(f))
        with (dest/'packages.csv').open() as f:
            packages=list(csv.DictReader(f))
        for package in packages:
            if 'packaged_constants_bytes' in package:
                package['sdk_constants_size32_sum']=package.pop('packaged_constants_bytes')
        ready=next(i for i,r in enumerate(resources) if r['stage']=='ready')
        mr=dict(samples=len(rows),resources=resources,packages=packages,
            resident_mib=(int(resources[0]['dram_free_kib'])-int(resources[ready]['dram_free_kib']))/1024,
            resources_stable=len({(r['dram_free_kib'],r['nsp_free']) for r in resources[ready:]})==1,
            cases={},latency=[])
        for path in dest.glob('*_y.bin'):
            key=path.name.removesuffix('_y.bin')
            workload,order_name,tag=key.rsplit('_',2)
            tag=int(tag.removeprefix('p'))
            w=next(w for w in info['workloads'] if w['name']==workload)
            profile=next(p for p in info['profiles'] if p['tag']==tag)
            y=np.fromfile(path,np.float16)
            reference_y=np.load(a.out/'reference'/f'{workload}_y.npy').reshape(-1)
            counts=np.fromfile(dest/f'{key}_counts.bin',np.int32)
            vr=dict(vs_cpu=compare(y,reference_y),counts_exact=np.array_equal(counts,np.fromfile(a.out/'reference'/f'{workload}_counts.bin',np.int32)))
            full=next(p for p in info['profiles'] if p['tokens']==w['tokens'] and p['name']=='full4')
            fullpath=dest/f"{workload}_{order_name}_p{full['tag']}_y.bin"
            if fullpath.exists():
                vr['vs_conservative']=compare(y,np.fromfile(fullpath,np.float16))
            else:
                control=base/'conservative'/f"{workload}_{order_name}_p{full['tag']}_y.bin"
                if control.exists():
                    vr['vs_conservative']=compare(y,np.fromfile(control,np.float16))
            fused=base/'fused15'/path.name
            if mode=='split' and fused.exists():
                vr['vs_fused']=compare(y,np.fromfile(fused,np.float16))
            vr['profile']=profile['name'];mr['cases'][key]=vr
        groups=sorted(set(tuple(r[k] for k in ['schedule','workload','order','policy','tag']) for r in rows))
        for group in groups:
            keys=['schedule','workload','order','policy','tag']
            selected=[r for r in rows if tuple(r[k] for k in keys)==group]
            values=dict(zip(keys,group));values['samples']=len(selected)
            for key in ['total_ms','router_ms','decision_ms','handoff_ms','bind_ms','expert_ms']:
                ts=[float(r[key]) for r in selected]
                values[key]=dict(median=float(np.median(ts)),p10=float(np.percentile(ts,10)),p90=float(np.percentile(ts,90)))
            mr['latency'].append(values)
        report['modes'][mode]=mr
    values=[v for m in report['modes'].values() for v in m['cases'].values()]
    report['mechanism_pass']=all(v['counts_exact'] and all(v[k]['relative_l2']<.005 for k in ['vs_conservative','vs_fused'] if k in v) for v in values) and all(m['resources_stable'] for m in report['modes'].values())
    report['reference_accuracy_pass']=all(v['vs_cpu']['relative_l2']<.05 for v in values)
    # Convert NumPy scalar booleans before the shared JSON writer.
    for v in values:
        v['counts_exact']=bool(v['counts_exact'])
    report['mechanism_pass']=bool(report['mechanism_pass'])
    dump(base/'result.json',report)
    print('RESULT',a.out,a.precision,report['mechanism_pass'],report['reference_accuracy_pass'],
        'max_cpu_l2',max(v['vs_cpu']['relative_l2'] for v in values),flush=True)
    if not report['mechanism_pass'] or (a.precision=='fp16' and not report['reference_accuracy_pass']):
        raise RuntimeError('Validation failed')


def audit(a):
    """Measure real segment lengths; SDK constantsInfo.size is only uint32_t."""
    base=a.out/a.precision
    info=json.loads((a.out/'info.json').read_text())
    result=json.loads((base/'audit.json').read_text()) if (base/'audit.json').exists() else {}
    for variant in a.variants:
        qpc=base/variant/'programqpc.bin'
        if not qpc.exists():
            raise RuntimeError(f'Missing QPC {qpc}')
        dest=base/'audit'/variant
        if (dest/'result.json').exists():
            result[variant]=json.loads((dest/'result.json').read_text())
            continue
        dest.mkdir(parents=True,exist_ok=a.resume)
        if command([SDK/'tools/qaic-qpc','extract','--qpc',qpc,'--output-dir',dest,
            '-s','*StaticConstants.constants.bin','-s','constants.bin','-s','*networkdesc.bin_dir/networkdesc.json'],dest/'extract.log'):
            raise RuntimeError('QPC extraction failed')
        segments={str(p.relative_to(dest)):p.stat().st_size for p in dest.rglob('*.bin')}
        descriptions=[json.loads(p.read_text()) for p in dest.rglob('networkdesc.json')]
        if len(descriptions)!=1:
            raise RuntimeError('Expected one network descriptor')
        desc=descriptions[0]
        allowed=[]
        for shape in desc.get('allowed_shapes',[]):
            allowed.append(dict(zip([v['name'] for v in desc['inputs']],[v['dims'] for v in shape['shapes']])))
        if not allowed:
            allowed=[{v['name']:v['io_initial']['dims'] for v in desc['inputs']}]
        expected=info['profiles']
        if variant in ['fixed','conservative']:
            expected=[p for p in expected if p['tokens']==128 and p['name']==('medium4' if variant=='fixed' else 'full4')]
        if variant=='router1':
            assert {s['x'][0] for s in allowed}==set(info['tokens'])
        else:
            assert {(s['x'][0],s['profile_tag'][0],s['layout_tag'][0]) for s in allowed}=={(p['tokens'],p['tag'],NAMES.index(p['name'])+1) for p in expected}
        result[variant]=dict(qpc_bytes=qpc.stat().st_size,segments_bytes=segments,allowed_shapes=allowed,
            scope='Actual extracted file lengths; unlike uint32_t SDK constantsInfo.size, valid above 4 GiB')
        dump(dest/'result.json',result[variant])
        # Retain the original QPC, exact lengths, extraction log and descriptor;
        # discard only the newly extracted duplicate binary payloads.
        for path in dest.rglob('*.bin'):
            path.unlink()
        print('AUDIT',variant,segments,flush=True)
    dump(base/'audit.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','compile','run','analyze','audit'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--experts',type=int,default=32,choices=[32,64,128])
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--resume',action='store_true',help='Resume an interrupted export from its bank and raw graph cache')
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16')
    p.add_argument('--variants',nargs='+',default=['router1','expert15','fused15','fixed','conservative'])
    p.add_argument('--modes',nargs='+',default=['split','fused15','fixed','conservative'])
    p.add_argument('--run-name',default='run')
    p.add_argument('--iterations',type=int,default=10)
    a=p.parse_args()
    a.out=a.out.resolve()
    {'export':export,'compile':compile_graph,'run':run,'analyze':analyze,'audit':audit}[a.action](a)


if __name__=='__main__':
    main()
