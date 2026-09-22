#!/usr/bin/env python3
"""All-experts-active placement/capacity oracle with unchanged trained weights."""
import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper as nh

import moe_qwen3_active_oracle as active
import moe_qwen3_overhead_ablation as overhead
from moe_qwen3_cold_capacity import SDK, command, compile_model, dump, load, resources, save_csv, sha

STEM=overhead.STEM
NAMES=[['onnx::MatMul_31532','onnx::MatMul_31533','onnx::MatMul_31534'],
       ['onnx::MatMul_31540','onnx::MatMul_31541','onnx::MatMul_31542']]
MATRIX_BYTES=2048*768*2
CPU_L2_LIMIT=.005  # Declared before hardware measurements; no token dropping.


def source_banks(source):
    model=onnx.load(source,load_external_data=False)
    values={v.name:v for v in model.graph.initializer}
    banks=[]
    for group in NAMES:
        row=[]
        for name in group:
            value=values[name]; info={e.key:e.value for e in value.external_data}
            if value.data_type!=onnx.TensorProto.FLOAT16 or value.dims[0]!=64:
                raise ValueError('Expected full FP16 trained banks')
            row.append(np.memmap(source.parent/info['location'],dtype=np.float16,mode='r',
                                 offset=int(info['offset']),shape=tuple(value.dims)))
        banks.append(row)
    return model,banks


def mask_for_counts(counts,seed):
    rng=np.random.default_rng(seed)
    residual=counts.copy(); mask=np.zeros((128,128),np.float16)
    for token in rng.permutation(128):
        # Bipartite Havel-Hakimi: equal row degree 8, prescribed column degrees.
        order=np.lexsort((rng.random(128),-residual))[:8]
        if np.any(residual[order]<=0):
            raise ValueError('Infeasible routing degree sequence')
        mask[token,order]=np.float16(.125); residual[order]-=1
    # Randomize co-occurrence after constructing the degree sequence. Each
    # 2x2 switch preserves every expert count and every token's top-8 degree.
    for _ in range(8192):
        a,b=rng.choice(128,2,replace=False)
        left=np.flatnonzero((mask[a]>0)&(mask[b]==0))
        right=np.flatnonzero((mask[b]>0)&(mask[a]==0))
        if not len(left) or not len(right): continue
        e,f=rng.choice(left),rng.choice(right)
        mask[a,e]=mask[b,f]=0
        mask[a,f]=mask[b,e]=np.float16(.125)
    if np.any(residual) or not np.all((mask>0).sum(1)==8) or not np.array_equal((mask>0).sum(0),counts):
        raise ValueError('Routing synthesis failed')
    return mask


def rounded(value):
    return 1 << (int(value)-1).bit_length()


def prepare(args):
    root=args.out
    root.mkdir(parents=True,exist_ok=False)
    (root/'workloads').mkdir(); (root/'banks').mkdir()
    model,banks=source_banks(args.source)
    captured=np.fromfile(args.cold/'input_f16.bin',np.float16).reshape(128,2176)
    rng=np.random.default_rng(20260923)
    hot={letter:np.sort(rng.choice(128,32,replace=False)) for letter in 'abc'}
    workloads=[]
    for index,(name,kind,key) in enumerate([('uniform','uniform',None),('mild_a','mild','a'),
             ('skew_a','skew','a'),('skew_b','skew','b'),('skew_c','skew','c')]):
        counts=np.full(128,8 if kind=='uniform' else 4 if kind=='mild' else 1,np.int32)
        if key:
            counts[hot[key]]=20 if kind=='mild' else 29
        routes=mask_for_counts(counts,20260923+index)
        data=np.concatenate([captured[:,:2048],routes],axis=1)
        path=root/'workloads'/name; path.mkdir()
        data.tofile(path/'input_f16.bin'); counts.tofile(path/'counts_i32.bin')
        workloads.append(dict(name=name,kind=kind,hot_set=key,counts=counts.tolist(),
            input_sha256=sha(path/'input_f16.bin'),active_experts=int((counts>0).sum()),
            assignments=int(counts.sum()),peak=int(counts.max()),minimum=int(counts.min())))
    orders={'identity':list(range(128)),
            'stripe':[int(f'{i:07b}'[::-1],2) for i in range(128)]}
    aggregate=np.array([w['counts'] for w in workloads]).sum(0)
    orders['aggregate']=np.lexsort((np.arange(128),-aggregate)).tolist()
    for key in hot:
        counts=np.array(next(w['counts'] for w in workloads if w['name']=='skew_'+key))
        orders['sorted_'+key]=np.lexsort((np.arange(128),-counts)).tolist()
    manifests={}
    original_hashes=[[hashlib.sha256(banks[e//64][projection][e%64].tobytes()).hexdigest()
                      for e in range(128)] for projection in range(3)]
    # A bank file stores projection-major arrays [128,H,I], [128,H,I], [128,I,H].
    for name,order in orders.items():
        if sorted(order)!=list(range(128)):
            raise ValueError('Permutation omits or duplicates trained experts')
        path=root/'banks'/f'{name}.bin'
        with path.open('xb') as stream:
            for projection in range(3):
                for expert in order:
                    stream.write(banks[expert//64][projection][expert%64].tobytes())
        # Verify every matrix after materialization, not just a few spot checks.
        with path.open('rb') as stream:
            for projection in range(3):
                for expert in order:
                    if hashlib.sha256(stream.read(MATRIX_BYTES)).hexdigest()!=original_hashes[projection][expert]:
                        raise ValueError('Trained weight permutation changed bytes')
            if stream.read(1): raise ValueError('Unexpected extra weight bytes')
        manifests[name]=dict(path=str(path),permutation=order,sha256=sha(path),bytes=path.stat().st_size,
                             matrices_verified=384)
        print('BANK_OK',name,flush=True)
    plans={}

    def plan(order_name,caps,reverse=False):
        name=f'{order_name}_{"reverse_" if reverse else ""}c{caps[0]}_{caps[1]}'
        order=orders[order_name]
        if reverse: order=order[64:]+order[:64]
        eligible=[w['name'] for w in workloads if np.all(np.array(w['counts'])[order]<=np.repeat(caps,64))]
        if not eligible: raise ValueError('Useless plan')
        plans[name]=dict(name=name,order=order_name,permutation=order,reverse=reverse,
                        capacities=caps,eligible=eligible,logical_weight_bytes=128*3*MATRIX_BYTES)
        return name

    static_orders=['identity','stripe','aggregate']
    for order_name in static_orders:
        plan(order_name,[32,32])
        for w in workloads:
            counts=np.array(w['counts'])[orders[order_name]].reshape(2,64)
            plan(order_name,[rounded(v) for v in counts.max(1)])
    # Exact-size controls keep the comparison from assuming powers of two win.
    plan('identity',[29,29]); plan('identity',[20,20])
    for key in hot:
        plan('sorted_'+key,[32,32])  # Pure placement: same graph shapes as common control.
        plan('sorted_'+key,[32,1]); plan('sorted_'+key,[1,32],True)
    plan('sorted_a',[32,4]); plan('sorted_a',[4,32],True)
    dump(root/'experiment.json',dict(source=str(args.source),source_sha256=sha(args.source),
         source_input_sha256=sha(args.cold/'input_f16.bin'),seed=20260923,
         cpu_relative_l2_limit=CPU_L2_LIMIT,workloads=workloads,banks=manifests,plans=plans,
         static_orders=static_orders,hot_sets={k:v.tolist() for k,v in hot.items()},
         hot_union=len(set(np.concatenate(list(hot.values())).tolist())),
         routing_randomization='Havel-Hakimi followed by 8192 count-preserving 2x2 switch attempts',
         scope='Synthetic top-8 routing, trained FP16 layer-2 weights and captured hidden input; all experts active'))
    for name,spec in plans.items():
        out=root/name; out.mkdir(); (out/'inputs').mkdir()
        changed=copy.deepcopy(model)
        (out/'weights.bin').symlink_to(Path(manifests[spec['order']]['path']))
        for value in changed.graph.initializer:
            if value.external_data:
                group,projection=next((g,p) for g in range(2) for p in range(3) if NAMES[g][p]==value.name)
                physical_group=1-group if spec['reverse'] else group
                info={'location':'weights.bin','offset':str((projection*128+physical_group*64)*MATRIX_BYTES),
                      'length':str(64*MATRIX_BYTES)}
                for entry in value.external_data: entry.value=info[entry.key]
            elif value.name in [f'probe_stage{g}_{suffix}' for g in range(2) for suffix in ['stop','end']]:
                group=int(value.name[len('probe_stage')]); old=nh.to_array(value)
                value.CopyFrom(nh.from_array(np.full_like(old,spec['capacities'][group]),value.name))
        if list(changed.graph.node)!=list(model.graph.node):
            raise ValueError('Placement experiment must preserve every source node')
        original={v.name:v for v in model.graph.initializer}
        for value in changed.graph.initializer:
            if not value.external_data and not value.name.startswith('probe_stage') and value!=original[value.name]:
                raise ValueError('Unexpected initializer change')
        onnx.save(changed,out/'model.onnx'); onnx.checker.check_model(str(out/'model.onnx'))
        for w in workloads:
            if w['name'] not in spec['eligible']: continue
            data=np.fromfile(root/'workloads'/w['name']/'input_f16.bin',np.float16).reshape(128,2176)
            remapped=np.concatenate([data[:,:2048],data[:,2048:][:,spec['permutation']]],axis=1)
            if not np.array_equal(remapped[:,2048:][:,np.argsort(spec['permutation'])],data[:,2048:]):
                raise ValueError('Routing permutation does not invert')
            remapped.tofile(out/'inputs'/f'{w["name"]}.bin')
        dump(out/'export.json',dict(**spec,graph_sha256=sha(out/'model.onnx'),all_original_nodes_unchanged=True,
             all_trained_weight_matrices_preserved=True,bank_sha256=manifests[spec['order']]['sha256']))
    print('PREPARE_OK',len(plans),'plans',sum(len(p['eligible']) for p in plans.values()),
          'eligible pairs','hot union',len(set(np.concatenate(list(hot.values())).tolist())),flush=True)


def reference(args):
    import torch
    torch.set_num_threads(8)
    info=load(args.out/'experiment.json'); _,banks=source_banks(args.source)
    workloads=info['workloads']
    data=[np.fromfile(args.out/'workloads'/w['name']/'input_f16.bin',np.float16).reshape(128,2176) for w in workloads]
    hidden=torch.from_numpy(data[0][:,:2048].copy()).float()
    outputs=[np.zeros((128,2048),np.float32) for _ in workloads]
    with torch.inference_mode():
        for expert in range(128):
            weights=[torch.from_numpy(np.array(banks[expert//64][p][expert%64],copy=True)).float() for p in range(3)]
            for source,out in zip(data,outputs):
                indices=np.flatnonzero(source[:,2048+expert]>0)
                if not len(indices): raise ValueError('Inactive expert in all-active reference')
                x=hidden[indices]; gate=x@weights[0]; up=x@weights[1]
                y=((gate*torch.sigmoid(gate))*up)@weights[2]
                out[indices]+=y.numpy()*source[indices,2048+expert,None].astype(np.float32)
    for w,out in zip(workloads,outputs):
        if not np.isfinite(out).all() or not np.linalg.norm(out): raise ValueError('Bad reference')
        path=args.out/'workloads'/w['name']; np.save(path/'reference_f32.npy',out)
        dump(path/'reference.json',dict(dtype='float32',trained_weights='exact FP16 source promoted to FP32',
            relative_l2_limit=CPU_L2_LIMIT,input_sha256=w['input_sha256'],output_sha256=sha(path/'reference_f32.npy')))
        print('REFERENCE_OK',w['name'],flush=True)


def qpc(args,name):
    return args.scratch/f'{name}_stats{args.stats_level}_compile/qpc'


def compile_cases(args):
    info=load(args.out/'experiment.json')
    for name in args.plans or info['plans']:
        if name not in info['plans']: raise ValueError(name)
        target=qpc(args,name).parent
        try:
            compile_model(args.out/name/'model.onnx',target,stats_level=args.stats_level)
        finally:
            for file in ['compile.log','compile.command.json','result.json']:
                if (target/file).exists(): shutil.copy2(target/file,args.out/name/f'stats{args.stats_level}_{file}')


def capacity2_controls(args):
    """Add a nondegenerate small shape after screening capacity 1."""
    info=load(args.out/'experiment.json'); added=[]
    for original in list(info['plans'].values()):
        if 1 not in original['capacities']: continue
        spec=copy.deepcopy(original); caps=[max(2,c) for c in spec['capacities']]
        name=original['name'].rsplit('_c',1)[0]+f'_c{caps[0]}_{caps[1]}'
        if name in info['plans']: continue
        spec.update(name=name,capacities=caps)
        out=args.out/name; out.mkdir()
        (out/'weights.bin').symlink_to((args.out/original['name']/'weights.bin').resolve())
        shutil.copytree(args.out/original['name']/'inputs',out/'inputs')
        model=onnx.load(args.out/original['name']/'model.onnx',load_external_data=False)
        for value in model.graph.initializer:
            if value.name in [f'probe_stage{g}_{s}' for g in range(2) for s in ['stop','end']]:
                group=int(value.name[len('probe_stage')]); old=nh.to_array(value)
                value.CopyFrom(nh.from_array(np.full_like(old,caps[group]),value.name))
        onnx.save(model,out/'model.onnx'); onnx.checker.check_model(str(out/'model.onnx'))
        dump(out/'export.json',dict(**spec,graph_sha256=sha(out/'model.onnx'),all_original_nodes_unchanged=True,
             all_trained_weight_matrices_preserved=True,bank_sha256=info['banks'][spec['order']]['sha256']))
        info['plans'][name]=spec; added.append(name)
    info['capacity2_extension']=added
    dump(args.out/'experiment.json',info)
    print('CAPACITY2_CONTROLS',*added,flush=True)


def merge_screen(args):
    components=[load(args.out/f'{name}_cohort.json') for name in ['screen','capacity2']]
    if any(any(c[k]!=components[0][k] for k in ['stats_level','rounds','outer_rounds','iterations']) for c in components):
        raise ValueError('Cannot merge different timing protocols')
    merged=copy.deepcopy(components[0]); merged['prefix']='combined'
    merged['components']=['screen','capacity2']
    merged['entries']=[e for c in components for e in c['entries']]
    keys=[(e['plan'],e['workload'],e['outer'],e['inner']) for e in merged['entries']]
    if len(set(keys))!=len(keys): raise ValueError('Duplicate measurements in combined screen')
    dump(args.out/'combined_cohort.json',merged)


def validate_output(args,plan,workload,path):
    info=load(args.out/'experiment.json'); spec=info['plans'][plan]
    w=next(w for w in info['workloads'] if w['name']==workload)
    counts=np.fromfile(path/'counts.bin',np.int32)
    expected=np.array(w['counts'],np.int32)[spec['permutation']]
    y=np.fromfile(path/'y.bin',np.float16).astype(np.float64)
    ref=np.load(args.out/'workloads'/workload/'reference_f32.npy').reshape(-1).astype(np.float64)
    if (not np.array_equal(counts,expected) or counts.min()<1 or counts.sum()!=1024
        or np.any(counts>np.repeat(spec['capacities'],64)) or y.shape!=ref.shape or not np.isfinite(y).all()):
        raise ValueError('Invalid counts, omitted experts, overflow, or nonfinite output')
    result=dict(counts_exact=True,all_128_active=True,no_overflow=True,
                relative_l2_cpu=float(np.linalg.norm(y-ref)/np.linalg.norm(ref)),max_abs_cpu=float(np.max(np.abs(y-ref))),
                output_sha256=sha(path/'y.bin'))
    dump(path/'validation.json',result)
    if result['relative_l2_cpu']>CPU_L2_LIMIT:
        raise ValueError(f'FP32 reference error exceeds preregistered limit: {result}')
    return result


def build_host(args):
    target=args.scratch/'multi_input_host'
    command(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',f'-I{SDK}/dev/inc',
        Path(__file__).with_name('moe_qwen3_multi_input_host.cpp'),'-o',target,
        f'-L{SDK}/dev/lib/x86_64','-lQAic',f'-Wl,-rpath,{SDK}/dev/lib/x86_64'],args.out/'host_build.log')


def timing(args):
    if args.stats_level!=0: raise ValueError('Host cohort requires profiling disabled')
    info=load(args.out/'experiment.json')
    pairs=(load(args.selection)['pairs'] if args.selection else
           [dict(plan=p,workload=w) for p,spec in info['plans'].items() for w in spec['eligible']])
    if args.plans: pairs=[p for p in pairs if p['plan'] in args.plans]
    plans=list(dict.fromkeys(p['plan'] for p in pairs))
    entries=[]
    for outer in range(args.outer_rounds):
        for plan in plans if outer%2==0 else plans[::-1]:
            selected=[p['workload'] for p in pairs if p['plan']==plan]
            root=args.out/plan; target=root/f'{args.prefix}_o{outer}'
            manifest=root/f'{args.prefix}_o{outer}_inputs.txt'
            manifest.write_text(''.join(f'{w} {root/"inputs"/f"{w}.bin"}\n' for w in selected))
            for w in selected:
                data=np.fromfile(root/'inputs'/f'{w}.bin',np.float16).reshape(128,2176)
                expected=np.array(next(v['counts'] for v in info['workloads'] if v['name']==w))[info['plans'][plan]['permutation']]
                if (not np.array_equal((data[:,2048:]>0).sum(0),expected) or expected.min()<1
                    or not np.isfinite(data).all() or np.any(data[:,2048:]<0)
                    or not np.all((data[:,2048:]>0).sum(1)==8)
                    or not np.all(data[:,2048:].sum(1)==1)):
                    raise ValueError('Invalid prepared input')
            resources(root/f'{args.prefix}_o{outer}_before.txt')
            command([args.scratch/'multi_input_host',qpc(args,plan),manifest,target,args.iterations,10,args.rounds],
                    root/f'{args.prefix}_o{outer}.log')
            resources(root/f'{args.prefix}_o{outer}_after.txt')
            for w in selected:
                medians=[]
                for inner in range(args.rounds):
                    path=target/w/f'r{inner}'; validation=validate_output(args,plan,w,path)
                    raw=load(path/'timing.json')['samples_ms']; medians.append(float(np.median(raw)))
                    entries.append(dict(plan=plan,workload=w,outer=outer,inner=inner,path=str(path),validation=validation))
                print('TIMING_OK',plan,w,outer,medians,flush=True)
            dump(args.out/f'{args.prefix}_cohort.json',dict(prefix=args.prefix,stats_level=0,rounds=args.rounds,
                 outer_rounds=args.outer_rounds,iterations=args.iterations,entries=entries))


def summary(args):
    cohort=load(args.out/f'{args.prefix}_cohort.json'); entries=cohort['entries']; rows=[]
    for plan,workload in sorted({(e['plan'],e['workload']) for e in entries}):
        selected=[e for e in entries if (e['plan'],e['workload'])==(plan,workload)]
        if len(selected)!=cohort['rounds']*cohort['outer_rounds']: raise ValueError('Incomplete cohort')
        raw=[load(Path(e['path'])/'timing.json')['samples_ms'] for e in selected]
        if any(len(r)!=cohort['iterations'] for r in raw): raise ValueError('Missing timing samples')
        saved={sha(Path(e['path'])/'y.bin') for e in selected}
        if len(saved)!=1: raise ValueError('Nonrepeatable same-graph output')
        samples=np.concatenate(raw)
        rows.append(dict(plan=plan,workload=workload,median_ms=float(np.median(samples)),
            p10_ms=float(np.percentile(samples,10)),p90_ms=float(np.percentile(samples,90)),
            round_medians_ms=[float(np.median(r)) for r in raw],samples=len(samples),saved_outputs=len(selected),
            relative_l2_cpu_max=max(e['validation']['relative_l2_cpu'] for e in selected)))
    dump(args.out/f'{args.prefix}.json',dict(cases=rows,cohort={k:v for k,v in cohort.items() if k!='entries'}))
    save_csv(args.out/f'{args.prefix}.csv',rows)
    print('SUMMARY_OK',args.prefix,len(rows),flush=True)


def select(args):
    info=load(args.out/'experiment.json'); rows=load(args.out/f'{args.prefix}.json')['cases']
    workloads=[w['name'] for w in info['workloads']]; plans=info['plans']

    def choose(candidates):
        return min(candidates,key=lambda r:(r['median_ms'],r['plan']))

    common={p:np.mean([r['median_ms'] for r in rows if r['plan']==p]) for p,s in plans.items()
            if set(s['eligible'])==set(workloads)}
    common_plan=min(common,key=lambda p:(common[p],p))
    fixed_shape={p:v for p,v in common.items() if plans[p]['capacities']==[32,32]}
    fixed_shape_plan=min(fixed_shape,key=lambda p:(fixed_shape[p],p))
    layout_candidates={}
    for order in info['banks']:
        selected=[]
        for w in workloads:
            selected.append(choose([r for r in rows if r['workload']==w and plans[r['plan']]['order']==order
                                    and not plans[r['plan']]['reverse']]))
        layout_candidates[order]=selected
    layout=min(layout_candidates,key=lambda order:np.mean([r['median_ms'] for r in layout_candidates[order]]))
    policies={
        'common_static':{w:common_plan for w in workloads},
        'fixed_shape_static':{w:fixed_shape_plan for w in workloads},
        'fixed_layout_capacity_oracle':{r['workload']:r['plan'] for r in layout_candidates[layout]},
        'fixed_shape_placement_oracle':{w:choose([r for r in rows if r['workload']==w and
                                      plans[r['plan']]['capacities']==[32,32]])['plan'] for w in workloads},
        'joint_oracle':{w:choose([r for r in rows if r['workload']==w])['plan'] for w in workloads}}
    pairs=[dict(plan=p,workload=w) for p,w in sorted({(p,w) for policy in policies.values() for w,p in policy.items()})]
    result=dict(screen_prefix=args.prefix,objective='Equal-weight arithmetic mean of per-workload median latency',
                policies=policies,fixed_layout=layout,pairs=pairs,
                common_candidate_scores_ms=common,
                fixed_layout_scores_ms={o:float(np.mean([r['median_ms'] for r in rr])) for o,rr in layout_candidates.items()})
    dump(args.out/'selection.json',result)
    first=choose([r for r in rows if r['workload']=='skew_b' and plans[r['plan']]['order']=='sorted_b'
                  and not plans[r['plan']]['reverse'] and plans[r['plan']]['capacities'][1]<=2])['plan']
    last=choose([r for r in rows if r['workload']=='skew_b' and plans[r['plan']]['order']=='sorted_b'
                 and plans[r['plan']]['reverse'] and plans[r['plan']]['capacities'][0]<=2])['plan']
    representatives=['identity_c32_32','sorted_b_c32_32',first,last]
    profile_pairs=[dict(plan=p,workload=w) for p,w in dict.fromkeys([
        (common_plan,'skew_b'),('identity_c32_32','uniform'),('identity_c32_32','skew_b'),
        ('sorted_b_c32_32','skew_b'),(first,'skew_b'),(last,'skew_b')])]
    dump(args.out/'profile_selection.json',dict(pairs=profile_pairs,representatives=representatives))
    for policy,choices in policies.items():
        score=np.mean([next(r['median_ms'] for r in rows if (r['plan'],r['workload'])==(p,w)) for w,p in choices.items()])
        print('SELECT',policy,float(score),choices,flush=True)


def selected_pairs(args):
    if not args.selection: raise ValueError('Need explicit profile selection file')
    pairs=load(args.selection)['pairs']
    if args.plans: pairs=[p for p in pairs if p['plan'] in args.plans]
    return pairs


def profile(args):
    if args.stats_level!=70: raise ValueError('Profiling requires stats-level 70')
    entries=load(args.out/f'{args.prefix}_cohort.json')['entries']
    for pair in selected_pairs(args):
        plan,w=pair['plan'],pair['workload']; case=args.out/plan
        out=case/'profiles'/w; out.mkdir(parents=True,exist_ok=False)
        expected=Path(next(e['path'] for e in entries if e['plan']==plan and e['workload']==w))
        bindings=[]
        for key,path,dims,size,direction in [
            ('probe_input',case/'inputs'/f'{w}.bin',[128,2176],2,'in'),
            ('y',expected/'y.bin',[128,2048],2,'out'),('counts',expected/'counts.bin',[128],4,'out')]:
            if path.stat().st_size!=int(np.prod(dims))*size: raise ValueError('Unexpected IO size')
            bindings.append(dict(path=str(path),dims=dims,**{'elem-size':size,'io-direction':direction,'map-to':key}))
        dump(out/'io.json',{'IO-files':[bindings]})
        for folder in ['stats','outputs','trace']: (out/folder).mkdir()
        resources(out/'resources_before.txt')
        command([SDK/'exec/qaic-runner','-t',qpc(args,plan),'-D','0:1:2:3','--aic-batch-json-input',out/'io.json',
            '-n',5,'-S',1,'-T',1,'-c','--aic-profiling-type','raw_device_stats','--aic-profiling-start-iter',2,
            '--aic-profiling-num-samples',3,'--aic-profiling-out-dir',out/'stats','--write-output-start-iter',2,
            '--write-output-num-samples',3,'--write-output-dir',out/'outputs'],out/'runner.log')
        resources(out/'resources_after.txt')
        checked=Counter()
        for path in (out/'outputs').rglob('*'):
            if not path.is_file(): continue
            key={524288:'y',512:'counts'}.get(path.stat().st_size)
            if key:
                if path.read_bytes()!=(expected/f'{key}.bin').read_bytes():
                    raise ValueError('Profile output differs from uninstrumented same-graph output')
                checked[key]+=1
        if dict(checked)!={'y':3,'counts':3}: raise ValueError('Missing profile outputs')
        command([SDK/'exec/qaic-opstats','--qpc',qpc(args,plan)/'programqpc.bin','--input-dir',out/'stats',
            '--output-dir',out/'trace','--summary','--trace','--merge-mq-traces','true','--flow-events','full'],out/'opstats.log')
        dump(out/'validation.json',dict(bit_exact_to_own_timing=True,samples=3,reference=str(expected)))
        print('PROFILE_OK',plan,w,flush=True)


def analyze(args):
    if args.stats_level!=70: raise ValueError('Analyze profiling binaries')
    summaries=[]; metadata={}; signatures={}
    for pair in selected_pairs(args):
        plan,w=pair['plan'],pair['workload']; case=args.out/plan/'profiles'/w
        if plan not in metadata:
            metadata[plan]=active.metadata(args,plan,constants_root=args.out/'constant_segments')
            lookup,_=metadata[plan]
            # Exact operation descriptors, including placement, size and order.
            signatures[plan]=hashlib.sha256(repr(sorted(lookup.items())).encode()).hexdigest()
        lookup,package=metadata[plan]
        hmx_descriptors={}
        for stage,suffixes in [('stage0',['MatMul','MatMul_1','MatMul_2']),
                               ('stage1',['MatMul_3','MatMul_4','MatMul_5'])]:
            selected=[v for v in lookup.values() if active.canon(v['name']) in {STEM+s for s in suffixes}
                      and v['kind']=='aicconvolutiond32']
            hmx_descriptors[stage]=dict(descriptors=len(selected),output_size_histogram=dict(Counter(v['bytes'] for v in selected)))
        traces=sorted((case/'trace').glob('*merged*.trace.json'))
        if len(traces)!=3 or not load(case/'validation.json')['bit_exact_to_own_timing']:
            raise ValueError('Unvalidated profile')
        target=case/'analysis'; target.mkdir(exist_ok=True)
        samples=[]; all_rows={key:[] for key in ['work','p2p','projection_copies','cores']}
        for index,path in enumerate(traces):
            metrics,_,work,p2p=overhead.trace_metrics(path,allow_card0_reduction=True)
            metrics['graph_order_gap_ms']=metrics['cold_start_ms']-metrics['hot_end_ms']
            if metrics['hot_end_ms']<=metrics['cold_start_ms']:
                metrics['hmx_order']='group0_then_group1'
                metrics['between_groups_ms']=metrics['cold_start_ms']-metrics['hot_end_ms']
            elif metrics['cold_end_ms']<=metrics['hot_start_ms']:
                metrics['hmx_order']='group1_then_group0'
                metrics['between_groups_ms']=metrics['hot_start_ms']-metrics['cold_end_ms']
            else:
                raise ValueError('Expert groups overlap; inspect scheduling before classifying phase gaps')
            metrics['post_graph_group1_ms']=metrics['post_gemm_tail_ms']
            metrics['post_gemm_tail_ms']=metrics['final_end_ms']-max(metrics['hot_end_ms'],metrics['cold_end_ms'])
            copies,cores=active.extra_metrics(path,metrics,work,lookup)
            for field in ['node','port']:
                totals=Counter()
                for row in p2p: totals[row[field]]+=row['bytes']
                metrics['p2p_by_'+field+'_bytes']=dict(sorted(totals.items()))
            samples.append(dict(sample=index,trace=str(path),**metrics))
            for key,rows in [('work',work),('p2p',p2p),('projection_copies',copies),('cores',cores)]:
                all_rows[key].extend(dict(sample=index,**r) for r in rows)
        for key,rows in all_rows.items(): save_csv(target/f'{key}.csv',rows)
        chosen=sorted(samples,key=lambda s:s['device_ms'])[1]
        result=dict(plan=plan,workload=w,**chosen,samples=samples,package=package,
                    operation_descriptor_sha256=signatures[plan],hmx_descriptors=hmx_descriptors,
                    export=load(args.out/plan/'export.json'))
        dump(target/'summary.json',result); summaries.append(result)
        print('ANALYZE_OK',plan,w,chosen['device_ms'],chosen['p2p_bytes']/2**20,flush=True)
    dump(args.out/'profiles.json',dict(cases=summaries,operation_descriptor_signatures=signatures))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','reference','compile','build-host','timing','summary','select','profile','analyze',
                                   'capacity2-controls','merge-screen'])
    for key in ['out','scratch','source','cold']: p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--plans',nargs='+'); p.add_argument('--selection',type=Path)
    p.add_argument('--stats-level',type=int,choices=[0,70],default=0)
    p.add_argument('--rounds',type=int,default=3); p.add_argument('--outer-rounds',type=int,default=1)
    p.add_argument('--iterations',type=int,default=100); p.add_argument('--prefix',default='screen')
    args=p.parse_args()
    for key in ['out','scratch','source','cold']: setattr(args,key,getattr(args,key).resolve())
    args.scratch.mkdir(parents=True,exist_ok=True)
    if min(args.rounds,args.outer_rounds,args.iterations)<1: p.error('Positive timing counts required')
    {'prepare':prepare,'reference':reference,'compile':compile_cases,'build-host':build_host,
     'timing':timing,'summary':summary,'select':select,'profile':profile,'analyze':analyze,
     'capacity2-controls':capacity2_controls,'merge-screen':merge_screen}[args.action](args)


if __name__=='__main__': main()
