#!/usr/bin/env python3
"""Oracle omission of empty cold experts in the trained layer-2 replay."""
import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import re
import shutil

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as TP
from onnx.reference import ReferenceEvaluator
from onnx.utils import Extractor

import moe_qwen3_overhead_ablation as overhead
from moe_capacity_audit import CtxGather3D, CtxScatter3DInt
from moe_qwen3_cold_capacity import SDK, command, compile_model, dependencies, dump, load, save_csv, sha, tensor
from moe_qwen3_layer_detail import schema_from_sdk
from moe_qwen3_profile import events, canon

STEM = overhead.STEM
COLD = ['MatMul_3','MatMul_4','MatMul_5']


class CtxScatter3D(CtxScatter3DInt):
    """Same out-of-range skipping semantics, preserving the floating dtype."""


def config(name):
    if name == 'e64':
        return 64, False
    match = re.fullmatch(r'(e|ffn)(22|24|32)',name)
    if not match:
        raise ValueError(name)
    return int(match[2]), match[1]=='ffn'


def input_guard(args, name):
    width, _ = config(name)
    data = np.fromfile(args.cold/'input_f16.bin',np.float16).reshape(128,2176)
    counts = (data[:,2048:]>0).sum(0).astype(np.int32)
    expected = np.fromfile(args.cold/'expected_counts_i32.bin',np.int32)
    if (not np.isfinite(data).all() or np.any(data[:,2048:]<0) or not np.array_equal(counts,expected)
            or np.any(counts[64+width:]) or counts[64:].max()>2 or counts[:64].max()>128
            or not np.all((data[:,2048:]>0).sum(1)==8)):
        raise ValueError('Input does not satisfy the active-expert oracle guard')
    return dict(input_sha256=sha(args.cold/'input_f16.bin'),active_cold=np.flatnonzero(counts[64:]).tolist(),
                cold_slots=width,omitted_cold=list(range(width,64)),omitted_assignments=int(counts[64+width:].sum()))


def rewrite(model, width, ffn_only):
    before = copy.deepcopy(model)
    by_name = {n.name:n for n in model.graph.node}
    added, insert_after = [], defaultdict(list)

    def init(name, value, dtype=np.int64):
        name = 'active_oracle_'+name
        added.append(nh.from_array(np.asarray(value,dtype=dtype),name))
        return name

    axis, start, end = init('axis',[0]), init('start',[0]), init('end',[width])

    def prefix(value, output, name):
        return h.make_node('Slice',[value,start,end,axis],[output],name=STEM+'active_'+name)

    weights = []
    weight_names = {by_name[STEM+n].input[1] for n in COLD}
    for value in model.graph.initializer:
        if value.name in weight_names:
            original = copy.deepcopy(value)
            if list(value.dims) not in [[64,2048,768],[64,768,2048]] or value.data_type != TP.FLOAT16:
                raise ValueError('Unexpected trained cold weight bank')
            value.dims[0] = width
            external = {e.key:e.value for e in value.external_data}
            if int(external['length']) != 64*2048*768*2:
                raise ValueError('Unexpected external weight storage')
            for item in value.external_data:
                if item.key == 'length':
                    item.value = str(width*2048*768*2)
            weights.append(dict(name=value.name,original_shape=list(original.dims),shape=list(value.dims),
                external_before={e.key:e.value for e in original.external_data},
                external_after={e.key:e.value for e in value.external_data}))
        elif not ffn_only and value.name.startswith(STEM+'CumSum_1_retune_zero_'):
            old = nh.to_array(value)
            if old.shape[0]!=64 or np.any(old):
                raise ValueError('Unexpected cold scan zero tensor')
            value.CopyFrom(nh.from_array(old[:width].copy(),value.name))
    if len(weights)!=3:
        raise ValueError('Need three cold weight banks')

    if ffn_only:
        gather = by_name[STEM+'CtxGather3D_3']
        packed = 'active_oracle_ffn_input'
        insert_after[gather.name].append(prefix(gather.output[0],packed,'ffn_input'))
        for name in COLD[:2]:
            by_name[STEM+name].input[0] = packed
        down = by_name[STEM+'MatMul_5']
        output = down.output[0]
        down.output[0] = 'active_oracle_ffn_output'
        zeros = init('ffn_tail',np.zeros((64-width,2,2048)),np.float16)
        insert_after[down.name].append(h.make_node('Concat',[down.output[0],zeros],[output],
                                                axis=0,name=STEM+'active_restore_ffn_rows'))
    else:
        # Keep an independent full 64-expert count diagnostic. Even the omitted
        # routing columns remain observable; an invalid oracle input is rejected.
        for name in ['Gather_13','Gather_14']:
            node = by_name[STEM+name]
            output = node.output[0]
            node.output[0] = output+'_full'
            insert_after[node.name].append(prefix(node.output[0],output,name))
        cold_full = by_name[STEM+'Gather_13'].output[0]
        zero = init('zero',0,np.float16)
        full_counts = 'active_oracle_full_cold_counts'
        insert_after[STEM+'Gather_13'] += [
            h.make_node('Greater',[cold_full,zero],['active_oracle_full_mask'],name=STEM+'active_full_mask'),
            h.make_node('Cast',['active_oracle_full_mask'],['active_oracle_full_i32'],to=TP.INT32,
                        name=STEM+'active_full_i32'),
            h.make_node('Einsum',['active_oracle_full_i32'],[full_counts],equation='ij->i',
                        name=STEM+'active_full_counts')]
        by_name['probe_counts'].input[1] = full_counts
        # Only update the matching prefix of the existing hot accumulator. The
        # untouched tail must be restored, not zeroed: it holds hot contributions.
        hot = by_name[STEM+'CtxScatter3D'].output[0]
        hot_prefix, hot_tail = 'active_oracle_hot_prefix', 'active_oracle_hot_tail'
        insert_after[STEM+'CtxScatter3D'] += [prefix(hot,hot_prefix,'hot_prefix'),
            h.make_node('Slice',[hot,end,init('full_end',[64]),axis],[hot_tail],name=STEM+'active_hot_tail')]
        by_name[STEM+'CtxGather3D_5'].input[0] = hot_prefix
        scatter = by_name[STEM+'CtxScatter3D_1']
        scatter.input[0] = hot_prefix
        output = scatter.output[0]
        scatter.output[0] = 'active_oracle_updated_prefix'
        insert_after[scatter.name].append(h.make_node('Concat',[scatter.output[0],hot_tail],[output],axis=0,
                                                     name=STEM+'active_restore_accumulator'))
    nodes = []
    for node in model.graph.node:
        nodes.append(copy.deepcopy(node))
        nodes.extend(insert_after[node.name])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(added)
    after = {n.name:n for n in model.graph.node}
    changed = [n.name for n in before.graph.node if n != after[n.name]]
    for node in before.graph.node:
        # All hot expert math, capacities and final reduction nodes are protected.
        if node.name in [STEM+s for s in ['MatMul','MatMul_1','MatMul_2','Slice','Slice_1','Range_1','Range_3',
                                         'Einsum_3','Einsum_4']] or node.name.startswith(STEM+'reduce_'):
            if node != after[node.name]:
                raise ValueError('Protected hot/capacity/reduction node changed')
    after_init = {v.name:v for v in model.graph.initializer}
    for value in before.graph.initializer:
        if value.name in weight_names or (not ffn_only and value.name.startswith(STEM+'CumSum_1_retune_zero_')):
            continue
        if value != after_init[value.name]:
            raise ValueError('Unexpected original initializer change')
    return dict(weights=weights,changed_original_nodes=changed,ffn_only=ffn_only,
                cold_weight_bytes=width*3*2048*768*2)


def semantic_check(original, candidate, cold):
    """Evaluate actual cold graphs with small deterministic FP16 banks and hot state."""
    rng = np.random.default_rng(20260922)
    banks = {}
    for value in original.graph.initializer:
        if value.external_data:
            shape = [64,8,4] if value.dims[1]==2048 else [64,4,8]
            banks[value.name] = (rng.standard_normal(shape)*.1).astype(np.float16)
    evaluators = []
    for source in [original,candidate]:
        model = copy.deepcopy(source)
        for value in model.graph.initializer:
            if value.external_data:
                value.CopyFrom(nh.from_array(banks[value.name][:value.dims[0]].copy(),value.name))
            elif value.name in ['probe_hidden','probe_end']:
                value.CopyFrom(nh.from_array(np.array([8 if value.name=='probe_hidden' else 136],np.int64),value.name))
            elif value.name=='active_oracle_ffn_tail':
                value.CopyFrom(nh.from_array(np.zeros((value.dims[0],2,8),np.float16),value.name))
        model.graph.input[0].CopyFrom(tensor('probe_input',TP.FLOAT16,[128,136]))
        hot, output = STEM+'CtxScatter3D_output_0', STEM+'CtxScatter3D_1_output_0'
        model.graph.value_info.extend([tensor(hot,TP.FLOAT16,[64,128,8]),tensor(output,TP.FLOAT16,[64,128,8])])
        subset = Extractor(model).extract_model(['probe_input',hot],[output,'counts'])
        evaluators.append(ReferenceEvaluator(subset,new_ops=[CtxGather3D,CtxScatter3DInt,CtxScatter3D]))
    routes = np.fromfile(cold/'input_f16.bin',np.float16).reshape(128,2176)[:,2048:]
    checks = []
    for kind in ['captured','empty_cold','random_cold']:
        mask = routes.copy()
        if kind!='captured':
            mask[:,64:]=0
        if kind=='random_cold':
            for e in range(22):
                positions = rng.choice(128,size=int(rng.integers(0,3)),replace=False)
                mask[positions,64+e] = rng.uniform(.01,.8,len(positions)).astype(np.float16)
        inputs = {'probe_input':np.concatenate([rng.standard_normal((128,8)).astype(np.float16),mask],axis=1),
                  hot:rng.standard_normal((64,128,8)).astype(np.float16)}
        expected, actual = [e.run(None,inputs) for e in evaluators]
        if any(not np.array_equal(a,b) for a,b in zip(expected,actual)):
            raise ValueError(f'Cold branch CPU mismatch: {kind}')
        checks.append(dict(case=kind,cold_accumulator_exact=True,full_counts_exact=True))
    return checks


def export(args,name):
    width, ffn_only = config(name)
    guard = input_guard(args,name)
    out = args.out/name
    out.mkdir(exist_ok=False)
    source = args.source/'c2_tree/model.onnx'
    model = onnx.load(source,load_external_data=False)
    original = copy.deepcopy(model)
    changes = rewrite(model,width,ffn_only) if width!=64 else dict(cold_weight_bytes=64*3*2048*768*2)
    checks = semantic_check(original,model,args.cold) if width!=64 else []
    dependencies(out,source.parent)
    onnx.save(model,out/'model.onnx')
    onnx.checker.check_model(str(out/'model.onnx'))
    if width==64 and sha(out/'model.onnx')!=sha(source):
        raise ValueError('Control serialization changed')
    dump(out/'export.json',dict(case=name,source=str(source),source_sha256=sha(source),
         graph_sha256=sha(out/'model.onnx'),hot_capacity=128,cold_capacity=2,
         **guard,changes=changes,cpu_checks=checks))
    print('EXPORT_OK',name,flush=True)


def qpc(args,name):
    if name=='e64':
        return args.source_scratch/f'c2_tree_stats{args.stats_level}_compile/qpc'
    return args.scratch/f'{name}_stats{args.stats_level}_compile/qpc'


def compile_case(args,name):
    out = args.out/name
    if name=='e64':
        path = qpc(args,name)/'programqpc.bin'
        dump(out/f'stats{args.stats_level}_reuse.json',dict(qpc=str(path),qpc_sha256=sha(path),
             graph_sha256=sha(out/'model.onnx')))
        return
    target = qpc(args,name).parent
    try:
        compile_model(out/'model.onnx',target,stats_level=args.stats_level)
    finally:
        for filename in ['compile.log','compile.command.json','result.json']:
            if (target/filename).exists():
                shutil.copy2(target/filename,out/f'stats{args.stats_level}_{filename}')


def metadata(args,name):
    case=args.out/name
    target=case/'metadata'
    if not target.exists():
        command([SDK/'tools/qaic-qpc','extract','--qpc',qpc(args,name)/'programqpc.bin',
                 '--output-dir',target,'-s','*opstatsdesc.bin'],case/'metadata_extract.log')
    _,cls=schema_from_sdk(SDK/'exec/qaic-opstats')
    paths=sorted(target.glob('QAicGraph_slice*_dir/opstatsdesc.bin'))
    if len(paths)!=4:
        raise ValueError('Expected four card descriptors')
    lookup={}
    for path in paths:
        card=int(re.search(r'slice(\d+)',str(path))[1])
        obj=cls(); obj.ParseFromString(path.read_bytes())
        if (obj.major_version,obj.minor_version)!=(1,11):
            raise ValueError('Unreviewed SDK descriptor version')
        common=obj.opstats_metadata.common_metadata.op_details
        kinds={p.id:p.str.strip() for p in common.op_kind_id_str_pairs}
        memories={p.id:p.str for p in common.op_memory_id_str_pairs}
        for spec in obj.opstats_metadata.specialization_metadata:
            for core,groups in enumerate(spec.core_op_name_kinds):
                for group in groups.tg_op_name_kinds:
                    for op in group.op_name_kinds:
                        key=(card,core,int(op.oc))
                        if key in lookup:
                            raise ValueError('Duplicate descriptor key')
                        lookup[key]=dict(name=op.name,bytes=int(op.output_size),
                                         kind=kinds[op.kind_id],memory=memories[op.memory_id])
    constants=args.scratch/f'{name}_constant_segments'
    if not constants.exists():
        command([SDK/'tools/qaic-qpc','extract','--qpc',qpc(args,name)/'programqpc.bin',
                 '--output-dir',constants,'-s','*StaticConstants.constants.bin'],case/'constants_extract.log')
    segments=sorted(constants.rglob('StaticConstants.constants.bin'))
    if len(segments)!=4:
        raise ValueError('Expected four static constant segments')
    package=dict(qpc=str(qpc(args,name)),qpc_bytes=(qpc(args,name)/'programqpc.bin').stat().st_size,
                 total_constant_bytes=sum(p.stat().st_size for p in segments),
                 segments=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in segments])
    dump(case/'package.json',package)
    return lookup,package


def extra_metrics(path,metrics,work,lookup):
    threads={(e['pid'],e['tid']):e['args']['name'] for e in events(path,metadata_only=True)
             if e['name']=='thread_name'}
    copies,waits=[],defaultdict(list)
    pending,matched={},0
    names={stage:{STEM+n for n in suffixes} for stage,suffixes in
           [('hot',['MatMul','MatMul_1','MatMul_2']),('cold',COLD)]}
    for e in events(path):
        if e.get('ph')!='X':
            continue
        match=overhead.CORE.search(threads.get((e['pid'],e['tid']),''))
        if not match or not match[3]:
            continue
        card,core,engine=int(match[1]),int(match[2]),match[3]
        a=e.get('args',{}); node=canon(a.get('opName',''))
        stage=next((s for s in names if node in names[s]),None)
        if stage is None:
            continue
        kind=a.get('opKind','').strip()
        if engine in ['DMA','DMAIssue_DMA'] and kind=='aiccopytovtcm' and a.get('opMemory')=='DDR' and not e['name'].startswith('sync '):
            meta=lookup[card,core,int(a['oc'])]
            if meta['name']!=a['opName'] or meta['kind']!=kind or meta['memory']!='DDR':
                raise ValueError('Trace/descriptor mismatch')
            copies.append(dict(stage=stage,node=node,card=card,core=core,oc=int(a['oc']),bytes=meta['bytes'],
                               start_us=float(e['ts']),duration_us=float(e.get('dur',0))))
        if engine=='HMX' and kind=='aicconvolutiond32':
            key=(card,core,a['oc'],a['opName'])
            if e['name']=='sync HMX':
                start=float(e['ts']); end=start+float(a['opSyncDurUs'])
                pending[key]=end
                lo,hi=1000*metrics[stage+'_start_ms'],1000*metrics[stage+'_end_ms']
                if min(end,hi)>max(start,lo):
                    waits[stage,card,core].append((max(start,lo),min(end,hi)))
            elif e['name']=='aicconvolutiond32':
                if key not in pending or abs(pending.pop(key)-float(e['ts']))>1:
                    raise ValueError('HMX dependency wait does not lead to matching kernel')
                matched+=1
    if pending or not matched:
        raise ValueError('Incomplete HMX wait pairing')
    per_core=[]
    for stage in names:
        selected=[r for r in work if r['node'] in names[stage] and r['engine']=='HMX'
                  and r['kind']=='aicconvolutiond32']
        keys=sorted({(r['card'],r['core']) for r in selected})
        projection_copies=[r for r in copies if r['stage']==stage]
        if not projection_copies:
            raise ValueError('No projection DDR copies; inspect DMA engine naming')
        stats=[]
        for card,core in keys:
            kernels=[r for r in selected if (r['card'],r['core'])==(card,core)]
            intervals=sorted(waits[stage,card,core])
            if any(b>c+.01 for (_,b),(c,_) in zip(intervals,intervals[1:])):
                raise ValueError('Overlapping HMX dependency waits')
            item=dict(stage=stage,card=card,core=core,hmx_events=len(kernels),
                 hmx_ms=sum(r['duration_us'] for r in kernels)/1000,
                 wait_ms=sum(b-a for a,b in intervals)/1000,
                 projection_ddr_bytes=sum(r['bytes'] for r in projection_copies if (r['card'],r['core'])==(card,core)))
            per_core.append(item); stats.append(item)
        for field in ['hmx_ms','wait_ms','projection_ddr_bytes']:
            metrics[f'{stage}_{field}_max_core']=max(r[field] for r in stats)
            metrics[f'{stage}_{field}_median_core']=float(np.median([r[field] for r in stats]))
        metrics[stage+'_projection_ddr_bytes']=sum(r['bytes'] for r in projection_copies)
        metrics[stage+'_hmx_events']=len(selected)
        metrics[stage+'_projection_copy_histogram']=[dict(bytes=size,events=count) for size,count in
                 sorted(Counter(r['bytes'] for r in projection_copies).items())]
    metrics['paired_hmx_wait_events']=matched
    return copies,per_core


def analyze(args):
    cases=[]
    for name in args.cases:
        case=args.out/name
        target=case/'analysis'; target.mkdir(exist_ok=True)
        lookup,package=metadata(args,name)
        traces=sorted((case/'profile/trace').glob('*merged*.trace.json'))
        if len(traces)!=3 or not load(case/'profile/validation.json')['bit_exact_to_own_timing']:
            raise ValueError('Need three validated profiler outputs')
        samples,all_work,all_p2p,all_copies,all_cores=[],[],[],[],[]
        for index,path in enumerate(traces):
            metrics,_,work,p2p=overhead.trace_metrics(path,require_full_expert_coverage=False,
                                                  allow_card0_reduction=True)
            if metrics['hot_cores']!=64:
                raise ValueError('Hot expert core coverage changed; inspect lowering')
            metrics['between_groups_ms']=metrics['cold_start_ms']-metrics['hot_end_ms']
            for field in ['node','port']:
                totals=Counter()
                for row in p2p:
                    totals[row[field]]+=row['bytes']
                metrics['p2p_by_'+field+'_bytes']=dict(sorted(totals.items()))
            copies,cores=extra_metrics(path,metrics,work,lookup)
            samples.append(dict(case=name,sample=index,trace=str(path),**metrics))
            for source,target_rows in [(work,all_work),(p2p,all_p2p),(copies,all_copies),(cores,all_cores)]:
                target_rows.extend(dict(sample=index,**r) for r in source)
        for key,values in [('work',all_work),('p2p',all_p2p),('projection_copies',all_copies),('cores',all_cores)]:
            save_csv(target/f'{key}.csv',values)
        chosen=sorted(samples,key=lambda s:s['device_ms'])[1]
        result=dict(chosen,samples=samples,package=package,export=load(case/'export.json'))
        dump(target/'summary.json',result)
        cases.append(result)
        print('ANALYZE_OK',name,'device',chosen['device_ms'],'cold_cores',chosen['cold_cores'],
              'cold_DDR_MiB',chosen['cold_projection_ddr_bytes']/2**20,flush=True)
    dump(args.out/f'{args.summary_name}.json',dict(cases=cases))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','compile','timing','timing-summary','profile','analyze'])
    for key in ['cold','source','source-scratch','out','scratch','base-scratch']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--stats-level',type=int,choices=[0,70],default=0)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--timing-prefix',default='timing')
    p.add_argument('--summary-name',default='timing_summary')
    args=p.parse_args()
    if args.rounds<1 or args.iterations<1 or len(set(args.cases))!=len(args.cases):
        p.error('Need positive timing counts and distinct cases')
    if args.action in ['profile','analyze'] and args.stats_level!=70:
        p.error('Profiling requires stats-level 70')
    for key in ['cold','source','source_scratch','out','scratch','base_scratch']:
        setattr(args,key,getattr(args,key).resolve())
    args.out.mkdir(parents=True,exist_ok=True)
    args.scratch.mkdir(parents=True,exist_ok=True)
    for name in args.cases:
        input_guard(args,name)
    if args.action=='timing':
        overhead.timing(args,qpc_fn=qpc)
    elif args.action=='timing-summary':
        overhead.timing_summary(args)
    elif args.action=='analyze':
        analyze(args)
    else:
        for name in args.cases:
            if args.action=='profile':
                overhead.profile(args,name,qpc_fn=qpc)
            else:
                {'export':export,'compile':compile_case}[args.action](args,name)


if __name__=='__main__':
    main()
