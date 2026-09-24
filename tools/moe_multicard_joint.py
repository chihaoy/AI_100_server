#!/usr/bin/env python3
"""Four independent resident expert banks with current-count host dispatch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from moe_joint_scale import H,I,SDK,Router,export as export_bank,reference
from moe_resident_selection import command,compare,dump


def export(a):
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=a.source.resolve();shard0=a.shard0.resolve()
    (root/'card0').symlink_to(shard0,target_is_directory=True)
    for card in [1,2,3]:
        export_bank(argparse.Namespace(out=root/f'card{card}',experts=32,offset=32*card,resume=False))
    weights={};position=0
    for key,shape in [('Wg',(128,H,I)),('Wu',(128,H,I)),('Wd',(128,I,H)),('router',(128,H))]:
        values=np.memmap(source/'weights_l0.bin',dtype=np.float16,mode='c',offset=position,shape=shape)
        weights[key]=torch.from_numpy(values);position+=values.nbytes
    hashes={}
    for card in range(4):
        info=json.loads((root/f'card{card}/info.json').read_text())
        hashes[card]={key:hashlib.sha256(weights[key][card*32:(card+1)*32].numpy().tobytes()).hexdigest() for key in ['Wg','Wu','Wd']}
        assert all(value==info['bank_sha256'][key] for key,value in hashes[card].items())
    dest=root/'global';dest.mkdir();(dest/'reference').mkdir()
    for name in ['profiles.txt','specializations.json','router_specializations.json','fused.onnx','router.onnx','expert.onnx','weights_l0.bin']:
        (dest/name).symlink_to(source/name)
    (dest/'inputs').symlink_to(shard0/'inputs',target_is_directory=True)
    for precision in ['fp16','mxfp6']:
        (dest/precision).mkdir()
        for variant in ['router1','fused15','expert15','fixed','conservative']:
            (dest/precision/variant).symlink_to(source/precision/variant,target_is_directory=True)
    info=json.loads((source/'info.json').read_text());workloads=[]
    router=Router(weights,128,0)
    for original in info['workloads']:
        name=original['name'];tokens=original['tokens']
        x=torch.from_numpy(np.fromfile(dest/'inputs'/f'{name}_x.bin',np.float16).reshape(tokens,H))
        y,counts,rw=reference(x,weights,router)
        counts.tofile(dest/'reference'/f'{name}_counts.bin');rw.tofile(dest/'reference'/f'{name}_route_weights.bin')
        np.save(dest/'reference'/f'{name}_y.npy',y)
        ids=np.lexsort((np.arange(128),-counts)).astype(np.int32)
        selected=min([p for p in info['profiles'] if p['tokens']==tokens and np.all(counts[ids]<=np.repeat(p['capacities'],p['widths']))],key=lambda p:(p['padded_rows'],p['tag']))
        workloads.append({**original,'peak':int(counts.max()),'assignments':int(counts.sum()),'selected_tag':selected['tag'],
            'selected_name':selected['name'],'owner_assignments':counts.reshape(4,32).sum(1).tolist(),
            'owner_peaks':counts.reshape(4,32).max(1).tolist()})
    (dest/'workloads.txt').write_text(''.join(f"{w['name']} {w['tokens']}\n" for w in workloads))
    dump(dest/'info.json',{**info,'workloads':workloads,'scope':'Full synthetic 128-expert reference for four stationary 32-expert banks; inputs emphasize card-0 routing skew; no KV'})
    dump(root/'info.json',dict(source=str(source),shard0=str(shard0),bank_slice_sha256=hashes,
        global_reference=str(dest),ownership={str(c):list(range(c*32,(c+1)*32)) for c in range(4)},
        scope='One current router on card0/core1; four independent expert programs on 15 cores each; host routing handoff and FP32 reduction; no migration/replication'))
    print('MULTICARD_EXPORT_OK',root,flush=True)


def run(a):
    base=a.out/a.precision/a.run_name;base.mkdir(parents=True,exist_ok=False)
    host=base/'host'
    if command(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-mavx','-mf16c',f'-I{SDK}/dev/inc',
        Path(__file__).with_name('moe_multicard_joint_host.cpp'),'-o',host,f'-L{SDK}/dev/lib/x86_64','-lQAic',
        f'-Wl,-rpath,{SDK}/dev/lib/x86_64'],base/'build.log'):
        raise RuntimeError('Host build failed')
    for mode in a.modes:
        if command([host,a.out/'global',a.out,a.precision,base/mode,a.iterations,int(mode=='serial')],base/f'{mode}.log'):
            raise RuntimeError(f'Host failed: {mode}')


def analyze(a):
    info=json.loads((a.out/'global/info.json').read_text());base=a.out/a.precision/a.run_name
    result={'precision':a.precision,'scope':json.loads((a.out/'info.json').read_text())['scope'],'modes':{}}
    for mode in a.modes:
        dest=base/mode
        rows=list(csv.DictReader((dest/'samples.csv').open()))
        resources=list(csv.DictReader((dest/'resources.csv').open()))
        mr={'samples':len(rows),'cases':{},'latency':[],'resources':resources,'device_memory':{}}
        for card in range(4):
            rs=[r for r in resources if int(r['device'])==card];ready=next(i for i,r in enumerate(rs) if r['stage']=='ready')
            mr['device_memory'][str(card)]=dict(resident_mib=(int(rs[0]['dram_free_kib'])-int(rs[ready]['dram_free_kib']))/1024,
                stable=len({(r['dram_free_kib'],r['nsp_free']) for r in rs[ready:]})==1)
        for w in info['workloads']:
            name=w['name'];ref=np.load(a.out/'global/reference'/f'{name}_y.npy').reshape(-1)
            fulltag=next(p['tag'] for p in info['profiles'] if p['tokens']==w['tokens'] and p['name']=='full4')
            control=a.out/'global'/a.precision/'control/fused15'/f'{name}_sorted_p{fulltag}_y.bin'
            if not control.exists():
                raise RuntimeError(f'Missing one-card control {control}')
            for policy in ['independent','common','conservative']:
                key=name+'_'+policy;y=np.fromfile(dest/f'{key}_y.bin',np.float16)
                partials=[np.fromfile(dest/f'{key}_card{card}_y.bin',np.float16).astype(np.float32) for card in range(4)]
                merged=np.zeros_like(partials[0])
                for partial in partials:
                    merged+=partial
                counts=np.fromfile(dest/f'{key}_counts.bin',np.int32)
                mr['cases'][key]=dict(vs_cpu=compare(y,ref),vs_one_card=compare(y,np.fromfile(control,np.float16)),
                    vs_conservative=compare(y,np.fromfile(dest/f'{name}_conservative_y.bin',np.float16)),
                    merge_exact=bool(np.array_equal(y,merged.astype(np.float16))),
                    counts_exact=bool(np.array_equal(counts,np.fromfile(a.out/'global/reference'/f'{name}_counts.bin',np.int32))))
                for schedule in ['held','cycle']:
                    selected=[r for r in rows if r['workload']==name and r['policy']==policy and r['schedule']==schedule]
                    val=dict(workload=name,policy=policy,schedule=schedule,samples=len(selected),
                        tag_tuples=sorted(set(tuple(int(r[f'tag{c}']) for c in range(4)) for r in selected)))
                    for k in ['total_ms','router_ms','decision_ms','handoff_ms','dispatch_wait_ms','merge_ms']:
                        ts=[float(r[k]) for r in selected];val[k]=dict(median=float(np.median(ts)),p10=float(np.percentile(ts,10)),p90=float(np.percentile(ts,90)))
                    mr['latency'].append(val)
        result['modes'][mode]=mr
    values=[v for m in result['modes'].values() for v in m['cases'].values()]
    result['mechanism_pass']=all(v['counts_exact'] and v['merge_exact'] and v['vs_one_card']['relative_l2']<.005 and v['vs_conservative']['relative_l2']<.005 for v in values) and all(d['stable'] for m in result['modes'].values() for d in m['device_memory'].values())
    result['reference_accuracy_pass']=all(v['vs_cpu']['relative_l2']<.05 for v in values)
    dump(base/'result.json',result)
    print('MULTICARD_RESULT',a.precision,result['mechanism_pass'],result['reference_accuracy_pass'],flush=True)
    if not result['mechanism_pass'] or (a.precision=='fp16' and not result['reference_accuracy_pass']):
        raise RuntimeError('Validation failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['export','run','analyze'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--shard0',type=Path)
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16');p.add_argument('--run-name',default='run')
    p.add_argument('--modes',nargs='+',default=['parallel','serial']);p.add_argument('--iterations',type=int,default=10)
    a=p.parse_args();a.out=a.out.resolve();{'export':export,'run':run,'analyze':analyze}[a.action](a)
