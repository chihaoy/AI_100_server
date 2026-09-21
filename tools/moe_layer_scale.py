#!/usr/bin/env python3
"""Dependent synthetic MoE layer chains with protected routing features."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh,TensorProto as TP
import torch
from moe_joint_scale import H,I,Router,make_weights,reference,tensor_info
from moe_resident_selection import dump


def rename_graph(graph,prefix,replacements):
    def name(n):
        return replacements.get(n,prefix+n) if n else n
    graph=copy.deepcopy(graph)
    def visit(g):
        for n in g.node:
            n.name=prefix+n.name
            n.input[:]=[name(v) for v in n.input]
            n.output[:]=[name(v) for v in n.output]
            for a in n.attribute:
                if a.type==onnx.AttributeProto.GRAPH:
                    visit(a.g)
        for values in [g.input,g.output,g.value_info,g.initializer]:
            for v in values:
                v.name=name(v.name)
    visit(graph)
    return graph


def make_l1(source,root):
    """Matched one-layer/two-profile control for the layer-scaling comparison."""
    source=source.resolve();dest=root.resolve()/'l1';dest.mkdir()
    info=json.loads((source/'info.json').read_text())
    ps=[p for p in info['profiles'] if p['tokens']==128 and p['name'] in ['full4','medium4']]
    for name in ['fused.onnx','weights_l0.bin','inputs','reference']:
        (dest/name).symlink_to(source/name,target_is_directory=name in ['inputs','reference'])
    selected=[w for w in info['workloads'] if w['tokens']==128]
    (dest/'workloads.txt').write_text(''.join(f"{w['name']} 128\n" for w in selected))
    (dest/'profiles.txt').write_text(''.join(f"{p['name']} 128 {p['tag']} {len(p['widths'])} "+' '.join(f'{w} {c}' for w,c in zip(p['widths'],p['capacities']))+'\n' for p in ps))
    dump(dest/'specializations.json',{'specializations':[{'T':'128','P':str(p['tag']),'W':'1' if p['name']=='full4' else '3'} for p in ps]})
    dump(dest/'info.json',{**info,'layers':1,'tokens':[128],'profiles':ps,'workloads':selected})


def export(a):
    torch.set_num_threads(8)
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=a.source.resolve()
    base=onnx.load(source/'fused.onnx',load_external_data=False)
    info=json.loads((source/'info.json').read_text())
    assert info['experts']==128 and info['offset']==0
    ps=[p for p in info['profiles'] if p['tokens']==128 and p['name'] in ['full4','medium4']]
    banks=[];all_inits=[];hashes=[]
    for l in range(4):
        if l==0:
            (root/'weights_l0.bin').symlink_to(source/'weights_l0.bin')
            values={}
            position=0
            for key,shape in [('Wg',(128,H,I)),('Wu',(128,H,I)),('Wd',(128,I,H)),('router',(128,H))]:
                v=np.memmap(root/'weights_l0.bin',dtype=np.float16,mode='c',offset=position,shape=shape)
                values[key]=torch.from_numpy(v);position+=v.nbytes
            inits=[copy.deepcopy(v) for v in base.graph.initializer if v.name in values]
            digest=info['bank_sha256']
        else:
            values,inits,digest=make_weights(root,128,layer=l)
        banks.append(values);all_inits.append(inits);hashes.append(digest)
        print('BANK_READY',l,flush=True)
    selected_ws=[w for w in info['workloads'] if w['tokens']==128]
    refs={w['name']:dict(x=torch.from_numpy(np.fromfile(source/'inputs'/f"{w['name']}_x.bin",np.float16).reshape(128,H)),counts=[],ys=[]) for w in selected_ws}
    for l in range(4):
        for w in selected_ws:
            r=refs[w['name']]
            y,counts,_=reference(r['x'],banks[l],Router(banks[l],128,0))
            r['counts'].append(counts);r['ys'].append(y)
            next_x=(r['x'].float()+torch.from_numpy(y)*.1).half()
            next_x[:,:128]=r['x'][:,:128]
            r['x']=next_x
        print('REFERENCE_LAYER',l,flush=True)
    for layers in [2,4]:
        dest=root/f'l{layers}';dest.mkdir();(dest/'inputs').mkdir();(dest/'reference').mkdir()
        for l in range(layers):
            (dest/f'weights_l{l}.bin').symlink_to(root/f'weights_l{l}.bin')
        nodes=[];inits=[];count_names=[]
        for l in range(layers):
            prefix=f'layer{l}/'
            g=rename_graph(base.graph,prefix,{'x':'x' if l==0 else f'x{l}',
                'expert_ids':f'expert_ids{l}','profile_tag':'profile_tag','layout_tag':'layout_tag',
                'y':'y' if l==layers-1 else f'y{l}'})
            bank_map={v.name:v for v in all_inits[l]}
            for v in g.initializer:
                key=v.name.removeprefix(prefix)
                if key in bank_map:
                    new=copy.deepcopy(bank_map[key]);new.name=v.name;inits.append(new)
                else:
                    inits.append(v)
            nodes.extend(g.node)
            count_names.append(f'counts{l}')
            nodes.append(h.make_node('Unsqueeze',[prefix+'counts','stack_axis'],[count_names[-1]]))
            if l<layers-1:
                xname='x' if l==0 else f'x{l}'
                nodes += [h.make_node('Mul',[f'y{l}','residual_scale'],[f'scaled{l}']),
                    h.make_node('Add',[xname,f'scaled{l}'],[f'residual{l}']),
                    h.make_node('Slice',[xname,'head_start','tail_start','feature_axis'],[f'head{l}']),
                    h.make_node('Slice',[f'residual{l}','tail_start','tail_end','feature_axis'],[f'tail{l}']),
                    h.make_node('Concat',[f'head{l}',f'tail{l}'],[f'x{l+1}'],axis=1)]
        nodes.append(h.make_node('Concat',count_names,['counts'],axis=0))
        for key,value in [('stack_axis',np.array([0],np.int64)),('feature_axis',np.array([1],np.int64)),
            ('head_start',np.array([0],np.int64)),('tail_start',np.array([128],np.int64)),
            ('tail_end',np.array([H],np.int64)),('residual_scale',np.array(.1,np.float16))]:
            inits.append(nh.from_array(value,key))
        ins=[tensor_info('x',TP.FLOAT16,['T',H]),tensor_info('profile_tag',TP.INT32,['P']),tensor_info('layout_tag',TP.INT32,['W'])]
        ins += [tensor_info(f'expert_ids{l}',TP.INT32,[128]) for l in range(layers)]
        outs=[tensor_info('y',TP.FLOAT16,['T',H]),tensor_info('counts',TP.INT32,[layers,128])]
        m=h.make_model(h.make_graph(nodes,'dependent_moe',ins,outs,inits),opset_imports=list(base.opset_import),functions=list(base.functions),ir_version=9)
        onnx.save(m,dest/'fused.onnx');onnx.checker.check_model(str(dest/'fused.onnx'))
        dump(dest/'specializations.json',{'specializations':[{'T':'128','P':str(p['tag']),'W':'1' if p['name']=='full4' else '3'} for p in ps]})
        (dest/'profiles.txt').write_text(''.join(f"{p['name']} 128 {p['tag']} {len(p['widths'])} "+' '.join(f'{w} {c}' for w,c in zip(p['widths'],p['capacities']))+'\n' for p in ps))
        for w in selected_ws:
            name=w['name'];(dest/'inputs'/f'{name}_x.bin').symlink_to(source/'inputs'/f'{name}_x.bin')
            np.stack(refs[name]['counts'][:layers]).tofile(dest/'reference'/f'{name}_counts.bin')
            np.save(dest/'reference'/f'{name}_y.npy',refs[name]['ys'][layers-1])
        (dest/'workloads.txt').write_text(''.join(f"{w['name']} 128\n" for w in selected_ws))
        dump(dest/'info.json',{**info,'layers':layers,'tokens':[128],'profiles':ps,'workloads':selected_ws,
            'bank_sha256':hashes[:layers],'scope':'Dependent synthetic MoE chain; protected first 128 routing features, residual tail x+0.1*y; last contribution output; no attention/KV; oracle profile controls'})
        print('EXPORT_OK',dest,flush=True)
    make_l1(source,root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    export(p.parse_args())
