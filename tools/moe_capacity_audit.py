#!/usr/bin/env python3
"""Inspect per-card QPC constants and optionally collect capacity-specific traces."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx.utils import Extractor


class CtxScatter3DInt(OpRun):
    op_domain='com.qualcomm.cloud'
    def _run(self,data,position_ids,updates):
        out=data.copy()
        valid=(position_ids>=0)&(position_ids<data.shape[1])
        batch=np.broadcast_to(np.arange(data.shape[0])[:,None],position_ids.shape)
        out[batch[valid],position_ids[valid]]=updates[valid]
        return (out,)


class CtxGather3D(OpRun):
    op_domain='com.qualcomm.cloud'
    def _run(self,data,position_ids):
        out=np.zeros((*position_ids.shape,data.shape[-1]),dtype=data.dtype)
        valid=(position_ids>=0)&(position_ids<data.shape[1])
        batch=np.broadcast_to(np.arange(data.shape[0])[:,None],position_ids.shape)
        out[valid]=data[batch[valid],position_ids[valid]]
        return (out,)


def graph_shapes(root,info):
    """Execute the actual ONNX packing dependencies, without expert GEMMs.

This audits logical GEMM input sizes; device traces are a separate check. It
does not assert a particular physical HMX tile shape or expert-to-core mapping.
"""
    m=onnx.load(root/'model.onnx')
    outputs=[n.input[0] for n in m.graph.node if n.name in ['/MatMul_1','/MatMul_4']]
    if len(outputs)!=2: raise ValueError('Expected two expert stages')
    sub=Extractor(m).extract_model([v.name for v in m.graph.input],outputs)
    evaluator=ReferenceEvaluator(sub,new_ops=[CtxScatter3DInt,CtxGather3D])
    x=np.fromfile(root/'x.bin',np.float16).reshape(128,info['H']); result={}
    for i,(c0,c1) in enumerate(info['profiles']):
        inputs={'x':x}
        if info['selector']=='joint':
            inputs['grid']=np.arange(c0,dtype=np.int32)[:,None]+np.arange(c1,dtype=np.int32)[None,:]
        else:
            inputs.update(rows0=np.arange(c0,dtype=np.int32),rows1=np.arange(c1,dtype=np.int32))
            if info['selector']=='tagged': inputs['tag']=np.zeros(i+1,dtype=np.int32)
        shapes=[list(y.shape) for y in evaluator.run(None,inputs)]
        assert shapes==[[64,c0,info['H']],[64,c1,info['H']]],shapes
        result[f'{c0}_{c1}']={'packed_activation_shapes':shapes,'expert_gemm_rows':[c0]*3+[c1]*3}
    return result


def command(args, log):
    with log.open('w') as f:
        f.write(repr(list(map(str,args)))+'\n'); f.flush()
        subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precision',choices=['fp16','mxfp6'],default='fp16')
    p.add_argument('--profile',action='store_true')
    a=p.parse_args(); root=a.out.resolve(); base=root/a.precision
    info=json.loads((root/'info.json').read_text())
    report={'logical_graph_shapes':graph_shapes(root,info)}
    for name in ['multi','single']:
        dest=base/f'{name}_segments'
        if not dest.exists():
            command(['/opt/qti-aic/tools/qaic-qpc','extract','--qpc',base/name/'programqpc.bin',
                     '--output-dir',dest,'-s','*StaticConstants.constants.bin',
                     '-s','*networkdesc.bin_dir/networkdesc.json'],base/f'{name}_extract.log')
        constants={str(f.relative_to(dest)):f.stat().st_size for f in dest.rglob('StaticConstants.constants.bin')}
        shapes={}
        for f in dest.rglob('networkdesc.json'):
            d=json.loads(f.read_text())
            shapes[d['network_name']]={'input_names':[x['name'] for x in d['inputs']],
                                      'allowed_shapes':d['allowed_shapes']}
        report[name]={'static_constants_bytes':sum(constants.values()),'per_card_constants':constants,
                      'qpc_bytes':(base/name/'programqpc.bin').stat().st_size,'network_shapes':shapes}
    report['multi_single_constants_ratio']=report['multi']['static_constants_bytes']/report['single']['static_constants_bytes']
    report['scope']='Packed constant storage; not a PCIe trace or complete runtime memory accounting'
    (base/'package_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CONSTANTS',report['single']['static_constants_bytes'],'->',report['multi']['static_constants_bytes'],flush=True)
    if not a.profile: return
    dest=base/'profiling'; dest.mkdir(exist_ok=False)
    for index,(c0,c1) in enumerate(info['profiles']):
        sub=dest/f'{c0}_{c1}'; sub.mkdir()
        ios=[{'path':str(root/'x.bin'),'dims':[128,info['H']],'elem-size':2,'io-direction':'in','map-to':'x'}]
        if info['selector']=='tagged':
            values={'rows0':np.arange(c0,dtype=np.int32),'rows1':np.arange(c1,dtype=np.int32),
                    'tag':np.zeros(index+1,dtype=np.int32)}
        elif info['selector']=='joint':
            values={'grid':np.arange(c0,dtype=np.int32)[:,None]+np.arange(c1,dtype=np.int32)[None,:]}
        else: raise ValueError('Profiling requires a supported selector')
        for name,v in values.items():
            file=sub/f'{name}.bin';v.tofile(file)
            ios.append({'path':str(file),'dims':list(v.shape),'elem-size':4,'io-direction':'in','map-to':name})
        # Outputs have fixed dimensions for every profile. Include expected files
        # so runner's -c comparison verifies the exact benchmark outputs too.
        for name,shape,size in [('y',[128,info['H']],2),('counts',[128],4)]:
            ios.append({'path':str(base/'run'/f'{c0}_{c1}_{name}.bin'),'dims':shape,
                        'elem-size':size,'io-direction':'out','map-to':name})
        config=sub/'io.json';config.write_text(json.dumps({'IO-files':[ios]},indent=2)+'\n')
        stats=sub/'stats';stats.mkdir()
        command(['/opt/qti-aic/exec/qaic-runner','-t',base/'multi','-D',':'.join(map(str,range(info['devices']))),
                 '--aic-batch-json-input',config,'-n','3','-S','1','-T','1','-c',
                 '--aic-profiling-type','raw_device_stats','--aic-profiling-num-samples','1',
                 '--aic-profiling-out-dir',stats],sub/'runner.log')
        traces=sub/'trace';traces.mkdir()
        command(['/opt/qti-aic/exec/qaic-opstats','--qpc',base/'multi/programqpc.bin','--input-dir',stats,
                 '--output-dir',traces,'--summary','--trace','--merge-mq-traces','true','--flow-events','none'],
                 sub/'opstats.log')
        print('PROFILE_OK',sub,flush=True)


if __name__=='__main__':main()
