#!/usr/bin/env python3
"""After make_tokencombine.py tokenowned on a wide-cold-stage graph: stage index by position >= 64, stage-1 card/local with
L1/4 lanes per card, stage-1 data reshape [4, (L1/4)*C1, 2048]. Usage: patch_wide_to.py <dir> <L1>"""
import sys, os, onnx, numpy as np
from onnx import helper as h, numpy_helper as nh
d, L1 = sys.argv[1], int(sys.argv[2]); PER = L1 // 4
m = onnx.load(f'{d}/model.onnx', load_external_data=False); g = m.graph; STEM = '/model/layers.2/mlp/'
bn = {n.name: n for n in g.node}; init = {t.name: t for t in g.initializer}
C1 = int(nh.to_array(init['probe_stage1_stop']))
# stage = pos >= 64 (positions 64..64+L1-1 are stage 1)
st = bn[STEM + 'to_stage']; st.op_type = 'Greater'; st.input[1] = 'to_c63'; g.initializer.append(nh.from_array(np.array(63, np.int32), 'to_c63'))
st.output[0] = 'to_stage_bool'
new = [h.make_node('Cast', ['to_stage_bool'], ['to_stage'], name=STEM + 'to_stage_cast', to=onnx.TensorProto.INT32)]
nodes = list(g.node); pos = next(i for i, n in enumerate(nodes) if n.name == STEM + 'to_stage'); nodes.insert(pos + 1, new[0])
# stage-1 card/local
g.initializer.append(nh.from_array(np.array(PER, np.int32), 'to_cPER'))
extra = [h.make_node('Div', ['to_lane', 'to_cPER'], ['to_card_s1'], name=STEM + 'to_card_s1'),
         h.make_node('Mul', ['to_card_s1', 'to_cPER'], ['to_cardPER_s1'], name=STEM + 'to_cardPER_s1'),
         h.make_node('Sub', ['to_lane', 'to_cardPER_s1'], ['to_local_s1'], name=STEM + 'to_local_s1')]
pos = next(i for i, n in enumerate(nodes) if n.name == STEM + 'to_local'); nodes[pos + 1:pos + 1] = extra; del g.node[:]; g.node.extend(nodes)
for n in g.node:
    if n.name.endswith('_s1') and n.name.startswith(STEM + 'to_oncard'):
        n.input[0] = 'to_card_s1'
    if n.name == STEM + 'to_localrow_s1': n.input[0] = 'to_local_s1'
sh = init['to_shape_d1']; sh.CopyFrom(nh.from_array(np.array([4, PER * C1, 2048], np.int64), 'to_shape_d1'))
_c = os.getcwd(); os.chdir(d); onnx.checker.check_model(m); os.chdir(_c); onnx.save(m, f'{d}/model.onnx'); print(f'{d}: stage by position>=64, stage-1 {PER} lanes per card, data [4, {PER * C1}, 2048]')
