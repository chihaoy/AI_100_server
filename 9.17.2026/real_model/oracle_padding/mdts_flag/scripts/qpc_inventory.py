#!/usr/bin/env python3
"""Compile-time op inventory per node from QPC opstatsdesc.bin descriptors; no device needed."""
import sys, re
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools')
from moe_qwen3_layer_detail import schema_from_sdk
PAT = re.compile(sys.argv[2] if len(sys.argv) > 2 else r'^(MatMul(_[1-5])?|Mul_4|Mul_9|Add|Add_2|Einsum_4|CumSum|CumSum_1)$|down|Transpose|concat|split')
def load_rows(meta_dir, cls):
    rows = []
    for path in sorted(Path(meta_dir).glob('QAicGraph_slice*_dir/opstatsdesc.bin')):
        card = int(re.search(r'slice(\d+)', str(path))[1])
        obj = cls(); obj.ParseFromString(path.read_bytes())
        common = obj.opstats_metadata.common_metadata.op_details
        kinds = {p.id: p.str.strip() for p in common.op_kind_id_str_pairs}
        mems = {p.id: p.str for p in common.op_memory_id_str_pairs}
        for spec in obj.opstats_metadata.specialization_metadata:
            for core, groups in enumerate(spec.core_op_name_kinds):
                for group in groups.tg_op_name_kinds:
                    for op in group.op_name_kinds:
                        rows.append(dict(card=card, core=core, name=op.name, bytes=int(op.output_size), kind=kinds.get(op.kind_id, f'kind{op.kind_id}'), memory=mems.get(op.memory_id, f'mem{op.memory_id}')))
    return rows
def short(name):
    m = re.search(r'/mlp/([^/]+)', name)
    return m[1] if m else name
_, cls = schema_from_sdk(Path('/opt/qti-aic/exec/qaic-opstats'))
rows = load_rows(sys.argv[1], cls)
print(f'{sys.argv[1]}: {len(rows)} ops, cards {sorted({r["card"] for r in rows})}')
by_node = defaultdict(list)
for r in rows: by_node[short(r['name'])].append(r)
def summarize(sel):
    by = defaultdict(list)
    for r in sel: by[(r['kind'], r['memory'])].append(r)
    parts = []
    for (kind, mem), rr in sorted(by.items(), key=lambda kv: -sum(x['bytes'] for x in kv[1])):
        cores = len({(r['card'], r['core']) for r in rr}); sizes = Counter(r['bytes'] for r in rr)
        parts.append(f"{kind}/{mem}: {len(rr)} ops/{cores} cores [{', '.join(f'{n}x{b//1024}K' for b, n in sizes.most_common(2))}] {sum(r['bytes'] for r in rr)/2**20:.1f}MiB")
    return ' | '.join(parts)
for node in sorted(by_node, key=lambda n: (not PAT.search(n), n)):
    if PAT.search(node): print(f'  {node:18s}: {summarize(by_node[node])}')
ddr_stage = sum(r['bytes'] for r in rows if r['kind']=='aiccopytovtcm' and r['memory']=='DDR' and short(r['name']) in ('Add','Add_2'))
print(f'  >> re-shard signature: DDR staging copies attributed to Add/Add_2 = {ddr_stage/2**20:.1f} MiB; '
      f'Mul_4/Mul_9 multicast ops = {sum(1 for r in rows if r["kind"]=="aicmulticastvtcm" and short(r["name"]) in ("Mul_4","Mul_9"))}; '
      f'total ops = {len(rows)}')
