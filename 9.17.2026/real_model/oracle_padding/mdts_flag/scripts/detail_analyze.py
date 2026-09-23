#!/usr/bin/env python3
"""Full-flow trace analysis for a flag build: per-core work/waits, weight copies, P2P, dependency producers and the
critical path from the last-finishing operation. Usage: detail_analyze.py <merged full-flow trace> <metadata dir> <label> <out dir>"""
import sys, re, json, csv, statistics as st
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, '/home/wentao/workspace/AI_100_server/tools')
from moe_qwen3_layer_detail import schema_from_sdk
from moe_qwen3_profile import canon
trace, meta_dir, label, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
STEM = '/model/layers.2/mlp/'; CORE = re.compile(r'_slice(\d+)_Core_(\d+)(?:_(.*))?$')
HOT = {'MatMul', 'MatMul_1', 'MatMul_2'}; COLD = {'MatMul_3', 'MatMul_4', 'MatMul_5'}
def short(n): return canon(n).replace(STEM, '')
# compiler metadata
_, cls = schema_from_sdk(Path('/opt/qti-aic/exec/qaic-opstats')); lookup = {}
for path in sorted(meta_dir.glob('QAicGraph_slice*_dir/opstatsdesc.bin')):
    card = int(re.search(r'slice(\d+)', str(path))[1]); obj = cls(); obj.ParseFromString(path.read_bytes())
    common = obj.opstats_metadata.common_metadata.op_details
    kinds = {p.id: p.str.strip() for p in common.op_kind_id_str_pairs}; mems = {p.id: p.str for p in common.op_memory_id_str_pairs}
    for spec in obj.opstats_metadata.specialization_metadata:
        for core, groups in enumerate(spec.core_op_name_kinds):
            for group in groups.tg_op_name_kinds:
                for op in group.op_name_kinds:
                    lookup[(card, core, int(op.oc))] = dict(name=op.name, bytes=int(op.output_size), kind=kinds.get(op.kind_id, ''), memory=mems.get(op.memory_id, ''))
raw = json.load(open(trace))['traceEvents']
threads = {(e['pid'], e['tid']): e['args']['name'] for e in raw if e['ph'] == 'M' and e['name'] == 'thread_name'}
work, waits, execution = {}, [], defaultdict(list)
for e in raw:
    if e['ph'] != 'X': continue
    a = e.get('args', {}); th = threads.get((e['pid'], e['tid']), ''); m = CORE.search(th)
    card, core, eng = (int(m[1]), int(m[2]), m[3] or 'execution') if m else (-1, -1, 'P2P')
    s, d = float(e['ts']), float(e.get('dur', 0))
    if eng == 'execution': execution[card].append((s, s + d)); continue
    if 'oc' not in a or e['name'].startswith('barrier '): continue
    is_wait = e['name'].startswith('sync ')
    if is_wait: d = float(a.get('opSyncDurUs', d))
    meta = lookup.get((card, core, int(a['oc'])), {})
    row = dict(card=card, core=core, engine=eng, kind=a.get('opKind', '').strip(), memory=a.get('opMemory', ''), node=canon(a.get('opName', '')),
               lowered=a.get('opName', ''), oc=int(a['oc']), start=s, end=s + d, dur=d, bytes=int(a.get('opOutputSize', meta.get('bytes', 0))),
               port=a.get('PortDescription', ''), transaction=a.get('TransactionId', ''), pid=e['pid'], tid=e['tid'], name=e['name'])
    if is_wait: waits.append(row)
    else: work.setdefault((e['pid'], e['tid'], row['oc']), row)
origin = min(a for v in execution.values() for a, _ in v); finish = max(b for v in execution.values() for _, b in v)
card_end = {c: (max(b for _, b in v) - origin) / 1e3 for c, v in execution.items() if c >= 0}
ms = lambda t: (t - origin) / 1e3
# dependency flows
pending, incoming, outgoing = {}, defaultdict(list), defaultdict(list)
for e in raw:
    if e['ph'] == 's': pending[e['id']] = e
    elif e['ph'] == 'f' and e['id'] in pending:
        src = pending.pop(e['id']); a = (src['pid'], src['tid'], int(src['args']['oc'])); b = (e['pid'], e['tid'], int(e['args']['oc']))
        if a in work and b in work: incoming[b].append((a, src.get('cat') == 'true')); outgoing[a].append((b, src.get('cat') == 'true'))
W = list(work.values()); hmx = [r for r in W if r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32']
stage_of = lambda r: 'hot' if short(r['node']) in HOT else ('cold' if short(r['node']) in COLD else None)
# ---- stage spans per card and per core ----
stages = {}
for stg, names in (('hot', HOT), ('cold', COLD)):
    sel = [r for r in hmx if short(r['node']) in names]
    if not sel: continue
    percard = {c: (ms(min(r['start'] for r in sel if r['card'] == c)), ms(max(r['end'] for r in sel if r['card'] == c))) for c in range(4) if any(r['card'] == c for r in sel)}
    stages[stg] = dict(span=(ms(min(r['start'] for r in sel)), ms(max(r['end'] for r in sel))), percard=percard,
                       cores_per_card={c: len({r['core'] for r in sel if r['card'] == c}) for c in range(4)})
# ---- per core accounting ----
cores = []
for card in range(4):
    for core in range(16):
        rows = [r for r in W if (r['card'], r['core']) == (card, core)]; wrows = [r for r in waits if (r['card'], r['core']) == (card, core)]
        rec = dict(card=card, core=core, last_end_ms=ms(max(r['end'] for r in rows)) if rows else None, idle_tail_ms=(finish - max(r['end'] for r in rows)) / 1e3 if rows else None)
        for stg, names in (('hot', HOT), ('cold', COLD)):
            g = [r for r in rows if r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32' and short(r['node']) in names]
            rec[f'{stg}_gemm_events'] = len(g); rec[f'{stg}_gemm_us'] = sum(r['dur'] for r in g)
            rec[f'{stg}_gemm_start_ms'] = ms(min(r['start'] for r in g)) if g else None; rec[f'{stg}_gemm_end_ms'] = ms(max(r['end'] for r in g)) if g else None
            rec[f'{stg}_weight_MiB'] = sum(r['bytes'] for r in rows if r['kind'] == 'aiccopytovtcm' and r['memory'] == 'DDR' and short(r['node']) in names) / 2**20
            sw = [w for w in wrows if w['engine'] == 'HMX' and short(w['node']) in names]
            rec[f'{stg}_hmx_wait_us'] = sum(w['dur'] for w in sw)
        rec['hmx_wait_total_us'] = sum(w['dur'] for w in wrows if w['engine'] == 'HMX'); rec['hvx_wait_total_us'] = sum(w['dur'] for w in wrows if w['engine'] == 'HVX')
        hv = [r for r in rows if r['engine'] == 'HVX']
        rec['hvx_us_scan'] = sum(r['dur'] for r in hv if r['kind'] == 'cumsum' or 'retune' in r['node'] or short(r['node']).startswith(('Sub', 'Add_1', 'Concat', 'Slice')))
        rec['hvx_us_reduce'] = sum(r['dur'] for r in hv if r['kind'] == 'aicbatchedreduceadd')
        rec['hvx_us_other'] = sum(r['dur'] for r in hv) - rec['hvx_us_scan'] - rec['hvx_us_reduce']
        rec['ddr_copy_MiB_total'] = sum(r['bytes'] for r in rows if r['kind'] == 'aiccopytovtcm' and r['memory'] == 'DDR') / 2**20
        rec['multicast_ops'] = sum(1 for r in rows if 'multicast' in r['kind'])
        cores.append(rec)
# ---- node summary ----
nodes = defaultdict(list)
for r in W:
    if r['card'] >= 0: nodes[(short(r['node']), r['engine'], r['kind'], r['memory'])].append(r)
node_rows = []
for (n, eng, kind, mem), rows in sorted(nodes.items(), key=lambda kv: -sum(r['dur'] for r in kv[1])):
    busy = defaultdict(float)
    for r in rows: busy[(r['card'], r['core'])] += r['dur']
    node_rows.append(dict(node=n, engine=eng, kind=kind, memory=mem, events=len(rows), cores=len(busy), max_core_us=max(busy.values()), median_core_us=st.median(busy.values()),
                          start_ms=ms(min(r['start'] for r in rows)), end_ms=ms(max(r['end'] for r in rows)), bytes_MiB=sum(r['bytes'] for r in rows) / 2**20))
# ---- P2P ----
p2p = [r for r in W if r['port'] and '->' in r['port']]; p2p_rows = []
for n in sorted({short(r['node']) for r in p2p}):
    rows = [r for r in p2p if short(r['node']) == n]
    p2p_rows.append(dict(node=n, sends=len(rows), MiB=sum(r['bytes'] for r in rows) / 2**20, start_ms=ms(min(r['start'] for r in rows)), end_ms=ms(max(r['end'] for r in rows)),
                         edges='; '.join(f"{r['port']} {ms(r['start']):.3f}-{ms(r['end']):.3f}" for r in sorted(rows, key=lambda r: r['start']))))
# ---- critical path: walk back from the last-finishing op through the latest-ending true producer ----
def keyof(r): return (r['pid'], r['tid'], r['oc'])
last = max((r for r in W if r['card'] >= 0), key=lambda r: r['end']); chain = []; seen = set(); cur = last
for _ in range(40):
    k = keyof(cur); chain.append(dict(card=cur['card'], core=cur['core'], engine=cur['engine'], node=short(cur['node']), kind=cur['kind'], port=cur['port'], start_ms=ms(cur['start']), end_ms=ms(cur['end']), dur_us=cur['dur']))
    preds = [(work[a], t) for a, t in incoming.get(k, []) if a not in seen]
    if not preds: break
    # prefer true data producers; pick the one ending latest (the binding dependency)
    true = [p for p in preds if p[1]] or preds
    nxt = max(true, key=lambda p: p[0]['end'])[0]; seen.add(k); cur = nxt
    if ms(cur['end']) < 0.05: break
# ---- what do cores without cold GEMMs do during the cold stage? ----
idle_cold = []
if 'cold' in stages:
    lo, hi = stages['cold']['span']
    for card in range(4):
        for core in range(16):
            if any(r['card'] == card and r['core'] == core and r['engine'] == 'HMX' and r['kind'] == 'aicconvolutiond32' and short(r['node']) in COLD for r in W): continue
            rows = [r for r in W if (r['card'], r['core']) == (card, core) and lo * 1e3 + origin <= r['start'] <= hi * 1e3 + origin]
            kinds = Counter(f"{r['engine']}:{r['kind']}" for r in rows); busy = sum(r['dur'] for r in rows)
            idle_cold.append(dict(card=card, core=core, ops_in_cold_window=len(rows), busy_us=busy, top_kinds=dict(kinds.most_common(3))))
def wcsv(name, rows):
    if rows:
        with open(out / f'{name}.csv', 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
wcsv('work', [dict(**{k: v for k, v in r.items() if k not in ('pid', 'tid')}, start_ms=ms(r['start']), end_ms=ms(r['end'])) for r in W])
wcsv('waits', [dict(**{k: v for k, v in r.items() if k not in ('pid', 'tid')}, start_ms=ms(r['start']), end_ms=ms(r['end'])) for r in waits])
wcsv('cores', cores); wcsv('nodes', node_rows); wcsv('p2p_summary', p2p_rows); wcsv('critical_path', chain); wcsv('cores_without_cold_gemm', idle_cold)
summary = dict(label=label, trace=str(trace), device_ms=(finish - origin) / 1e3, card_end_ms=card_end, stages=stages, p2p_total_MiB=sum(r['MiB'] for r in p2p_rows),
               critical_path=chain, flows=sum(len(v) for v in incoming.values()))
json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
# ---- print ----
print(f"== {label}: device {summary['device_ms']:.3f} ms; card execution ends {{{', '.join(f'{c}: {v:.3f}' for c, v in sorted(card_end.items()))}}}; P2P {summary['p2p_total_MiB']:.3f} MiB; flows {summary['flows']}")
for stg, v in stages.items():
    print(f"   {stg}: span {v['span'][0]:.3f}-{v['span'][1]:.3f} ms; per card {{{', '.join(f'{c}: {a:.3f}-{b:.3f}' for c, (a, b) in v['percard'].items())}}}; cores/card {v['cores_per_card']}")
print('   per-card totals (median core / max core):')
for card in range(4):
    cc = [c for c in cores if c['card'] == card]
    f = lambda k: f"{st.median(x[k] for x in cc):.0f}/{max(x[k] for x in cc):.0f}"
    print(f"     card{card}: hot GEMM us {f('hot_gemm_us')} wait {f('hot_hmx_wait_us')} weights MiB {st.median(x['hot_weight_MiB'] for x in cc):.1f}/{max(x['hot_weight_MiB'] for x in cc):.1f} | cold GEMM us {f('cold_gemm_us')} wait {f('cold_hmx_wait_us')} weights MiB {st.median(x['cold_weight_MiB'] for x in cc):.1f}/{max(x['cold_weight_MiB'] for x in cc):.1f} | HVX scan/reduce/other us {f('hvx_us_scan')} {f('hvx_us_reduce')} {f('hvx_us_other')} | last op end ms {st.median(x['last_end_ms'] for x in cc):.3f}/{max(x['last_end_ms'] for x in cc):.3f}")
print('   top nodes by max-core busy:')
for n in node_rows[:14]: print(f"     {n['node']:22s} {n['engine']:12s} {n['kind']:24s} {n['memory']:5s} ev {n['events']:5d} cores {n['cores']:2d} max-core {n['max_core_us']:8.1f} us med {n['median_core_us']:8.1f} us window {n['start_ms']:.3f}-{n['end_ms']:.3f} bytes {n['bytes_MiB']:.1f} MiB")
print('   P2P:'); [print(f"     {p['node']:22s} {p['sends']:2d} sends {p['MiB']:8.3f} MiB window {p['start_ms']:.3f}-{p['end_ms']:.3f}") for p in p2p_rows]
print('   critical path (last op backwards through binding producers):')
for c in chain[:25]: print(f"     card{c['card']} core{c['core']:2d} {c['engine']:12s} {c['node']:24s} {c['kind']:26s} {c['port']:6s} {c['start_ms']:.3f}-{c['end_ms']:.3f} ms ({c['dur_us']:.0f} us)")
if idle_cold: print(f"   cores without cold GEMMs: {len(idle_cold)}; busy in cold window median {st.median(x['busy_us'] for x in idle_cold):.0f} us; e.g. {idle_cold[0]}")
