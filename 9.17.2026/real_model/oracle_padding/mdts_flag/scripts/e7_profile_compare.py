#!/usr/bin/env python3
"""Compare detail_analyze outputs: usage: e7_profile_compare.py label=analysis_dir [label=analysis_dir ...]"""
import sys, json, csv, statistics as st, collections
def load(d):
    s=json.load(open(f'{d}/summary.json')); cores=list(csv.DictReader(open(f'{d}/cores.csv'))); nodes=list(csv.DictReader(open(f'{d}/nodes.csv'))); return s,cores,nodes
def cls(n):
    if n.startswith('MatMul'): return 'weights'
    if n.startswith('CtxGather'): return 'gathers'
    if n.startswith('CtxScatter'): return 'scatters'
    if n.startswith('ConstantOfShape'): return 'splat'
    if n.startswith('reduce_tile'): return 'reduce tiles'
    if n.startswith('Einsum_4'): return 'final combine'
    return 'other'
for arg in sys.argv[1:]:
    label,d=arg.split('=',1); s,cores,nodes=load(d)
    print(f"\n=== {label}: device {s['device_ms']:.3f} ms; card ends {' / '.join(f'{v:.2f}' for k,v in sorted(s['card_end_ms'].items()))}; P2P {s['p2p_total_MiB']} MiB")
    for stg,v in s['stages'].items():
        parts=[]
        for card,sp in sorted(v['percard'].items()):
            n=v['cores_per_card'][card]; parts.append(f"card{card} {sp[0]:.2f}->{sp[1]:.2f} ({sp[1]-sp[0]:.2f} ms, {n} exp, {(sp[1]-sp[0])/max(n,1)*1e3:.0f} us/exp)")
        print(f"   {stg:4s}: "+'; '.join(parts))
    for card in '0123':
        cc=[c for c in cores if c['card']==card]
        hg=[float(c['hot_gemm_us']) for c in cc if float(c['hot_gemm_events'])>0]; cg=[float(c['cold_gemm_us']) for c in cc if float(c['cold_gemm_events'])>0]
        hw=[float(c['hot_hmx_wait_us']) for c in cc if float(c['hot_gemm_events'])>0]; cw=[float(c['cold_hmx_wait_us']) for c in cc if float(c['cold_gemm_events'])>0]
        he=[float(c['hot_gemm_end_ms']) for c in cc if c['hot_gemm_end_ms']]; ce=[float(c['cold_gemm_end_ms']) for c in cc if c['cold_gemm_end_ms']]
        print(f"   card{card}: hot GEMM us med/max {st.median(hg) if hg else 0:.0f}/{max(hg) if hg else 0:.0f} wait med {st.median(hw) if hw else 0:.0f} end max {max(he) if he else 0:.2f} | cold GEMM us med/max {st.median(cg) if cg else 0:.0f}/{max(cg) if cg else 0:.0f} wait med {st.median(cw) if cw else 0:.0f} end max {max(ce) if ce else 0:.2f} | last op {max(float(c['last_end_ms']) for c in cc):.2f}")
    ddr=collections.defaultdict(float); win={}
    for n in nodes:
        if 'DDR' in n['memory'] and float(n['bytes_MiB'])>=1 and not n['node'].startswith('aicopstats'):
            c=cls(n['node']); ddr[c]+=float(n['bytes_MiB']); a,b=float(n['start_ms']),float(n['end_ms']); w=win.get(c,(a,b)); win[c]=(min(w[0],a),max(w[1],b))
    print('   DDR traffic >=1 MiB by class: '+'; '.join(f"{k} {v:.0f} MiB [{win[k][0]:.2f}-{win[k][1]:.2f}]" for k,v in sorted(ddr.items(), key=lambda kv:-kv[1])))
    e4=[n for n in nodes if n['node'].startswith('Einsum_4')]
    if e4: print('   Einsum_4 ops: '+'; '.join(f"{n['engine']}/{n['kind']}/{n['memory']} max {float(n['max_core_us']):.0f} us [{float(n['start_ms']):.2f}-{float(n['end_ms']):.2f}]" for n in e4))
