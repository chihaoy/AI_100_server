#!/usr/bin/env python3
"""E8 summary: MXFP6 vs FP16 (same session) per block; exactness among MXFP6 variants; all-active policy means."""
import json, os, statistics as st, numpy as np, sys
S='/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad'; E=f'{S}/e8'
out=[]
def P(s=''): out.append(s)
def load(blk):
    p=f'{E}/timing_{blk}.json'
    if not os.path.isfile(p): return None
    rows=json.load(open(p)); labels=[]
    for r in rows:
        if r['label'] not in labels: labels.append(r['label'])
    pooled={l:st.median([s for r in rows if r['label']==l for s in r['samples']]) for l in labels}
    rmed={l:[r['median'] for r in rows if r['label']==l] for l in labels}
    rel={l:max(r['rel_l2_vs_ref'] for r in rows if r['label']==l) for l in labels}
    cok={l:all(r['counts_ok'] for r in rows if r['label']==l) for l in labels}
    return labels,pooled,rmed,rel,cok
def y(l,r=0):
    p=f'{E}/timing_{{}}_runs/{l}_r{r}/y.bin'
    for blk in 'ABD':
        q=p.format(blk)
        if os.path.isfile(q): return np.fromfile(q,np.float16)
    return None
def same(a,b):
    if a is None or b is None or a.size!=b.size: return 'n/a'
    if np.array_equal(a,b): return 'bit-exact'
    return f'rel {np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/np.linalg.norm(b.astype(np.float64)):.1e}'
def table(blk, title, anchor_mx, T_of=lambda l:128, fp16_of=None):
    d=load(blk)
    if d is None: P(f'\n### {title}: no timing yet'); return
    labels,pooled,rmed,rel,cok=d
    anchors={T_of(l.split('__')[0]): y(anchor_mx.format(T=T_of(l.split('__')[0]))) for l in labels} if '{T}' in anchor_mx else None
    ya=y(anchor_mx) if anchor_mx in pooled else None
    P(f'\n### {title}\n'); P('| Case | Precision | Chunk, ms (pooled) | Rounds | µs per token | MXFP6 / FP16 (same graph) | Output vs MXFP6 anchor | rel L2 vs FP16 ref | Counts exact |'); P('|---|---|---:|---|---:|---:|---|---:|---|')
    for l in labels:
        graph,prec=l.split('__'); T=T_of(graph); fp=f'{graph}__fp16'
        ratio=f'{pooled[l]/pooled[fp]:.3f}' if (prec!='fp16' and fp in pooled) else ('1.000' if prec=='fp16' else '–')
        ref_anchor = anchors[T] if anchors is not None else ya
        ex=same(y(l),ref_anchor) if (prec.startswith('mx') and ref_anchor is not None) else '–'
        P(f"| {graph} | {prec.replace('mx','MXFP6').replace('fp16','FP16').replace('_noflag',' no flag')} | {pooled[l]:.3f} | {' / '.join(f'{v:.3f}' for v in rmed[l])} | {pooled[l]/T*1000:.1f} | {ratio} | {ex} | {rel[l]:.2e} | {cok[l]} |")
table('A','Block A: T=128 sorted layout, capacity sweep (E1/E2 mirror)','cc_c128__mx')
table('B','Block B: T=128 placement (E4/E5 mirror)','c2tree_sorted__mx')
table('D','Block D: chunk size × capacity (E6/E7 mirror)','T{T}_hc_{T}_{T}__mx', T_of=lambda g:int(g.split('_')[0][1:]))
# all-active
p=f'{E}/allactive/results.json'
if os.path.isfile(p):
    res=json.load(open(p)); W=['uniform','mild_a','skew_a','skew_b','skew_c']
    P('\n### Block C: all-active plans (E3 mirror), FP16 flag vs MXFP6 flag, medians ms\n'); P('| Plan | '+' | '.join(W)+' |'); P('|---|'+'---:|'*len(W))
    for plan,e in res.items():
        cells=[]
        for w in W:
            cells.append(f"{st.median(e['fp16'][w]):.3f} → {st.median(e['mx'][w]):.3f} ({st.median(e['mx'][w])/st.median(e['fp16'][w]):.2f}, L2 {e['rel_l2'][w]:.1e})" if w in e['fp16'] else '')
        P(f'| {plan} | '+' | '.join(cells)+' |')
    sel={'one static plan':{w:'aggregate_c32_32' for w in W},
         'fixed layout + capacity oracle':{'uniform':'aggregate_c8_8','mild_a':'aggregate_c32_4','skew_a':'aggregate_c32_4','skew_b':'aggregate_c32_32','skew_c':'aggregate_c32_32'},
         'fixed shape + placement oracle':{'uniform':'sorted_c_c32_32','mild_a':'sorted_c_c32_32','skew_a':'sorted_a_c32_32','skew_b':'sorted_b_c32_32','skew_c':'aggregate_c32_32'},
         'joint oracle (frozen)':{'uniform':'stripe_c8_8','mild_a':'sorted_a_c32_4','skew_a':'sorted_a_c32_2','skew_b':'sorted_b_c32_32','skew_c':'aggregate_c32_32'},
         'no regrouping, 32/32':{w:'identity_c32_32' for w in W}}
    P('\n**Policy means over the five workloads (frozen selections from all_active/RESULTS.md):**\n'); P('| Policy | FP16, ms | MXFP6, ms | MXFP6 / FP16 |'); P('|---|---:|---:|---:|')
    for name,m in sel.items():
        try:
            f=st.mean(st.median(res[m[w]]['fp16'][w]) for w in W); x=st.mean(st.median(res[m[w]]['mx'][w]) for w in W); P(f'| {name} | {f:.3f} | {x:.3f} | {x/f:.3f} |')
        except KeyError as k: P(f'| {name} | missing plan {k} | | |')
    for prec in ('fp16','mx'):
        best=[min((st.median(e[prec][w]),pl) for pl,e in res.items() if w in e[prec]) for w in W]
        P(f"| best tested plan per workload ({prec}) | {st.mean(v for v,_ in best):.3f} | | [{' '.join(f'{v:.3f}({pl})' for v,pl in best)}] |")
    spread=[(pl,w,st.median(e['mx'][w])) for pl,e in res.items() for w in W if w in e['mx']]
    P(f"\nMXFP6 plan spread per workload: "+'; '.join(f"{w}: {min(v for pl,ww,v in spread if ww==w):.3f}–{max(v for pl,ww,v in spread if ww==w):.3f} ms" for w in W))
txt='\n'.join(out); print(txt); open(f'{E}/summary.md','w').write(txt+'\n')
