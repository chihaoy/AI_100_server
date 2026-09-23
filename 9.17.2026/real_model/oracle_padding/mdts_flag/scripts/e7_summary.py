#!/usr/bin/env python3
"""Summarize E7 timing: pooled medians per variant, µs/token, ratios vs the same-T baselines, output exactness."""
import json, glob, os, statistics as st, numpy as np
E='/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad/e7'
rows=json.load(open(f'{E}/timing.json')); labels=[]
for r in rows:
    if r['label'] not in labels: labels.append(r['label'])
info={l:json.load(open(f'{E}/{l}/info.json')) for l in labels}
pooled={l:st.median([s for r in rows if r['label']==l for s in r['samples']]) for l in labels}
rmed={l:[r['median'] for r in rows if r['label']==l] for l in labels}
rel={l:[r['rel_l2_vs_ref'] for r in rows if r['label']==l] for l in labels}
cok={l:all(r['counts_ok'] for r in rows if r['label']==l) for l in labels}
def y(l,r=0): return np.fromfile(f'{E}/timing_runs/{l}_r{r}/y.bin', np.float16)
out=[]
for T in (256,512):
    base=f'T{T}_hc_{T}_{T}'; lpt=f'T{T}_lpt_{T}_{T}'
    if base not in pooled: continue
    yb=y(base); out.append(f'\n### T={T} (grown counts, {info[base]["active_experts"]} active experts, {info[base]["assignments"]} assignments)\n')
    out.append('| Variant | Placement | Capacity hot/cold | Padded rows / real | Chunk, ms (pooled) | Rounds | µs per token | vs hc T/T | vs LPT T/T | Output vs hc T/T | rel L2 vs E6 LPT ref | Counts exact |')
    out.append('|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|')
    for l in [x for x in labels if x.startswith(f'T{T}_')]:
        i=info[l]; yy=y(l); exact='bit-exact' if np.array_equal(yy,yb) else f'max abs {np.abs(yy.astype(np.float32)-yb.astype(np.float32)).max():.3g}'
        rr=[r for r in rmed[l]]; same=all(np.array_equal(y(l,k),yy) for k in range(1,len(rr)))
        out.append(f"| {l} | {i['mode']} | {i['C_hot']}/{i['C_cold']} | {i['padded_over_real']}× | {pooled[l]:.3f} | {' / '.join(f'{v:.3f}' for v in rr)} | {pooled[l]/T*1000:.1f} | {pooled[l]/pooled[base]:.3f} | {pooled[l]/pooled[lpt]:.3f} | {exact}{'' if same else ' (rounds differ!)'} | {max(rel[l]):.2e} | {cok[l]} |")
print('\n'.join(out)); open(f'{E}/summary.md','w').write('\n'.join(out)+'\n')
json.dump(dict(pooled=pooled, round_medians=rmed, rel_l2=rel, counts_ok=cok), open(f'{E}/summary.json','w'), indent=1)
