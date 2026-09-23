#!/usr/bin/env python3
"""Per-core Gantt for selected cards from an analysis dir's work.csv/waits.csv. Usage: detail_plot.py <analysis dir> <title> <out.png> <card,card,...>"""
import sys, csv
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.patches import Patch
adir, title, outp, cards = sys.argv[1], sys.argv[2], sys.argv[3], [int(c) for c in sys.argv[4].split(',')]
work = list(csv.DictReader(open(f'{adir}/work.csv'))); waits = list(csv.DictReader(open(f'{adir}/waits.csv')))
HOT = {'MatMul', 'MatMul_1', 'MatMul_2'}; COLD = {'MatMul_3', 'MatMul_4', 'MatMul_5'}
def cat(r):
    n = r['node'].split('/mlp/')[-1]; k = r['kind']; e = r['engine']
    if e == 'HMX' and k == 'aicconvolutiond32': return 'gemm_hot' if n in HOT else ('gemm_cold' if n in COLD else 'gemm_other')
    if k == 'aiccopytovtcm' and r['memory'] == 'DDR' and (n in HOT or n in COLD): return 'weight_dma'
    if k == 'aicbatchedreduceadd': return 'reduce'
    if e.startswith('HVX') and 'multicast' in k: return 'multicast'
    if e.startswith('HVX'): return 'hvx_other'
    if e.startswith('DMAIssue'): return 'dma_other'
    return None
colors = dict(gemm_hot='#1f77b4', gemm_cold='#ff7f0e', gemm_other='#17becf', weight_dma='#9467bd', reduce='#2ca02c', multicast='#e377c2', hvx_other='#bcbd22', dma_other='#c7c7c7', hmx_wait='#d0d0d0')
lanes = dict(hmx_wait=0.0, gemm_hot=0.0, gemm_cold=0.0, gemm_other=0.0, weight_dma=0.3, dma_other=0.3, reduce=0.6, hvx_other=0.6, multicast=0.6)
dev_end = max(float(r['end_ms']) for r in work if int(r['card']) >= 0)
fig, axes = plt.subplots(len(cards), 1, figsize=(15, 3.2 * len(cards)), sharex=True, squeeze=False)
for ax, card in zip(axes[:, 0], cards):
    for r in waits:
        if int(r['card']) == card and r['engine'] == 'HMX':
            ax.broken_barh([(float(r['start_ms']), float(r['end_ms']) - float(r['start_ms']))], (int(r['core']) + 0.0, 0.28), color=colors['hmx_wait'])
    for r in work:
        if int(r['card']) != card: continue
        c = cat(r)
        if not c: continue
        ax.broken_barh([(float(r['start_ms']), max(float(r['end_ms']) - float(r['start_ms']), 0.004))], (int(r['core']) + lanes[c], 0.28), color=colors[c])
    ax.set_ylim(-0.5, 16.5); ax.set_yticks(range(16)); ax.set_ylabel(f'card {card} core'); ax.grid(axis='x', alpha=0.3); ax.axvline(dev_end, color='k', lw=0.8, ls='--')
axes[-1, 0].set_xlabel('ms since device start'); axes[0, 0].set_title(title)
axes[0, 0].legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], ncol=9, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.35))
fig.tight_layout(); fig.savefig(outp, dpi=110); print('saved', outp)
