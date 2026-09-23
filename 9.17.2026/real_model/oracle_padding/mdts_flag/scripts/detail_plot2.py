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
    if k == 'blockdequantize_mxfp6': return 'dequant'
    if n.startswith('tc_g_'): return 'tc_gather'
    if n.startswith(('tc_gm_', 'tc_mk_', 'tc_idx_', 'tc_add_', 'tc_rs_', 'tc_mask', 'tc_slot', 'tc_stage1')): return 'tc_mask_add'
    if n.startswith('tc_reduce_tile_'): return 'lane_reduce'
    if n.startswith('final_tile_') or n == 'Einsum_4' or n.startswith('final_slice_'): return 'card_reduce'
    if n == 'ConstantOfShape_1' and k == 'splat': return 'old_splat'
    if n.startswith('CtxScatter3D') and 'Int' not in n: return 'old_scatter'
    if n.startswith(('reduce_tile_', 'reduce_slice_')) or n == 'Einsum_3' or n == 'Reshape_6': return 'old_tiles'
    if k == 'aicbatchedreduceadd': return 'reduce'
    if e.startswith('HVX') and 'multicast' in k: return 'multicast'
    if e.startswith('HVX'): return 'hvx_other'
    if e.startswith('DMAIssue'): return 'dma_other'
    return None
colors = dict(gemm_hot='#1f77b4', gemm_cold='#ff7f0e', gemm_other='#17becf', weight_dma='#9467bd', dequant='#8c564b', tc_gather='#e377c2', tc_mask_add='#f7b6d2', lane_reduce='#2ca02c', card_reduce='#d62728', old_splat='#7f7f7f', old_scatter='#ffbb78', old_tiles='#98df8a', reduce='#2ca02c', multicast='#c49c94', hvx_other='#bcbd22', dma_other='#c7c7c7', hmx_wait='#d0d0d0')
lanes = dict(hmx_wait=0.0, gemm_hot=0.0, gemm_cold=0.0, gemm_other=0.0, weight_dma=0.3, dma_other=0.3, old_scatter=0.3, tc_gather=0.3, dequant=0.6, tc_mask_add=0.6, lane_reduce=0.6, card_reduce=0.6, old_splat=0.6, old_tiles=0.6, reduce=0.6, hvx_other=0.6, multicast=0.6)
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
fig.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], ncol=9, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.995))
fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(outp, dpi=110); print('saved', outp)
