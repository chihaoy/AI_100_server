#!/usr/bin/env python3
"""Per-core activity of the E44 full-model profile, in the style of detail_plot2.py but for RankTier and one layer at a time.
Every op of the work table is drawn on its card and core, in one of three lanes per core by engine (top: tensor unit (HMX),
with its waits in light gray from waits.csv; middle: DMA, including weight gathers, multicasts and P2P; bottom: vector unit
(HVX)), colored by the part of the layer it belongs to: attention core, head-group sum, norm/router/sort, the three expert
tiers, token-owned combine, other. Thin markers at the top of each panel mark the layer's phase boundaries from
e44_layers.json (critical timeline). The first run caches the parsed tables as e44_cores_cache.npz next to the work table.
Usage: e44_cores.py <analysis dir> layer <L> <out.png> [pad ms]
       e44_cores.py <analysis dir> window <t0 ms> <t1 ms> <out.png> [layer: times relative to its start, with its phase markers]
       e44_cores.py <analysis dir> overview <out.png> [bin ms] [first layer] [last layer]   (dominant part per core and time bin)"""
import sys, os, csv, re, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
d, mode = sys.argv[1], sys.argv[2]
FAM = ['attn', 'osum', 'route', 't2', 't0', 't1', 'combine', 'other', 'wait']
NAME = dict(attn='attention core', osum='head-group sum', route='norm, router, sort', t2='tier 2: 16 lanes x 64', t0='tier 0: 8 lanes x 512',
            t1='tier 1: 8 lanes x 128', combine='token-owned combine', other='other', wait='tensor unit waiting')
COL = dict(attn='#2a78d6', osum='#eb6834', route='#1baf7a', t2='#eda100', t0='#e87ba4', t1='#008300', combine='#4a3aa7', other='#a8a69e', wait='#e4e3dc')
LANE = {'HMX': 0, 'HVX_DMA': 1, 'DMAIssue_DMA': 1, 'DMAIssue': 1, 'P2P': 1, 'HVX': 2}
LAY = re.compile(r'^/model/layers\.(\d+)/(input_layernorm|self_attn|post_attention_layernorm|mlp)/(.*)$')
def tier_of(t):
    m = re.match(r'tr(\d+)(/|_gather_|_rows)', t)
    if m: return int(m.group(1))
    if re.match(r'(MatMul|MatMul_1|MatMul_2|act_fn|Where_3|CtxGather3D|Mul_4|Mul_5)(/|$)', t): return 0
    return None
def fam(node):
    m = LAY.match(node)
    if not m: return FAM.index('other'), -1
    L, blk, t = int(m.group(1)), m.group(2), m.group(3)
    if blk == 'input_layernorm': return FAM.index('attn'), L
    if blk == 'self_attn': return FAM.index('osum' if re.search(r'ReduceSum_o|hp/osum', t) else 'attn'), L
    if blk == 'post_attention_layernorm': return FAM.index('route'), L
    ti = tier_of(t)
    if ti is not None: return FAM.index(f't{ti}'), L
    if t.startswith(('to_', 'final_', 'fa_', 'ft')): return FAM.index('combine'), L
    return FAM.index('route'), L

cache = f'{d}/e44_cores_cache.npz'
if not os.path.exists(cache):
    cols = dict(s=[], e=[], card=[], core=[], lane=[], fam=[], layer=[])
    memo = {}
    def add(path, waits):
        with open(path) as f:
            r = csv.reader(f); h = next(r); ix = {k: i for i, k in enumerate(h)}
            ic, io, ie, inode, i0, i1 = ix['card'], ix['core'], ix['engine'], ix['node'], ix['start_ms'], ix['end_ms']
            for row in r:
                c = int(row[ic])
                if c < 0: continue
                eng = row[ie]
                if waits:
                    if eng != 'HMX': continue
                    fa, L = FAM.index('wait'), -1; ln = 0
                else:
                    ln = LANE.get(eng)
                    if ln is None: continue
                    n = row[inode]; v = memo.get(n)
                    if v is None: v = memo[n] = fam(n)
                    fa, L = v
                cols['s'].append(float(row[i0])); cols['e'].append(float(row[i1])); cols['card'].append(c); cols['core'].append(int(row[io]))
                cols['lane'].append(ln); cols['fam'].append(fa); cols['layer'].append(L)
    add(f'{d}/waits.csv', True); add(f'{d}/work.csv', False)
    np.savez_compressed(cache, **{k: np.array(v, dtype=np.float64 if k in 's e' else np.int16) for k, v in cols.items()})
    print('cached', len(cols['s']), 'intervals')
Z = np.load(cache); S, E, CARD, CORE, LANEA, FAMA = Z['s'], Z['e'], Z['card'], Z['core'], Z['lane'], Z['fam']
J = json.load(open(f'{d}/e44_layers.json')); LAYERS = {o['L']: o for o in J['layers']}

def legend(fig, keys, y=0.995):
    fig.legend(handles=[Patch(color=COL[k], label=NAME[k]) for k in keys], ncol=len(keys), fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, y), frameon=False)

if mode in ('layer', 'window'):
    if mode == 'layer':
        L = int(sys.argv[3]); out = sys.argv[4]; pad = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15; o = LAYERS[L]
        t0, t1 = o['A0'] - pad, o['F1'] + pad; ref = o['A0']
        title = f"Layer {L}: every op on every core of the four SoCs (RankTier, T=512, profiled)   lanes per core: top tensor unit (gray = waiting), middle DMA, bottom vector unit"
    else:
        L = int(sys.argv[6]) if len(sys.argv) > 6 else None; o = LAYERS[L] if L is not None else None; ref = o['A0'] if o else 0.0
        t0, t1, out = ref + float(sys.argv[3]), ref + float(sys.argv[4]), sys.argv[5]
        title = (f"Layer {L}, {t0 - ref:.2f} to {t1 - ref:.2f} ms after its start" if o else f"{t0:.2f} to {t1:.2f} ms") + ": every op on every core of the four SoCs (RankTier, T=512, profiled)"
    sel = (E > t0) & (S < t1)
    fig, axes = plt.subplots(4, 1, figsize=(17, 15), sharex=True)
    for card, ax in enumerate(axes):
        m = sel & (CARD == card)
        for ln, off in ((0, 0.0), (1, 0.32), (2, 0.64)):
            for fi, k in enumerate(FAM):
                mm = m & (LANEA == ln) & (FAMA == fi)
                if not mm.any(): continue
                cs, ss, ee = CORE[mm], np.maximum(S[mm], t0) - ref, np.minimum(E[mm], t1) - ref
                for core in np.unique(cs):
                    q = cs == core; w = np.maximum(ee[q] - ss[q], 0.004)
                    ax.broken_barh(list(zip(ss[q], w)), (int(core) + off, 0.27), facecolors=COL[k], linewidth=0)
        ax.set_ylim(-0.3, 16.2); ax.set_yticks(np.arange(16) + 0.45); ax.set_yticklabels(range(16), fontsize=7.5)
        ax.set_ylabel(f'SoC {card}, core', fontsize=9); ax.grid(axis='x', alpha=0.25, lw=0.6); ax.set_xlim(t0 - ref, t1 - ref)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        if o is not None:   # phase boundaries on the critical timeline
            per = o['per_soc']; crit = max(per, key=lambda c: per[c]['tier_end']); ts = sorted(per[crit]['tiers'], key=lambda t: t['start'])
            marks = [('A0', o['A0']), ('attention done', o['A1']), ('head sum done', o['O1']), ('sort done', o['S1'])] + \
                    [(f"tier {t['tier']}", t['start']) for t in ts] + [('tiers done', o['TE']), ('final sum done', o['F1'])]
            for j, (name, t) in enumerate(marks[1:]):
                if not (t0 <= t <= t1): continue
                ax.axvline(t - ref, color='#52514e', lw=0.6, alpha=0.55)
                if card == 0: ax.text(t - ref, 16.3 + 0.75 * (j % 2), name, fontsize=7.5, ha='center', va='bottom', color='#52514e')
            if card == 0: ax.set_ylim(-0.3, 17.6)
    axes[-1].set_xlabel('ms from the start of the layer (its input norm)' if o is not None else 'ms since device start', fontsize=9)
    fig.suptitle(title, fontsize=10.5, y=0.975); legend(fig, FAM, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.savefig(out, dpi=115); print('saved', out)
elif mode == 'overview':
    out = sys.argv[3]; b = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    if len(sys.argv) > 6: lo_t, hi_t = LAYERS[int(sys.argv[5])]['A0'] - 0.1, LAYERS[int(sys.argv[6])]['F1'] + 0.1
    else: lo_t, hi_t = 0.0, float(E.max())
    tmax = hi_t - lo_t; nb = int(np.ceil(tmax / b)); acc = np.zeros((64, nb, len(FAM) - 1), np.float32)
    m = (FAMA != FAM.index('wait')) & (E > lo_t) & (S < hi_t); row = CARD[m].astype(np.int64) * 16 + CORE[m]
    s, e, f = np.maximum(S[m], lo_t) - lo_t, np.minimum(E[m], hi_t) - lo_t, FAMA[m].astype(np.int64)
    i0 = np.clip((s / b).astype(np.int64), 0, nb - 1); i1 = np.clip((e / b).astype(np.int64), 0, nb - 1)
    one = i0 == i1   # most ops fit in one bin; split the rest bin by bin
    np.add.at(acc, (row[one], i0[one], f[one]), (e - s)[one])
    for k in np.nonzero(~one)[0]:
        for j in range(i0[k], i1[k] + 1):
            lo, hi = max(s[k], j * b), min(e[k], (j + 1) * b)
            if hi > lo: acc[row[k], j, f[k]] += hi - lo
    busy = acc.sum(-1); dom = acc.argmax(-1)
    cmap = ListedColormap([COL[k] for k in FAM[:-2]] + [COL['other'], '#ffffff'])
    img = np.where(busy > 0.05 * b, np.where(dom == FAM.index('other'), len(FAM) - 2, dom), len(FAM) - 1)
    fig, ax = plt.subplots(figsize=(17, 6.2))
    ax.imshow(img, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=len(FAM) - 1, extent=(lo_t, lo_t + nb * b, 64, 0))
    for c in range(1, 4): ax.axhline(16 * c, color='#0b0b0b', lw=0.8)
    ax.set_yticks([8, 24, 40, 56]); ax.set_yticklabels([f'SoC {c}\n16 cores' for c in range(4)], fontsize=8.5)
    every = 4 if tmax > 60 else 1
    for L, o in LAYERS.items():
        if lo_t <= o['A0'] <= hi_t and L % every == 0:
            ax.text(o['A0'], -0.8, f'L{L}', fontsize=7.5, ha='left', va='bottom', color='#52514e'); ax.axvline(o['A0'], color='#52514e', lw=0.5, alpha=0.5)
    ax.set_xlabel('ms since device start', fontsize=9); ax.set_xlim(lo_t, hi_t)
    for c in range(4):
        for k in range(0, 16, 4): ax.text(hi_t + tmax * 0.003, 16 * c + k + 0.5, f'c{k}', fontsize=6.5, va='center', color='#52514e')
    span = 'Whole 512-token prefill' if tmax > 60 else f"Layers {sys.argv[5]} to {sys.argv[6]}"
    ax.set_title(f'{span}: what each core is mostly doing in each {b * 1000:.0f} µs (white = idle, under 5% busy; rows are cores 0-15 of each SoC)', fontsize=10.5, pad=14)
    legend(fig, FAM[:-1], y=1.0); fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=115); print('saved', out)
