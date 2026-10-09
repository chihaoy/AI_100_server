#!/usr/bin/env python3
"""Figures for the MLSys draft (paper/main.tex). Usage: make_figures.py <mdts_flag dir> <out dir>

fig_rank_identity.pdf  rows a card must reserve at each position of its 32 experts: fixed expert order vs run-time rank
                       (routing captures under realcase/routing*, T=128 and T=512)
fig_ladder.pdf         the ladder from the joint fit of E46-E55 (paper/paper_numbers.json from scripts/paper_numbers.py), relative to production
fig_interaction.pdf    capacity x aggregation: E50's 2x2 at T = 128, 256, 512 (paper_numbers.json)
fig_ownership.pdf      equivalent arithmetic, different placement: E52's ownership representations against step 5 (latency from paper_numbers.json,
                       cross-SoC bytes per MoE layer from the instrumented two-layer profiles full_model/e48|e52/e4*_T512_profiles.txt)
fig_stages.pdf         GEMM window of each expert stage in execution order, two-layer T=512 profiles (README 3.30, E34)
"""
import sys, os, re, glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 7.5,
                     'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.1,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
BLUE, RED, GREY, LBLUE = '#1f5aa6', '#c0392b', '#7f7f7f', '#7fa6d6'
TIER_COL = ['#1f5aa6', '#4f8fd1', '#a9c8ea']


def counts(npz):   # [48, 128] tokens per expert and layer
    r = np.load(npz)['routing_idx'].astype(np.int64)
    return np.array([np.bincount(r[L].ravel(), minlength=128) for L in range(48)])


def by_rank(C):    # C [n, 48, 128] -> [n * 48 * 4, 32], each card's 32 counts sorted descending (native quarters)
    return -np.sort(-C.reshape(C.shape[0], 48, 4, 32), axis=-1).reshape(-1, 32)


def by_fixed_order(cal_mean, E):   # order each card's experts by a calibration mean, read evaluation counts at that position
    order = np.argsort(-cal_mean.reshape(48, 4, 32), axis=-1)
    return np.take_along_axis(E.reshape(E.shape[0], 48, 4, 32), order[None], axis=-1).reshape(-1, 32)


# ---- routing captures
W = {}
for d in sorted(glob.glob(f'{R}/realcase/routing*')):
    fs = sorted(glob.glob(f'{d}/T128_*.npz'), key=lambda f: (len(f), f))
    if fs: W[d] = np.array([counts(f) for f in fs])
cal = np.concatenate([v[:len(v) // 2] for v in W.values()]); ev = np.concatenate([v[len(v) // 2:] for v in W.values()])
fixed128 = by_fixed_order(cal.mean(0), ev); rank128 = by_rank(ev)
C512 = np.array([counts(f) for d in sorted(glob.glob(f'{R}/realcase/routing*')) for f in sorted(glob.glob(f'{d}/T512_*.npz'))])
fixed512 = np.concatenate([by_fixed_order(np.delete(C512, i, 0).mean(0), C512[i:i + 1]) for i in range(len(C512))])  # leave one out
rank512 = by_rank(C512)
print(f'T=128: {len(cal)} calibration / {len(ev)} evaluation prompts; T=512: {len(C512)} chunks (leave-one-out order)')
for T, f, r in ((128, fixed128, rank128), (512, fixed512, rank512)):
    print(f'  T={T} fixed order, max per position: {f.max(0).tolist()}  sum {f.max(0).sum()} of {32 * T}')
    print(f'  T={T} run-time rank, max per rank:    {r.max(0).tolist()}  sum {r.max(0).sum()}')

TIERS = {128: [(8, 128), (8, 48), (16, 16)], 512: [(8, 512), (8, 128), (16, 64)]}
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
x = np.arange(1, 33)
for ax, T, f, r, tag in ((axes[0], 128, fixed128, rank128, '(a) $T=128$, 200 held-out prompts'),
                         (axes[1], 512, fixed512, rank512, '(b) $T=512$, 20 chunks')):
    caps = np.concatenate([[c] * n for n, c in TIERS[T]])
    ax.axhline(T, color=GREY, lw=0.7, ls=':', zorder=1)
    ax.step(x, caps, where='mid', color='black', lw=0.9, zorder=3, label='RankTier tier capacity')
    ax.plot(x, f.max(0), color=RED, marker='o', ms=1.8, lw=0.9, zorder=4, label='fixed expert order, max')
    ax.plot(x, r.max(0), color=BLUE, marker='s', ms=1.8, lw=0.9, zorder=5, label='run-time rank, max')
    ax.plot(x, np.median(r, 0), color=LBLUE, ls='--', lw=0.8, zorder=4, label='run-time rank, median')
    ax.set_xlim(0.5, 32.5); ax.set_ylim(0, T * 1.06); ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_yticks(np.linspace(0, T, 5)); ax.set_title(tag, pad=2)
    ax.set_xlabel('position among a card\'s 32 experts', labelpad=1)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
axes[0].set_ylabel('tokens routed to the expert', labelpad=1)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.17), handlelength=1.6, columnspacing=1.0)
fig.subplots_adjust(wspace=0.32)
fig.savefig(f'{OUT}/fig_rank_identity.pdf'); plt.close(fig)

# ---- the ladder, from the joint fit
NUM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_numbers.json')))
steps = ['prod', 's2', 's2os', 's3sos', 's3os', 's4os', 's5os', 'Dos', 'Eos']
labels = ['Production compile', '+ expert/head-parallel (naive $T/T$)', '+ elementwise head-group sum', '+ tree prefix scans',
          '+ dense-combine repairs', '+ token-owned aggregation', '+ elementwise final sum', '+ rank sort, two tiers', '+ three tiers (RankTier)']
Ts, cols, hgt = ('128', '256', '512'), ('#a9c8ea', '#4f8fd1', '#1f3f7a'), 0.26
HAVE_PROD = all('prod' in NUM[T]['fit'] for T in Ts)
if not HAVE_PROD: print('fig_ladder.pdf skipped: production (E55) is not in paper_numbers.json yet')
fig, ax = plt.subplots(figsize=(3.3, 2.9))
y = np.arange(len(steps))[::-1]
for j, T in enumerate(Ts if HAVE_PROD else ()):
    fit = NUM[T]['fit']; base = fit[steps[0]]
    v = np.array([fit[s] for s in steps]) / base
    ax.barh(y + (1 - j) * hgt, v, height=hgt, color=cols[j], label=f'$T={T}$', zorder=3)
    for yi, vi, s in zip(y, v, steps):
        lab = f'{fit[s]:.0f} ms' + (f', {base / fit[s]:.2f}$\\times$' if s == steps[-1] else '')
        ax.text(vi + 0.012, yi + (1 - j) * hgt, lab, va='center', ha='left', fontsize=5.0, color='#333333')
ax.axvline(1.0, color=GREY, lw=0.6, ls=':', zorder=2)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.2)
ax.set_xlim(0, 1.36); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax.set_xlabel('prefill latency relative to the production compile', labelpad=1)
for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
ax.tick_params(axis='y', length=0, pad=2); ax.tick_params(axis='x', length=2, pad=1.5)
ax.legend(loc='lower right', frameon=False, ncol=1, handlelength=1.2, borderaxespad=0.2)
if HAVE_PROD: fig.savefig(f'{OUT}/fig_ladder.pdf')
plt.close(fig)

# ---- capacity x aggregation (E50)
DENSE, OWNED = '#9e9e9e', '#1f5aa6'
fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.45), sharey=False)
for ax, T in zip(axes, Ts):
    t = NUM[T]['twobytwo']; vals = [t['dense_uniform'], t['dense_tiers'], t['owned_uniform'], t['owned_tiers']]
    xs = [0, 0.9, 2.2, 3.1]; colors = [DENSE, DENSE, OWNED, OWNED]
    bars = ax.bar(xs, vals, width=0.8, color=colors, zorder=3)
    for b in (bars[1], bars[3]): b.set_hatch('////'); b.set_edgecolor('white'); b.set_linewidth(0)
    for x0, (a, b) in ((0.45, (vals[0], vals[1])), (2.65, (vals[2], vals[3]))):
        ch = 100 * (b / a - 1); ax.text(x0, max(a, b) * 1.03, f'{ch:+.0f}%'.replace('-', '\u2212'), ha='center', va='bottom', fontsize=6.0,
                                       color=RED if ch > 0 else BLUE)
    ax.set_xticks([0.45, 2.65]); ax.set_xticklabels(['dense\n(repaired)', 'token-\nowned'], fontsize=6.0)
    ax.set_ylim(0, max(vals) * 1.22); ax.set_title(f'$T={T}$', pad=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
axes[0].set_ylabel('full-model prefill, ms', labelpad=1)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=GREY, label='uniform capacity $T$'), Patch(facecolor=GREY, hatch='////', edgecolor='white', label='ranked tiers')],
           loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.53, 1.13), handlelength=1.4, columnspacing=1.2)
fig.subplots_adjust(wspace=0.45)
fig.savefig(f'{OUT}/fig_interaction.pdf'); plt.close(fig)

# ---- equivalent arithmetic, different placement (E52; traffic from the instrumented two-layer profiles)
mib = {}
for f in (f'{R}/full_model/e48/e48_T512_profiles.txt', f'{R}/full_model/e52/e52_T512_profiles.txt'):
    for line in open(f):
        m = re.search(r'T=512 (p2\w+) def, layer 1: MoE ops .*\(([\d.]+) ms\).*P2P within the MoE ([\d.]+) MiB', line)
        if m: mib[m.group(1)] = float(m.group(3))
rows = [('SoC as a leading batch\naxis (RankTier)', 's5os', 'p2D'), ('local top-k:\nexplicit per-SoC slices', 'ls', 'p2ls'),
        ('local top-k:\nglobal row indices', 'lg', 'p2lg')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.4, 1.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
yy = np.arange(len(rows))[::-1]
for j, T in enumerate(Ts):
    v = [NUM[T]['vs_D'].get(k, 1.0) if k != 's5os' else 1.0 for _, k, _ in rows]
    a1.barh(yy + (1 - j) * 0.26, v, height=0.26, color=cols[j], label=f'$T={T}$', zorder=3)
a1.set_yticks(yy); a1.set_yticklabels([r[0] for r in rows], fontsize=6.0); a1.set_xlabel('latency relative to the SoC axis', labelpad=1)
a1.set_xlim(0, 5.8); a1.axvline(1.0, color=GREY, lw=0.6, ls=':')
h_, l_ = a1.get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.12), handlelength=1.2, columnspacing=1.0)
a2.barh(yy, [mib[r[2]] for r in rows], height=0.5, color=[OWNED, RED, RED], zorder=3)
for yi, r in zip(yy, rows): a2.text(mib[r[2]] * 1.15, yi, f'{mib[r[2]]:.0f}', va='center', fontsize=6.0)
a2.set_xscale('log'); a2.set_xlim(5, 12000); a2.set_yticks([]); a2.set_xlabel('cross-SoC MiB per layer', labelpad=1)
for ax in (a1, a2):
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=2); ax.tick_params(axis='x', length=2, pad=1.5)
fig.subplots_adjust(wspace=0.08)
fig.savefig(f'{OUT}/fig_ownership.pdf'); plt.close(fig)

# ---- stage windows at T=512 (README 3.30: GEMM window of each stage, in execution order, ranges over SoCs)
progs = [('two stages\n(run-time sort)', [('16$\\times$512', 2.18, 2.20), ('16$\\times$64', 1.03, 1.31)]),
         ('three tiers\n(RankTier)', [('16$\\times$64', 0.59, 0.61), ('8$\\times$512', 0.79, 0.97), ('8$\\times$128', 0.65, 0.72)]),
         ('four tiers', [('16$\\times$64', 0.56, 0.63), ('8$\\times$128', 0.62, 0.75), ('4$\\times$512', 0.56, 0.58), ('4$\\times$240', 0.43, 0.46)])]
shade = {'512': '#1f5aa6', '240': '#4f8fd1', '128': '#7fa6d6', '64': '#c9dcf0'}
fig, ax = plt.subplots(figsize=(3.3, 1.05))
for i, (name, st) in enumerate(progs):
    t0, yi = 0.0, len(progs) - 1 - i
    for lab, lo, hi in st:
        mid = (lo + hi) / 2; cap = lab.split('times$')[1]
        ax.barh(yi, mid, left=t0, height=0.62, color=shade[cap], edgecolor='white', lw=0.8, zorder=3)
        ax.text(t0 + mid / 2, yi, lab, ha='center', va='center', fontsize=6.0, color='white' if cap in ('512', '240') else 'black', zorder=5)
        t0 += mid
    ax.text(t0 + 0.04, yi, f'{t0:.2f} ms', va='center', ha='left', fontsize=6.2)
ax.set_yticks(range(len(progs))); ax.set_yticklabels([p[0] for p in progs][::-1], fontsize=6.6)
ax.set_xlim(0, 3.9); ax.set_xlabel('sum of per-stage GEMM windows, ms (lanes per SoC $\\times$ capacity)', labelpad=1)
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0, pad=2); ax.tick_params(axis='x', length=2, pad=1.5)
fig.savefig(f'{OUT}/fig_stages.pdf'); plt.close(fig)
print('figures written to', OUT)
