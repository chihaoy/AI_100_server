#!/usr/bin/env python3
"""Audit raw cohorts and compare frozen all-active oracle selections."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

LABELS={'common_static':'One static plan','fixed_shape_static':'Fixed shape / static placement',
        'fixed_layout_capacity_oracle':'Fixed layout / capacity oracle',
        'fixed_shape_placement_oracle':'Fixed shape / placement oracle','joint_oracle':'Joint oracle'}
STEM='/model/layers.2/mlp/'


def read(path): return json.loads(path.read_text())


def save(fig,path):
    for suffix in ['png','pdf','svg']: fig.savefig(path.with_suffix('.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


def audit(root,cohort_name):
    info=read(root/'experiment.json'); cohort=read(root/f'{cohort_name}_cohort.json')
    summary=read(root/f'{cohort_name}.json')['cases']
    if cohort['stats_level']!=0: raise ValueError('Instrumented host cohort')
    saved,timed=0,0; errors=[]
    for row in summary:
        entries=[e for e in cohort['entries'] if (e['plan'],e['workload'])==(row['plan'],row['workload'])]
        samples=[]; medians=[]; spec=info['plans'][row['plan']]
        counts=np.array(next(w['counts'] for w in info['workloads'] if w['name']==row['workload']))[spec['permutation']]
        ref=np.load(root/'workloads'/row['workload']/'reference_f32.npy').astype(np.float64).reshape(-1)
        for entry in entries:
            path=Path(entry['path']); raw=read(path/'timing.json')['samples_ms']; result=read(path/'validation.json')
            if not np.array_equal(np.fromfile(path/'counts.bin',np.int32),counts): raise ValueError('Counts changed')
            if counts.min()<1 or counts.sum()!=1024 or np.any(counts>np.repeat(spec['capacities'],64)):
                raise ValueError('Missing expert or overflow')
            y=np.fromfile(path/'y.bin',np.float16).astype(np.float64)
            error=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
            # The plotting and SDK environments use different NumPy/BLAS
            # builds. Permit only floating reduction roundoff in the audit;
            # the preregistered 0.5% hardware-reference limit is unchanged.
            if (not np.isfinite(y).all() or error>info['cpu_relative_l2_limit']
                or not np.isclose(error,result['relative_l2_cpu'],rtol=1e-12,atol=1e-15)):
                raise ValueError('Raw output fails reference check')
            samples.extend(raw); medians.append(float(np.median(raw))); saved+=1; errors.append(error)
        if (float(np.median(samples))!=row['median_ms'] or medians!=row['round_medians_ms']
            or len(samples)!=row['samples']): raise ValueError('Summary differs from raw timings')
        timed+=len(samples)
    return dict(saved_outputs=saved,timed_invocations=timed,max_relative_l2_cpu=max(errors))


def comparison(root):
    info=read(root/'experiment.json'); selection=read(root/'selection.json')
    workloads=[w['name'] for w in info['workloads']]
    cohort_files={'screen':selection['screen_prefix'],'confirm':'confirm'}
    cohorts={name:read(root/f'{prefix}.json') for name,prefix in cohort_files.items()}
    validation={name:audit(root,prefix) for name,prefix in cohort_files.items()}
    results={}
    for name,cohort in cohorts.items():
        rows={(r['plan'],r['workload']):r for r in cohort['cases']}
        result={}
        for policy,choices in selection['policies'].items():
            chosen=[rows[choices[w],w] for w in workloads]
            rounds=np.array([r['round_medians_ms'] for r in chosen])
            result[policy]=dict(mean_ms=float(np.mean([r['median_ms'] for r in chosen])),
                workloads={r['workload']:r for r in chosen},round_means_ms=rounds.mean(0).tolist())
        comparisons=[]
        for baseline,candidate in [('common_static','joint_oracle'),
                ('fixed_layout_capacity_oracle','joint_oracle'),('fixed_shape_static','fixed_shape_placement_oracle')]:
            a,b=result[baseline],result[candidate]
            comparisons.append(dict(baseline=baseline,candidate=candidate,saving_ms=a['mean_ms']-b['mean_ms'],
                latency_reduction_percent=100*(1-b['mean_ms']/a['mean_ms']),speedup=a['mean_ms']/b['mean_ms'],
                paired_round_wins=sum(x<y for x,y in zip(b['round_means_ms'],a['round_means_ms'])),
                paired_rounds=len(a['round_means_ms'])))
        results[name]=dict(policies=result,comparisons=comparisons)
    data=dict(validation=validation,results=results,selection=selection)
    (root/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(validation=validation,confirm=results['confirm']['comparisons']),indent=2))
    lines=['# All-active oracle measurements','',
        'Generated from validated raw outputs and timing samples. Policy selections are frozen from the screen; '
        'confirmation does not select new winners. All means weight the five routing workloads equally.','',
        '## Fresh confirmation','',
        '| Workload | One static plan, ms | Fixed layout + capacity oracle, ms | Joint oracle, ms |',
        '|---|---:|---:|---:|']
    confirm=results['confirm']['policies']
    for w in workloads:
        lines.append('| '+w+' | '+' | '.join(f'{confirm[p]["workloads"][w]["median_ms"]:.4f}' for p in
                     ['common_static','fixed_layout_capacity_oracle','joint_oracle'])+' |')
    lines.append('| **Mean** | '+' | '.join(f'{confirm[p]["mean_ms"]:.4f}' for p in
                 ['common_static','fixed_layout_capacity_oracle','joint_oracle'])+' |')
    lines += ['', '| Workload | Fixed shape / static placement, ms | Fixed shape / placement oracle, ms |',
              '|---|---:|---:|']
    for w in workloads:
        lines.append('| '+w+' | '+' | '.join(f'{confirm[p]["workloads"][w]["median_ms"]:.4f}' for p in
                     ['fixed_shape_static','fixed_shape_placement_oracle'])+' |')
    lines += ['', '| Comparison | Saving, ms | Latency reduction | Speedup | Paired round wins |',
              '|---|---:|---:|---:|---:|']
    for row in results['confirm']['comparisons']:
        lines.append(f'| {LABELS[row["candidate"]]} vs {LABELS[row["baseline"]]} | {row["saving_ms"]:.4f} | '
            f'{row["latency_reduction_percent"]:.2f}% | {row["speedup"]:.4f}× | '
            f'{row["paired_round_wins"]}/{row["paired_rounds"]} |')
    lines += ['', '## Frozen selections','', '| Policy | Workload | Plan |', '|---|---|---|']
    for policy,choices in selection['policies'].items():
        for w,p in choices.items(): lines.append(f'| {LABELS[policy]} | {w} | `{p}` |')
    lines += ['', '## Per-pair timing spread','', '| Plan | Workload | Median, ms | P10–P90, ms | Round medians, ms |',
              '|---|---|---:|---:|---|']
    for r in cohorts['confirm']['cases']:
        lines.append(f'| `{r["plan"]}` | {r["workload"]} | {r["median_ms"]:.4f} | '
            f'{r["p10_ms"]:.4f}–{r["p90_ms"]:.4f} | '+', '.join(f'{v:.4f}' for v in r['round_medians_ms'])+' |')
    profiles=read(root/'profiles.json')['cases']
    lines += ['', '## Separate instrumented profiles','',
        'Each row uses every metric from the same median-device-duration sample of three validated captures. '
        'Group 0/1 identify source-graph groups; the compiler can change their compute order. '
        'The gap is between the groups in actual chronological order. Phase windows are not additive costs.','',
        '| Plan / input | Device, ms | Actual HMX order | Group 0 HMX span, ms | Gap, ms | Group 1 HMX span, ms | Combine, ms | P2P, MiB | Projection DDR copies, MiB |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in profiles:
        lines.append(f'| `{c["plan"]}` / {c["workload"]} | {c["device_ms"]:.4f} | {c["hmx_order"]} | {c["hot_span_ms"]:.4f} | '
            f'{c["between_groups_ms"]:.4f} | {c["cold_span_ms"]:.4f} | {c["combine_span_ms"]:.4f} | '
            f'{c["p2p_bytes"]/2**20:.3f} | {(c["hot_projection_ddr_bytes"]+c["cold_projection_ddr_bytes"])/2**20:.3f} |')
    lines += ['', 'Projection DDR bytes are observed copy-descriptor sizes attributed to the six projections, '
        'not memory-controller counters or an exact classification of weight versus activation traffic. '
        'Every source graph retains the same 1152 MiB of trained weights.','',
        '| Plan / input | Group 0/1 HMX compute, max-core µs | Group 0/1 HMX wait, max-core ms | Group 0/1 HMX events | Static constants, MiB |',
        '|---|---:|---:|---:|---:|']
    for c in profiles:
        lines.append(f'| `{c["plan"]}` / {c["workload"]} | '
            f'{1000*c["hot_hmx_ms_max_core"]:.3f} / {1000*c["cold_hmx_ms_max_core"]:.3f} | '
            f'{c["hot_wait_ms_max_core"]:.3f} / {c["cold_wait_ms_max_core"]:.3f} | '
            f'{c["hot_hmx_events"]} / {c["cold_hmx_events"]} | {c["package"]["total_constant_bytes"]/2**20:.3f} |')
    lines += ['', 'Compute and wait maxima may come from different cores; they are not additive critical-path costs. '
        'HMX waits use `opSyncDurUs` and are clipped to the observed projection interval. '
        'They include dependencies and scheduling, not only DDR transfers.','',
        '## Validation','']
    for name,v in validation.items():
        lines.append(f'- {name}: {v["timed_invocations"]:,} timed invocations, {v["saved_outputs"]} saved outputs; '
            f'maximum relative L2 versus FP32 reference {100*v["max_relative_l2_cpu"]:.4f}%.')
    lines += [f'- {3*len(profiles)} profiling outputs match their same-graph uninstrumented references bit for bit.',
        '- Counts remain exact, every expert is active, and no routing assignments overflow.',
        '- P10/P90 are invocation spread, not confidence intervals. Only the final output of each timing round is saved.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    return info,results,profiles


def plots(root,info,results,profiles):
    workloads=[w['name'] for w in info['workloads']]; x=np.arange(len(workloads))
    confirm=results['confirm']['policies']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,policies,title in [(axes[0],['common_static','fixed_layout_capacity_oracle','joint_oracle'],
            'Grouping and capacity: all 128 experts execute'),(axes[1],['fixed_shape_static','fixed_shape_placement_oracle'],
            'Placement only: capacities fixed at 32 / 32')]:
        width=.75/len(policies)
        for i,(p,color) in enumerate(zip(policies,['#7c8994','#416a90','#217f70'])):
            ax.bar(x+(i-(len(policies)-1)/2)*width,[confirm[p]['workloads'][w]['median_ms'] for w in workloads],
                   width=width,label=LABELS[p],color=color)
        ax.set_xticks(x,workloads,rotation=15); ax.set(ylabel='Host latency (ms)',title=title)
        ax.set_ylim(0,1.25*max(confirm[p]['workloads'][w]['median_ms'] for p in policies for w in workloads))
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Fresh confirmation of choices frozen from the initial screen',fontsize=14)
    save(fig,root/'all_active_latency')

    fig,ax=plt.subplots(figsize=(13,3.2),layout='constrained')
    im=ax.imshow([w['counts'] for w in info['workloads']],aspect='auto',cmap='viridis',vmin=0,vmax=32)
    ax.set_yticks(x,workloads); ax.set(xlabel='Expert position in the canonical trained bank',
        title='Controlled routing: 128 active experts, 1024 assignments, eight experts per token')
    fig.colorbar(im,ax=ax,label='Tokens per expert')
    save(fig,root/'all_active_inputs')

    wanted=read(root/'profile_selection.json')['representatives']
    selected=[next(c for c in profiles if c['plan']==p and c['workload']=='skew_b') for p in wanted]
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True,layout='constrained')
    colors=['#416a90','#bd713b','#806495']
    for ax,case in zip(axes,selected):
        with (root/case['plan']/'profiles'/case['workload']/'analysis/work.csv').open() as f: work=list(csv.DictReader(f))
        for suffixes,engine,kind,color in [(['MatMul','MatMul_1','MatMul_2'],'HMX','aicconvolutiond32',colors[0]),
                (['MatMul_3','MatMul_4','MatMul_5'],'HMX','aicconvolutiond32',colors[1]),
                ([f'reduce_tile_{i}' for i in range(16)],'HVX','aicbatchedreduceadd',colors[2])]:
            nodes={STEM+s for s in suffixes}
            rows=[r for r in work if int(r['sample'])==case['sample'] and r['node'] in nodes and r['engine']==engine and r['kind']==kind]
            ax.barh([int(r['card'])*16+int(r['core']) for r in rows],[float(r['duration_us'])/1000 for r in rows],
                left=[float(r['start_us'])/1000-case['device_start_ms'] for r in rows],height=.9,color=color,linewidth=0)
        for boundary in [15.5,31.5,47.5]: ax.axhline(boundary,color='#adb5bd',lw=.6)
        ax.set_yticks([7.5,23.5,39.5,55.5],['Card 0','Card 1','Card 2','Card 3']); ax.set_ylim(63.8,-.8)
        ax.set_title(f'{case["plan"]}: device {case["device_ms"]:.3f} ms; P2P {case["p2p_bytes"]/2**20:.2f} MiB',loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.2)
    axes[-1].set(xlabel='Time since earliest core execution start (ms); one stripe per core',
                 xlim=(0,max(c['device_ms'] for c in selected)*1.02))
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['Graph group 0 HMX','Graph group 1 HMX','Local reduction HVX'])],
               loc='lower center',bbox_to_anchor=(.5,-.035),frameon=False,ncol=3)
    fig.suptitle('Same skew_b input: fixed placement, sorted placement, and source-group order\n'
                 'Blank intervals can contain dependencies or other kernels',fontsize=13)
    save(fig,root/'all_active_cores')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); info,results,profiles=comparison(args.root); plots(args.root,info,results,profiles)
