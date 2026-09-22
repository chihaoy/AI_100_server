#!/usr/bin/env python3
"""Validate saved active-expert cohorts and plot latency, traffic and core work."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


LABELS = {'e64':'64 cold experts (control)', 'ffn22':'22 FFNs / 64 routing lanes',
          'e22':'22 FFNs / 22 routing lanes', 'e24':'24 FFNs / 24 routing lanes',
          'e32':'32 FFNs / 32 routing lanes', 'ffn24':'24 FFNs / 64 routing lanes',
          'ffn32':'32 FFNs / 64 routing lanes'}
STEM = '/model/layers.2/mlp/'


def read(path):
    return json.loads(path.read_text())


def save(fig, path):
    for suffix in ['png','pdf','svg']:
        fig.savefig(path.with_suffix('.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


def validate(root, cohorts, profiles):
    timed, saved = 0, 0
    for cohort in cohorts.values():
        if cohort['stats_level']!=0:
            raise ValueError('Host timing must have profiling disabled')
        for case in cohort['cases']:
            samples=[]
            for i in range(cohort['rounds']):
                path=root/case['case']/f'{cohort["timing_prefix"]}_r{i}'
                result=read(path/'validation.json')
                if not all(result[k] for k in ['bit_exact','counts_exact','no_overflow']):
                    raise ValueError('Unvalidated timing round')
                raw=read(path/'timing.json')['samples_ms']
                if len(raw)!=cohort['iterations'] or float(np.median(raw))!=case['host_round_medians_ms'][i]:
                    raise ValueError('Round summary differs from raw samples')
                samples.extend(raw); saved+=1
            if len(samples)!=case['host_samples'] or float(np.median(samples))!=case['host_median_ms']:
                raise ValueError('Cohort summary differs from raw samples')
            timed+=len(samples)
    for case in profiles:
        if len(case['samples'])!=3 or not read(root/case['case']/'profile/validation.json')['bit_exact_to_own_timing']:
            raise ValueError('Need three validated profiler captures per variant')
        if case['sample']!=sorted(case['samples'],key=lambda s:s['device_ms'])[1]['sample']:
            raise ValueError('Representative row must use median-device-duration sample')
        for stage in ['hot','cold']:
            total=sum(r['bytes']*r['events'] for r in case[stage+'_projection_copy_histogram'])
            if total!=case[stage+'_projection_ddr_bytes']:
                raise ValueError('DDR histogram disagrees with total')
    result=dict(timed_invocations=timed,saved_timing_outputs=saved,
                saved_profile_outputs=sum(len(c['samples']) for c in profiles),all_saved_outputs_bit_exact=True)
    (root/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def latency(ax, cases, title):
    y=np.arange(len(cases))
    values=np.array([c['host_median_ms'] for c in cases])
    colors=['#7c8994' if c['case']=='e64' else '#217f70' if c['case'].startswith('ffn') else '#b36c37'
            for c in cases]
    bars=ax.barh(y,values,color=colors)
    ax.errorbar(values,y,xerr=[values-[c['host_p10_ms'] for c in cases],
                             [c['host_p90_ms'] for c in cases]-values],fmt='none',ecolor='#263341',capsize=3)
    ax.set_yticks(y,[LABELS[c['case']] for c in cases])
    ax.invert_yaxis()
    ax.set(xlabel='Host latency (ms); bars = median, whiskers = P10–P90',title=title)
    ax.set_xlim(0,max(c['host_p90_ms'] for c in cases)*1.16)
    for bar,value,case in zip(bars,values,cases):
        ax.text(case['host_p90_ms']+.08,bar.get_y()+bar.get_height()/2,f'{value:.3f}',va='center',fontsize=9)


def plots(root, cohorts, profiles):
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    latency(axes[0,0],cohorts['screen']['cases'],'Initial screen: three rounds × 100 invocations')
    latency(axes[0,1],cohorts['confirm']['cases'],'Fresh confirmation: five rounds × 100 invocations')
    names=[c['case'] for c in profiles]
    x=np.arange(len(names)); ax=axes[1,0]
    bars=ax.bar(x,[c['cold_projection_ddr_bytes']/2**20 for c in profiles],color='#416a90')
    ax.set_xticks(x,names)
    ax.set(ylabel='Cold projection-attributed DDR copies (MiB)',title='Descriptor bytes, summed over four cards')
    for bar,case in zip(bars,profiles):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+8,
                f'{bar.get_height():.0f}',ha='center',fontsize=9)
    ax.set_ylim(0,max(bar.get_height() for bar in bars)*1.17)
    ax=axes[1,1]
    for i,(field,label,color) in enumerate([
        ('cold_span_ms','Cold HMX span','#416a90'),
        ('between_groups_ms','Between hot and cold','#a38750'),
        ('combine_span_ms','Combine span','#7d647d')]):
        ax.bar(x+(i-1)*.24,[c[field] for c in profiles],width=.24,label=label,color=color)
    ax.set_xticks(x,names)
    ax.set(ylabel='Observed window (ms)',title='Separate instrumented builds; windows are not additive costs')
    ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Qwen3 layer 2: omit inactive experts, hot C128 / cold C2',fontsize=15)
    save(fig,root/'active_oracle')

    selected=[next(c for c in profiles if c['case']==name) for name in ['e64','ffn22','e22']]
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
    colors={'hot':'#416a90','cold':'#bd713b','local reduction':'#806495'}
    for ax,case in zip(axes,selected):
        with (root/case['case']/'analysis/work.csv').open() as stream:
            work=list(csv.DictReader(stream))
        for stage,suffixes in [('hot',['MatMul','MatMul_1','MatMul_2']),
                               ('cold',['MatMul_3','MatMul_4','MatMul_5']),
                               ('local reduction',[f'reduce_tile_{i}' for i in range(16)])]:
            wanted={STEM+s for s in suffixes}
            engine,kind=('HVX','aicbatchedreduceadd') if stage=='local reduction' else ('HMX','aicconvolutiond32')
            rows=[r for r in work if int(r['sample'])==case['sample'] and r['node'] in wanted
                  and r['engine']==engine and r['kind']==kind]
            ax.barh([int(r['card'])*16+int(r['core']) for r in rows],
                    [float(r['duration_us'])/1000 for r in rows],
                    left=[float(r['start_us'])/1000-case['device_start_ms'] for r in rows],
                    height=.9,color=colors[stage],linewidth=0)
        for boundary in [15.5,31.5,47.5]:
            ax.axhline(boundary,color='#adb5bd',lw=.6)
        ax.set_yticks([7.5,23.5,39.5,55.5],['Card 0','Card 1','Card 2','Card 3'])
        ax.set_ylim(63.8,-.8)
        ax.set_title(f'{LABELS[case["case"]]}: device {case["device_ms"]:.3f} ms; '
                     f'cold HMX executes on {case["cold_cores"]}/64 cores',loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlabel('Time since earliest core execution start (ms); each stripe is one core')
    fig.legend(handles=[Patch(color=color,label=stage.title()) for stage,color in colors.items()],
               loc='lower center',bbox_to_anchor=(.5,-.035),frameon=False,ncol=3)
    axes[-1].set_xlim(0,max(c['device_ms'] for c in selected)*1.02)
    fig.suptitle('Expert HMX and local-reduction HVX kernels: four cards / 16 cores per card\n'
                 'Blank intervals can contain dependencies or other kernels',fontsize=13)
    save(fig,root/'active_oracle_cores')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    args=p.parse_args()
    cohorts={name:read(args.root/f'{name}.json') for name in ['screen','confirm']}
    profiles=read(args.root/'profiles.json')['cases']
    validate(args.root,cohorts,profiles)
    plots(args.root,cohorts,profiles)


if __name__=='__main__':
    main()
