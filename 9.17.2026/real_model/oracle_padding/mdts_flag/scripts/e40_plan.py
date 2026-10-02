#!/usr/bin/env python3
"""Plan file for moe_qwen3_multichunk_host.cpp: capacity profiles over RankTier's tier layout (routing_counts lane order:
tier by tier, card-major within a tier) and the attempt policy. Usage:
  e40_plan.py <out> --tiers 8x128,8x48,16x16 --profile tiered:48,16 --profile full:128,128 --profile tight:48,4
              [--no-inputs] --primary tiered [--fallback full] [--force <chunk>:<profile>]...
Profile capacities list tiers 1.. (tier 0 stays at its capacity T); a profile's cap_tag length is its 1-based index.
--no-inputs writes profiles without cap_* inputs (programs compiled for a single capacity profile); --capture runs every
chunk as an independent prefill and collects the routing counts (routing capture for experiment 4b)."""
import argparse
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--tiers', required=True); ap.add_argument('--profile', action='append', required=True)
ap.add_argument('--primary', required=True); ap.add_argument('--fallback', default=None); ap.add_argument('--force', action='append', default=[]); ap.add_argument('--no-inputs', action='store_true'); ap.add_argument('--capture', action='store_true')
a = ap.parse_args()
tiers = [tuple(int(v) for v in x.split('x')) for x in a.tiers.split(',')]; assert sum(l for l, _ in tiers) == 32
names, lines = [], []
for i, spec in enumerate(a.profile):
    nm, caps = spec.split(':'); caps = [tiers[0][1]] + [int(v) for v in caps.split(',')] if caps else [c for _, c in tiers]; assert len(caps) == len(tiers)
    lane = [c for (l, _), c in zip(tiers, caps) for _ in range(4 * l)]; assert len(lane) == 128
    inputs = [] if a.no_inputs else [('cap_tag', i + 1)] + [(f'cap_rows{t}', caps[t]) for t in range(1, len(tiers))]
    names.append(nm); lines.append(f'{nm} {len(inputs)} ' + ' '.join(f'{k} {v}' for k, v in inputs) + ' ' + ' '.join(map(str, lane)))
txt = f'PROFILES {len(lines)}\n' + '\n'.join(lines) + f'\nPRIMARY {names.index(a.primary)} FALLBACK {names.index(a.fallback) if a.fallback else -1}\n'
for f in a.force: c, p = f.split(':'); txt += f'FORCE {int(c)} {names.index(p)}\n'
if a.capture: txt += 'CAPTURE\n'
open(a.out, 'w').write(txt)
