#!/bin/bash
# Sidecar for the device queue (run_e38c.sh): once the second E39 session at T=512 has started, set the queue's pause file
# so the next queue step waits; when that session is done, run the T=512 instrumented profiles (A-E at their tuned tiles,
# A at the default tile, the dropless 64-row block program and the fastest block program of the two E39 sessions) and the
# experiment 3 profiles (the four combine variants of the E36 ladder at capacity T: export combine, dense-path rewrites,
# token-owned, token-owned + elementwise final sum), then release the queue.
M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; cd $M || exit 1
until [ -f $F/e39/e39_T512_P2.txt ]; do sleep 15; done; touch $F/e38/queue.pause; echo "$(date +%F_%T) queue paused"
until grep -q E39_P2_DONE $F/e39/e39_T512_P2.txt; do sleep 20; done
best=$(/home/chihao/qeff-venv/bin/python - <<'EOP'
import re
F = '/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag/full_model/e39'
m = {}
for s in ('P1', 'P2'):
    t = open(f'{F}/e39_T512_{s}.txt').read()
    if 'session means' in t: m.update({k: float(v) for k, v in re.findall(r'(blk\S+) ([\d.]+) ms', t.split('session means (both passes): ')[1].split('\n')[0])})
k = min(m, key=m.get); lab, ts = k.rsplit('_', 1); print(f'{lab}:{ts}')
EOP
)
echo "$(date +%F_%T) fastest block program: $best"
progs="e38A:def e38A:512 e38B:1024 e38C:1024 e38D:1024 e38E:1024 blk64:1024"; [[ " $progs " == *" $best "* ]] || progs="$progs $best"
bash scripts/run_e38p.sh 512 $progs > $F/e39/e38p_T512_profiles.txt 2>&1
bash scripts/run_e38p.sh 512 e42R0:def e42R1:def e42R2:def e42R3:def > $F/e39/e42_T512_combine_profiles.txt 2>&1   # experiment 3: combine variants at capacity T
rm -f $F/e38/queue.pause; echo "$(date +%F_%T) profiles done, queue released"
