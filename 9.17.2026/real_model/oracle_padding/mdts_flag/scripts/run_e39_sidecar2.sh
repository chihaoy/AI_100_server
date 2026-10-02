#!/bin/bash
# Second sidecar for the device queue (run_e38c.sh), holding the queue's pause file set by the first one: after the T=512
# profiles of A-E and the block programs (run_e38p.sh, started by run_e38p_sidecar.sh) finish, profile the experiment 3
# combine variants, run the third E39 session at T=512 (every block variant with --direct-gather, which removes the per-lane
# Expand of the hidden states and cut the 64-row dropless program from 36.6 to 24.6 ms in the second session), profile the
# 64-row dropless direct-gather program and the fastest direct-gather block program, then release the queue.
M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; cd $M || exit 1
until grep -q E38P_DONE $F/e39/e38p_T512_profiles.txt 2>/dev/null; do sleep 20; done; echo "$(date +%F_%T) A-E and block profiles done"
bash scripts/run_e38p.sh 512 e42R0:def e42R1:def e42R2:def e42R3:def > $F/e39/e42_T512_combine_profiles.txt 2>&1; echo "$(date +%F_%T) combine profiles done"
bash scripts/run_e39.sh 512 P3 e38E:1024 blk16dg:def blk16caldg:def blk16orcdg:def blk32dg:def blk32caldg:def blk32orcdg:def blk64dg:def blk64caldg:def blk64orcdg:def blk128dg:def blk128orcdg:def > $F/e39/e39_T512_P3.txt 2>&1
echo "$(date +%F_%T) $(grep 'session means' $F/e39/e39_T512_P3.txt | cut -c1-400)"
best=$(/home/chihao/qeff-venv/bin/python -c "
import re; t = open('$F/e39/e39_T512_P3.txt').read().split('session means (both passes): ')[1].split('\n')[0]
m = {k: float(v) for k, v in re.findall(r'(blk\S+) ([\d.]+) ms', t)}; k = min(m, key=m.get); print(k.rsplit('_', 1)[0] + ':' + k.rsplit('_', 1)[1])")
progs="blk64dg:def"; [ "$best" != "blk64dg:def" ] && progs="$progs $best"
bash scripts/run_e38p.sh 512 $progs > $F/e39/e38p_T512_dg_profiles.txt 2>&1; echo "$(date +%F_%T) direct-gather profiles done ($progs)"
rm -f $F/e38/queue.pause; echo "$(date +%F_%T) queue released"
