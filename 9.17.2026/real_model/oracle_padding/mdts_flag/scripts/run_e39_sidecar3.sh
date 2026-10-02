#!/bin/bash
# Third sidecar for the device queue (run_e38c.sh), holding the queue's pause file: after the experiment 3 combine profiles
# (run_e38p.sh on the E36 ladder's combine variants) finish, run the third E39 session at T=512, which gives the block
# baseline the two lowering fixes found in the profiles: --direct-gather (no per-lane Expand of the hidden states) for every
# block variant, and --block-pad invalid (padded rows and unused slots index the export's INT32_MAX empty-position marker,
# as RankTier's padded rows do); then profile the dropless 64-row programs with each fix and the fastest block program, and
# release the queue.
M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; cd $M || exit 1
until grep -q E38P_DONE $F/e39/e42_T512_combine_profiles.txt 2>/dev/null; do sleep 20; done; echo "$(date +%F_%T) combine profiles done"
bash scripts/run_e39.sh 512 P3 e38E:1024 blk16dg:def blk16caldg:def blk16orcdg:def blk32dg:def blk32caldg:def blk32orcdg:def blk64dg:def blk64caldg:def blk64orcdg:def blk128dg:def blk128orcdg:def blk64iv:def blk128iv:def blk64orciv:def blk128orciv:def > $F/e39/e39_T512_P3.txt 2>&1
echo "$(date +%F_%T) $(grep 'session means' $F/e39/e39_T512_P3.txt | cut -c1-600)"
best=$(/home/chihao/qeff-venv/bin/python -c "
import re; t = open('$F/e39/e39_T512_P3.txt').read().split('session means (both passes): ')[1].split('\n')[0]
m = {k: float(v) for k, v in re.findall(r'(blk\S+) ([\d.]+) ms', t)}; k = min(m, key=m.get); print(k.rsplit('_', 1)[0] + ':' + k.rsplit('_', 1)[1])")
progs="blk64dg:def blk64iv:def"; [[ " $progs " == *" $best "* ]] || progs="$progs $best"
bash scripts/run_e38p.sh 512 $progs > $F/e39/e38p_T512_fix_profiles.txt 2>&1; echo "$(date +%F_%T) block-fix profiles done ($progs)"
rm -f $F/e38/queue.pause; echo "$(date +%F_%T) queue released"
