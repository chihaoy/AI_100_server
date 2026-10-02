#!/bin/bash
# Device queue for experiments 1-4 (RANKTIER_EXPERIMENT_PLAN.md) after the T=512 sessions S1-S4. Every step skips itself
# when its log already holds its done marker, so the queue can be restarted at any time.
#   E40a two-layer overflow mechanism; S5 (T=512 common setting); E41 routing capture (experiment 4b) on the static program
#   of each chunk length; E40c two-chunk check on the A and E programs of S5 (T=512), S1 (T=256) and S2 (T=128); E39 at
#   T=512 (three sessions, run by hand and by the sidecars run_e38p_sidecar.sh / run_e39_sidecar3.sh); T=256 sessions S1-S5;
#   T=128 sessions S1-S4; E40b full-model overflow recovery; E39 at T=256 and T=128 with direct gather (one session each,
#   e39_dg_programs.py); the remaining T=512 tile candidates on the full model (S6-S8).
# Experiment 1 sessions: common setting (every program at the default tile size) and tuned setting (each program at its
# best tile size in the two-layer sweep E38a); every session holds an anchor (A_def at T=128 and T=256, A_512 at T=512).
# Before each step the queue waits while full_model/e38/queue.pause exists, stops if queue.stop exists, and stops if the
# free disk would fall below 100 GiB after the step's compiles.
M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; OUT=$F/e38
cd $M || exit 1
done_in() { grep -q "$2" "$1" 2>/dev/null; }
gate() { local need=$1 what=$2
  while [ -f $OUT/queue.pause ]; do sleep 60; done
  [ -f $OUT/queue.stop ] && { echo "$(date +%F_%T) stopped before $what (queue.stop)"; exit 0; }
  local free=$(df -BG --output=avail $F | tail -1 | tr -dc 0-9)
  [ $free -lt $need ] && { echo "$(date +%F_%T) stopped before $what: $free GiB free, $need GiB needed"; exit 1; }
  echo "$(date +%F_%T) $what ($free GiB free)"; }
run() { local T=$1 S=$2 keep=$3; shift 3; local need=100 pr; done_in $OUT/e38b_T${T}_$S.txt E38B_${S}_DONE && return
  for pr in "$@"; do [ -f $F/qpc_e38_${T}_${pr/:/_}/programqpc.bin ] && continue; case $pr in A:*) need=$((need+25));; *) need=$((need+88));; esac; done
  gate $need "E38 T=$T $S: $* (keep '$keep')"
  E38_KEEP="$keep" bash scripts/run_e38b.sh $T $S "$@" > $OUT/e38b_T${T}_$S.txt 2>&1
  echo "$(date +%F_%T)    $(grep 'session means' $OUT/e38b_T${T}_$S.txt | cut -c1-220)"; }
capture() { local T=$1 q=$2; done_in $F/e41/e41_T$T.txt E41_DONE && return
  gate 100 "E41 routing capture T=$T"; bash scripts/run_e41.sh $T $q > $F/e41/e41_T$T.txt 2>&1; tail -2 $F/e41/e41_T$T.txt | sed 's/^/      /' | cut -c1-200; }
twochunk() { local T=$1; shift; done_in $F/e40/e40c_T$T.txt E40C_DONE && return
  gate 100 "E40c T=$T: $*"; bash scripts/run_e40c.sh $T "$@" > $F/e40/e40c_T$T.txt 2>&1; grep -E "vs|===" $F/e40/e40c_T$T.txt | sed 's/^/      /' | cut -c1-240; }
blocks_dg() { local T=$1; done_in $F/e39/e39_T${T}_P1dg.txt E39_P1dg_DONE && return
  local p=$(/home/chihao/qeff-venv/bin/python scripts/e39_dg_programs.py $M $T); gate 200 "E39 T=$T P1dg: $p"
  bash scripts/run_e39.sh $T P1dg $p > $F/e39/e39_T${T}_P1dg.txt 2>&1
  echo "$(date +%F_%T)    $(grep 'session means' $F/e39/e39_T${T}_P1dg.txt | cut -c1-400)"; }
done_in $F/e40/e40a.txt E40A_DONE || { gate 110 "E40a two-layer overflow mechanism"; bash scripts/run_e40a.sh > $F/e40/e40a.txt 2>&1; }
run 512 S5 "A_def E_def" A:def D:def E:def
capture 512 $F/qpc_e38_512_A_def
twochunk 512 $F/qpc_e38_512_A_def $F/qpc_e38_512_E_def; rm -rf $F/qpc_e38_512_A_def $F/qpc_e38_512_E_def
echo done > $OUT/e38b_T512_common.done
run 256 S1 "A_def A_512 E_1024" A:def A:512 E:1024
capture 256 $F/qpc_e38_256_A_def
twochunk 256 $F/qpc_e38_256_A_512 $F/qpc_e38_256_E_1024; rm -rf $F/qpc_e38_256_A_512 $F/qpc_e38_256_E_1024
run 256 S2 "A_def" A:def B:512 C:512
run 256 S3 "A_def" A:def D:512 E:def
run 256 S4 "A_def" A:def B:def C:def
run 256 S5 ""      A:def D:def
run 128 S1 "A_def" A:def B:1024 C:512
capture 128 $F/qpc_e38_128_A_def
run 128 S2 "A_def E_512" A:def D:512 E:512
twochunk 128 $F/qpc_e38_128_A_def $F/qpc_e38_128_E_512; rm -rf $F/qpc_e38_128_E_512
run 128 S3 "A_def" A:def B:def C:def
run 128 S4 ""      A:def D:def E:def
done_in $F/e40/e40b.txt E40B_DONE || { gate 220 "E40b full-model overflow recovery"; bash scripts/run_e40b.sh > $F/e40/e40b.txt 2>&1; grep -E "===|bit-identical|rejected|schedule|compile" $F/e40/e40b.txt | sed 's/^/      /' | cut -c1-300; }
blocks_dg 256
blocks_dg 128
run 512 S6 "A_512" A:512 A:1024 B:512
run 512 S7 "A_512" A:512 C:512 D:512
run 512 S8 ""      A:512 E:512
echo done > $OUT/e38_queue.done
