#!/bin/bash
# E50: combine x capacity, the controlled 2x2 of the follow-up plan (RANKTIER_REDUCTION_FOLLOWUP_PLAN.md §6) with the repaired dense
# combine as the stronger baseline. All four cells rank experts at run time and gather their weights from the bank, carry the
# elementwise head-group sum and the elementwise final sum, and differ only in the per-card combine and in the capacities:
#                         repaired dense combine (densefix)        token-owned combine
#   uniform capacity T    cdf: tiers 16xT,16xT                       Cos (E46)
#   RankTier tiers        edf: tiers 8xT,8xC2,16xC3                  Eos (E46)
# densefix (build_full_tiered.py --combine densefix): one zeroed [4E, T, 2048] accumulator per group of equal-width tiers, no read
# of the zeros, read-modify-write for a group's later tiers, the lane sum in 16 token tiles. Each program at the tile of its
# token-owned counterpart (C: 512/512/1024 KiB, E: 512/1024/1024 at T = 128/256/512). Sessions per T, each with the anchor s5os
# shared with E46-E48: V1 {s5os, cdf}, V2 {s5os, edf}, every program twice in mirrored order with telemetry and a capacity check.
# Usage: run_e50.sh [T ...]   (default 512 256 128; a T whose E50_T<T>_DONE marker is in its log is skipped)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e50; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e50_${T}_$1; }
compile() { local id=$1 graph=$2 extra=$3; local out=$(qp $id); [ -f $out/programqpc.bin ] && return; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### ${T}_$id compile rc=$? $(( ($(date +%s)-t0)/60 )) min, $(du -sh $out 2>/dev/null | cut -f1), sha256 $(sha256sum $out/programqpc.bin 2>/dev/null | cut -c1-16)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && { [ $w -gt 1 ] && echo "   (waited $(( (w - 1) * 10 )) s for the board)"; return; }; sleep 10; done; echo '   (board still busy after 1 h)'; }
telemetry() { while true; do echo "@ $(date +%s.%N)"; /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -E "NSP Frequency|SOC power|Board power"; sleep 2; done > $1; }
hostrun() { local n=$1 id=$2 caps=$3 try tp; local q=$(qp $id); [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }
  for try in 1 2 3 4; do   # the board is shared: another job can take a card between the idle check and the load
    rm -rf $OUT/run_$n $OUT/run_$n.log; idle   # the run tool creates <out>.log exclusively
    telemetry $OUT/telemetry_$n.txt & tp=$!
    $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run_$n --precision mxfp6 --qpc $q --reference $ref --routing-counts > $OUT/run_$n.stdout 2>&1   # every program outputs routing_counts; the tool insists
    kill $tp 2>/dev/null; wait $tp 2>/dev/null
    if [ ! -f $OUT/run_$n/result.json ] && grep -q 'already in use' $OUT/run_$n.log 2>/dev/null; then echo "   $n: a card was taken by another job (try $try), retrying"; sleep 60; else break; fi
  done
  $PY - "$OUT/run_$n" "$caps" "$OUT/telemetry_$n.txt" "$n" <<'EOP'
import sys, json, re, numpy as np
d, caps, tel, n = sys.argv[1:5]
try: r = json.load(open(f'{d}/result.json'))
except Exception as ex: print(f'   {n}: run failed ({ex})'); sys.exit()
t, l = r['timing'], r['logits']; s = f"   {n}: median {t['median_ms']:.2f} ms, rounds {[round(v, 2) for v in t['round_medians_ms'].values()]}; rel L2 vs FP32 {l['relative_l2']:.4f}, next '{l['next_token']}' (FP32 '{l['reference_next_token']}')"
if caps:
    c = np.array(r['routing_counts']); off = 0; out = []
    for part in caps.split(','):
        L, cap = (int(v) for v in part.split('x')); blk = c[:, off:off + 4 * L]; out.append(f'{L}x{cap} max {blk.max()} over {int((blk > cap).sum())}'); off += 4 * L
    s += '; capacities ' + ', '.join(out)
fr = [[float(x) for x in re.findall(r'NSP Frequency\(Mhz\):([\d.]+)', b)] for b in open(tel).read().split('@ ')[1:]]
fr = np.array([f for f in fr if len(f) == 4 and min(f) > 600])
if len(fr): s += f"; MHz per SoC min {[int(v) for v in fr.min(0)]}"
print(s)
EOP
}
session() { local S=$1; shift; local names=("$@") i; for i in "${names[@]}"; do hostrun T${T}_${S}_$i $i "${CAP[$i]}"; done
  for (( i=${#names[@]}-1; i>=0; i-- )); do hostrun T${T}_${S}_${names[$i]}_again ${names[$i]} "${CAP[${names[$i]}]}"; done; }
sdone() { local S=$1 i; shift; for i in "$@"; do [ -f $OUT/run_T${T}_${S}_$i/result.json ] && [ -f $OUT/run_T${T}_${S}_${i}_again/result.json ] || return 1; done; }
for T in ${@:-512 256 128}; do
  grep -q "E50_T${T}_DONE" $OUT/e50_run_T$T.txt 2>/dev/null && continue
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; p=full; tc=512; te=512; caps_e="8x128,8x48,16x16";;
    256) spec=specializations_T256.json; ref=$F/ref256; p=full256; tc=512; te=1024; caps_e="8x256,8x64,16x32";;
    512) spec=specializations_T512.json; ref=$F/ref512; p=full512; tc=1024; te=1024; caps_e="8x512,8x128,16x64";;
  esac
  pc=full$T
  declare -A CAP=([s5os]="" [cdf]="16x$T,16x$T" [edf]=$caps_e)
  {
  echo "================ T=$T; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  for g in ${pc}_cdf_hpos ${pc}_edf_hpos ${p}_naive_tileadd_hpos; do [ -f $F/$g/model.onnx ] || { echo "### missing graph $g"; exit 1; }; done
  for job in "V1 cdf ${pc}_cdf_hpos $tc" "V2 edf ${pc}_edf_hpos $te"; do
    read -r S id graph tile <<< "$job"; sdone $S s5os $id && { echo "== $S complete, kept"; continue; }
    [ -f $(qp s5os)/programqpc.bin ] || { gate 125; compile s5os ${p}_naive_tileadd_hpos ""; }
    [ -f $(qp $id)/programqpc.bin ] || gate 160; compile $id $graph "-size-split-granularity=$tile"
    echo "== $S {s5os, $id}"; session $S s5os $id; rm -rf $(qp $id)
  done
  rm -rf $(qp s5os)
  echo "$(date +%F_%T) E50_T${T}_DONE"
  } >> $OUT/e50_run_T$T.txt 2>&1
done
$PY $M/scripts/e47_summary.py $F/e46 $F/e47 $F/e48 $OUT > $OUT/e50_summary.txt 2>&1; cat $OUT/e50_summary.txt
echo E50_DONE
