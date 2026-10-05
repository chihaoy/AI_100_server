#!/bin/bash
# E46: full-model remeasurement with the elementwise head-group sum (E45), the core set per chunk length:
#   s2   ladder step 2 (naive T/T: flag, head-parallel attention, the export's combine, capacity T), ReduceSum head-group sum
#   s2os the same with the elementwise head-group sum          s5os step 5 (token-owned combine, elementwise final sum), fixed
#   Eos  RankTier (step 7) fixed, E38's tuned tile             Cos  E38's program C (run-time rank, capacity T) fixed, tuned tile
#   E    RankTier with the ReduceSum (the E38 program), same tile, for the fix's effect on the full model
# Sessions per T, all holding the anchor s5os, each program timed twice in mirrored order with telemetry and (for the ranked
# programs) a capacity check: S1 {s2, s2os, s5os}, S2 {s5os, Eos}, S3 {s5os, Cos}, and with E46_WITH_E=1 S4 {s5os, E}. Programs are compiled one
# gathered program at a time and deleted after their session (disk gate: 100 GiB stays free). The board is shared: a pass
# whose load finds a card taken by another job waits for the board and is retried. Summary: e46_summary.py.
# Usage: run_e46.sh [T ...]   (default 512 256 128; a T whose E46_T<T>_DONE marker is in the log is skipped, and on a relaunch a
# session whose passes all have results is kept)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e46; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e46_${T}_$1; }
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
  grep -q "E46_T${T}_DONE" $OUT/e46_run_T$T.txt 2>/dev/null && continue
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; p=full; pc=full128; te=512; tc=512; caps_e="8x128,8x48,16x16";;
    256) spec=specializations_T256.json; ref=$F/ref256; p=full256; pc=full256; te=1024; tc=512; caps_e="8x256,8x64,16x32";;
    512) spec=specializations_T512.json; ref=$F/ref512; p=full512; pc=full512; te=1024; tc=1024; caps_e="8x512,8x128,16x64";;
  esac
  declare -A CAP=([s2]="" [s2os]="" [s5os]="" [Eos]=$caps_e [Cos]="16x$T,16x$T" [E]=$caps_e)
  {
  echo "================ T=$T; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  if sdone S1 s2 s2os s5os; then echo "== S1 complete, kept"; [ -f $(qp s5os)/programqpc.bin ] || gate 125; compile s5os ${p}_naive_tileadd_hpos ""
  else gate 175; compile s2 native_c128_hp "" & compile s2os native_c128_hpos "" & compile s5os ${p}_naive_tileadd_hpos "" & wait
    echo "== S1 {s2, s2os, s5os}"; session S1 s2 s2os s5os; rm -rf $(qp s2) $(qp s2os); fi
  jobs=("S2 Eos ${p}_tier3_tileadd_hpos $te" "S3 Cos ${pc}_e38C_hpos $tc"); [ -n "$E46_WITH_E" ] && jobs+=("S4 E ${p}_tier3_tileadd_hp $te")
  for job in "${jobs[@]}"; do
    read -r S id graph tile <<< "$job"; sdone $S s5os $id && { echo "== $S complete, kept"; continue; }
    [ -f $(qp $id)/programqpc.bin ] || gate 190; compile $id $graph "-size-split-granularity=$tile"
    echo "== $S {s5os, $id}"; session $S s5os $id; rm -rf $(qp $id)
  done
  rm -rf $(qp s5os)
  echo "$(date +%F_%T) E46_T${T}_DONE"
  } >> $OUT/e46_run_T$T.txt 2>&1
done
$PY $M/scripts/e46_summary.py $OUT > $OUT/e46_summary.txt 2>&1; cat $OUT/e46_summary.txt
echo E46_DONE
