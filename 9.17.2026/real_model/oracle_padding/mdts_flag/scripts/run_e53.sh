#!/bin/bash
# E53: tier-count sweep (assessment §5, priority 3): how saved expert rows trade against added stages and per-tier gathers. With
# C (uniform, 1 capacity), D (2 tiers) and E (RankTier, 3 tiers) from E46/E47, this adds the cheapest 4- and 5-tier structures of
# tier_budget.py under the same calibration rule (first tier T, lower tiers 1.28 x the calibrated maximum of their first rank,
# rounded to 16; lanes per SoC from {1, 2, 4, 8, 16}):
#   T=512: t4 4x512,4x240,8x128,16x64 (5056 rows per SoC), t5 4x512,4x240,4x128,4x80,16x64 (4864); E: 6144, D: 9216, C: 16384
#   T=256: t4 4x256,4x160,8x64,16x32 (2688),               t5 4x256,2x160,2x96,8x64,16x32 (2560);  E: 3072, D: 4608, C: 8192
# All rank at run time, gather weights from the bank, use the token-owned combine (one [T*k, d] gather per tier and SoC) and the
# elementwise sums, at E's tile (1024 KiB). Sessions per T: X1 {s5os, t4}, X2 {s5os, t5}, every program twice in mirrored order
# with a capacity check. Then instrumented two-layer profiles at T=512 of C, D, E, t4 and t5 (stage windows, combine time).
# Usage: run_e53.sh [T ...] (default 512 256)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e53; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e53_${T}_$1; }
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
for T in ${@:-512 256}; do
  grep -q "E53_T${T}_DONE" $OUT/e53_run_T$T.txt 2>/dev/null && continue
  case $T in
    256) spec=specializations_T256.json; ref=$F/ref256; p=full256; st=stack256_native_rw_ret_dense; t4=4x256,4x160,8x64,16x32; t5=4x256,2x160,2x96,8x64,16x32;;
    512) spec=specializations_T512.json; ref=$F/ref512; p=full512; st=stack512_native_rw_ret_dense; t4=4x512,4x240,8x128,16x64; t5=4x512,4x240,4x128,4x80,16x64;;
  esac
  declare -A CAP=([s5os]="" [t4]=$t4 [t5]=$t5); declare -A TS=([t4]=$t4 [t5]=$t5)
  {
  echo "================ T=$T; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  for v in t4 t5; do g=full${T}_$v; [ -f $F/${g}_hpos/model.onnx ] && continue
    $PY $M/scripts/build_full_tiered.py $F/$st $F/$g --T $T --tiers ${TS[$v]} --final tileadd > $OUT/build_$g.log 2>&1 || { echo "### build failed $g"; tail -3 $OUT/build_$g.log; exit 1; }
    $PY $M/scripts/headpar_graph.py $g ${g}_hpos --osum addtree > $OUT/build_${g}_hpos.log 2>&1 || { echo "### twin failed $g"; exit 1; }
    if diff -rq ${g}_hpos/weights_hp native_c128_hp/weights_hp > /dev/null; then rm -rf ${g}_hpos/weights_hp; ln -s $F/native_c128_hp/weights_hp ${g}_hpos/weights_hp; fi
    echo "built ${g}_hpos: $(tail -1 $OUT/build_$g.log | cut -c1-150)"; done
  for job in "X1 t4" "X2 t5"; do
    read -r S id <<< "$job"; sdone $S s5os $id && { echo "== $S complete, kept"; continue; }
    [ -f $(qp s5os)/programqpc.bin ] || { gate 125; compile s5os ${p}_naive_tileadd_hpos ""; }
    [ -f $(qp $id)/programqpc.bin ] || gate 160; compile $id full${T}_${id}_hpos "-size-split-granularity=1024"
    echo "== $S {s5os, $id}"; session $S s5os $id; rm -rf $(qp $id)
  done
  rm -rf $(qp s5os)
  echo "$(date +%F_%T) E53_T${T}_DONE"
  } >> $OUT/e53_run_T$T.txt 2>&1
done
if [ ! -f $OUT/e53_T512_profiles.txt ] || ! grep -q E38P_DONE $OUT/e53_T512_profiles.txt; then
  for g in e38C e38D tier3_tileadd t4 t5; do src=full512_${g}_hpos; [ -f $F/trunc2_512_p3$g/model.onnx ] || $PY $M/scripts/truncate_layers.py $F/$src $F/trunc2_512_p3$g 2 $C/custom_io.yaml > /dev/null 2>&1; done
  bash $M/scripts/run_e38p.sh 512 p3e38C:1024 p3e38D:1024 p3tier3_tileadd:1024 p3t4:1024 p3t5:1024 > $OUT/e53_T512_profiles.txt 2>&1
fi
echo E53_DONE
