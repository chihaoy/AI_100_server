#!/bin/bash
# E52: the "local top-k" formulations of the assessment (RANKTIER_E47_E51_ASSESSMENT_AND_NEXT_STEPS.md §3): each SoC gathers only
# its own selected expert rows, sums them per token into a local partial, and the partials are summed across SoCs. D (ladder step
# 5) is exactly this computation; the two variants keep D's arithmetic, index values, final sum and everything else, and change
# only how each SoC's locality is expressed in the graph (build_full_dyncard.py --combine):
#   s5os  D: a layout-preserving reshape of the 64-lane outputs to [4, 16C, 2048] makes the SoC a leading batch axis of one gather
#   lg    local_global: the same per-SoC index rows, but as global rows (lane*C + slot) into the unsplit [1, 64C, 2048] outputs,
#         so locality exists only in the index values
#   ls    local_slices: each SoC's 16 lanes cut out by an explicit Slice, then its own k-row gather and sum
# All at capacity T with static weights and the elementwise head-group sum. Phase 1: two-layer cuts against D. Phase 2: full model,
# session W1 {s5os, lg, ls}, every program twice in mirrored order (s5os is the anchor of E46-E51). Phase 3: instrumented two-layer
# profiles of lg and ls at T=512 (D's is E48's p2D). Usage: run_e52.sh [T ...] (default 512 256 128)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e52; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e52_${T}_$1; }
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
K=$F/kvtest
trunc() { [ -f $F/$2/model.onnx ] && [ $F/$2/model.onnx -nt $F/$1/model.onnx ] && return; rm -rf $F/$2; $PY $M/scripts/truncate_layers.py $F/$1 $F/$2 2 $C/custom_io.yaml > /dev/null 2>&1 || echo "### truncate failed: $2"; }
comp2() { local name=$1 graph=$2 extra=$3; local out=$K/qpc_$name; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$F/$graph/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $K/compile_$name.log 2>&1
  echo "### $name compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $K/compile_$name.log | sed -n 3,12p | cut -c1-220; }
run2() { local n=$1 q=$K/qpc_$1 try; [ -f $q/programqpc.bin ] || { echo "   $n: no program"; return; }
  for try in 1 2 3 4; do rm -rf $OUT/run2_$n $OUT/run2_$n.log; idle
    $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run2_$n --precision mxfp6 --qpc $q --reference $ref > $OUT/run2_$n.stdout 2>&1
    if [ ! -f $OUT/run2_$n/result.json ] && grep -q 'already in use' $OUT/run2_$n.log 2>/dev/null; then echo "   $n: a card was taken by another job (try $try), retrying"; sleep 60; else break; fi
  done
  $PY - "$OUT" "$n" "$ref" "${n%_*}_D" <<'EOP'
import sys, json, os, numpy as np
O, n, ref, d = sys.argv[1:5]
try: r = json.load(open(f'{O}/run2_{n}/result.json'))
except Exception as ex: print(f'   {n}: run failed ({ex})'); sys.exit()
y = np.fromfile(f'{O}/run2_{n}/logits_f32.bin', np.float32).astype(np.float64); fr = np.load(f'{ref}/trunc2_logits_last.npy').astype(np.float64)
s = f"   {n}: median {r['timing']['median_ms']:.3f} ms; two-layer rel L2 vs FP32 {np.linalg.norm(y - fr) / np.linalg.norm(fr):.4f}, argmax {int(y.argmax())} (FP32 {int(fr.argmax())})"
if os.path.exists(f'{O}/run2_{d}/logits_f32.bin') and d != n:
    x = np.fromfile(f'{O}/run2_{d}/logits_f32.bin', np.float32).astype(np.float64); s += f'; vs D: bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.2e}'
print(s)
EOP
}
VAR=(D lg ls)
for T in ${@:-512 256 128}; do
  grep -q "E52_T${T}_DONE" $OUT/e52_run_T$T.txt 2>/dev/null && continue
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; p=full; pc=full128;;
    256) spec=specializations_T256.json; ref=$F/ref256; p=full256; pc=full256;;
    512) spec=specializations_T512.json; ref=$F/ref512; p=full512; pc=full512;;
  esac
  declare -A G=([D]=${p}_naive_tileadd_hpos [s5os]=${p}_naive_tileadd_hpos [lg]=${pc}_lg_hpos [ls]=${pc}_ls_hpos)
  declare -A CAP=([s5os]="" [lg]="" [ls]="")
  {
  echo "================ T=$T phase 1, two-layer cuts; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  for v in "${VAR[@]}"; do [ -f $F/${G[$v]}/model.onnx ] || { echo "### missing graph ${G[$v]}"; exit 1; }; trunc ${G[$v]} trunc2_${T}_p2$v; done
  for v in "${VAR[@]}"; do comp2 e52_${T}_$v trunc2_${T}_p2$v "" & done; wait
  for v in "${VAR[@]}"; do run2 e52_${T}_$v; done
  rm -rf $K/qpc_e52_${T}_*
  echo "================ T=$T phase 2, full model; $(date +%F_%T)"
  if sdone W1 s5os lg ls; then echo "== W1 complete, kept"
  else gate 175; for v in s5os lg ls; do compile $v ${G[$v]} "" & done; wait
    echo "== W1 {s5os, lg, ls}"; session W1 s5os lg ls; rm -rf $(qp lg) $(qp ls); fi
  rm -rf $(qp s5os)
  echo "$(date +%F_%T) E52_T${T}_DONE"
  } >> $OUT/e52_run_T$T.txt 2>&1
done
if [ ! -f $OUT/e52_T512_profiles.txt ] || ! grep -q E38P_DONE $OUT/e52_T512_profiles.txt; then
  bash $M/scripts/run_e38p.sh 512 p2lg:def p2ls:def > $OUT/e52_T512_profiles.txt 2>&1
fi
echo E52_DONE
