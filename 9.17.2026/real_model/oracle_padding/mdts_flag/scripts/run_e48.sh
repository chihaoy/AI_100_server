#!/bin/bash
# E48: the direct sparse-combine baselines of the follow-up plan (RANKTIER_REDUCTION_FOLLOWUP_PLAN.md §4) against the token-owned
# combine D. All at capacity T with static weights, native expert order, the elementwise head-group sum, the KV cache retained and
# identical routing metadata (tree scans); only the combine differs:
#   s5os    D: token-owned combine (per-card [4, T*8] gathers, local sums over k, elementwise final sum per token tile), ladder step 5
#   ctopk   C-topk: one gather of each token's k rows from all 64 lanes of all cards (CtxGather3D on [1, 64C, 2048]), sum over k,
#           no per-card partials (build_full_dyncard.py --combine topk)
#   ctopkg  C-topk with ONNX Gather on [64C, 2048] (--combine topk_gather), the second natural spelling
#   tcadd   C-lanes: token-centric, per token tile a gather of all 32 lanes of each token on the 64-lane axis, lane sum, elementwise
#           card sum (build_full_stack.py --combine tokencentric --tc-final addtree)
#   tcsl    C-slices: the same with each card's 16 lanes cut out by explicit slices, the E11 dead end (--combine tokencentric_slices)
#   s3os    the repaired dense combine (reference; also timed in E47)
#   tcc     D with its per-card gather and sum replaced by the TokenCombine custom op (HVX kernel, operands in VTCM; E49,
#           scripts/customop/tokencombine), compiled with -register-custom-op
# Phase 1, per T: two-layer cuts of all six, run against the two-layer FP32 reference, logits compared with D's.
# Phase 2, per T: full model, sessions Q1 {s5os, s3os, ctopk, ctopkg}, Q2 {s5os, tcadd, tcsl} and Q3 {s5os, tcc}, every program twice in mirrored
#   order (s5os is also E46's and E47's anchor, so e47_summary.py fits all three experiments together).
# Phase 3: instrumented two-layer profiles of all six at T=512 (run_e38p.sh), for combine time, DRAM and cross-SoC bytes, core spread.
# Usage: run_e48.sh [T ...]   (default 512 256 128; phases 1-2 of a T whose E48_T<T>_DONE marker is in its log are skipped)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e48; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e48_${T}_$1; }
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
VAR=(D s3os ctopk ctopkg tcadd tcsl tcc); TCC="-register-custom-op=$M/scripts/customop/tokencombine/custom_op_config_e48.yaml"
for T in ${@:-512 256 128}; do
  grep -q "E48_T${T}_DONE" $OUT/e48_run_T$T.txt 2>/dev/null && continue
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; st=stack_native_rw_ret_dense; p=full; pc=full128;;
    256) spec=specializations_T256.json; ref=$F/ref256; st=stack256_native_rw_ret_dense; p=full256; pc=full256;;
    512) spec=specializations_T512.json; ref=$F/ref512; st=stack512_native_rw_ret_dense; p=full512; pc=full512;;
  esac
  declare -A G=([D]=${p}_naive_tileadd_hpos [s5os]=${p}_naive_tileadd_hpos [s3os]=${st}_hpos [ctopk]=${pc}_ctopk_hpos [ctopkg]=${pc}_ctopkg_hpos
                [tcadd]=stack${T}_native_rw_ret_tcadd_hpos [tcsl]=stack${T}_native_rw_ret_tcslices_hpos [tcc]=${pc}_tcc_hpos)
  declare -A CAP=([s5os]="" [s3os]="" [ctopk]="" [ctopkg]="" [tcadd]="" [tcsl]="" [tcc]="")
  {
  echo "================ T=$T phase 1, two-layer cuts; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  for v in "${VAR[@]}"; do [ -f $F/${G[$v]}/model.onnx ] || { echo "### missing graph ${G[$v]}"; exit 1; }; trunc ${G[$v]} trunc2_${T}_p2$v; done
  for v in "${VAR[@]}"; do comp2 e48_${T}_$v trunc2_${T}_p2$v "$([ $v = tcc ] && echo $TCC)" & done; wait
  for v in "${VAR[@]}"; do run2 e48_${T}_$v; done
  rm -rf $K/qpc_e48_${T}_*
  echo "================ T=$T phase 2, full model; $(date +%F_%T)"
  if sdone Q1 s5os s3os ctopk ctopkg; then echo "== Q1 complete, kept"
  else gate 200; for v in s5os s3os ctopk ctopkg; do compile $v ${G[$v]} "" & done; wait
    echo "== Q1 {s5os, s3os, ctopk, ctopkg}"; session Q1 s5os s3os ctopk ctopkg; rm -rf $(qp s3os) $(qp ctopk) $(qp ctopkg); fi
  if sdone Q2 s5os tcadd tcsl; then echo "== Q2 complete, kept"
  else [ -f $(qp s5os)/programqpc.bin ] || { gate 175; compile s5os ${G[s5os]} ""; }; gate 150; for v in tcadd tcsl; do compile $v ${G[$v]} "" & done; wait
    echo "== Q2 {s5os, tcadd, tcsl}"; session Q2 s5os tcadd tcsl; rm -rf $(qp tcadd) $(qp tcsl); fi
  if sdone Q3 s5os tcc; then echo "== Q3 complete, kept"
  else [ -f $(qp s5os)/programqpc.bin ] || { gate 175; compile s5os ${G[s5os]} ""; }; gate 150; compile tcc ${G[tcc]} "$TCC"
    echo "== Q3 {s5os, tcc}"; session Q3 s5os tcc; rm -rf $(qp tcc); fi
  rm -rf $(qp s5os)
  echo "$(date +%F_%T) E48_T${T}_DONE"
  } >> $OUT/e48_run_T$T.txt 2>&1
done
if [ ! -f $OUT/e48_T512_profiles.txt ] || ! grep -q E38P_DONE $OUT/e48_T512_profiles.txt; then
  echo "================ phase 3, instrumented two-layer profiles at T=512; $(date +%F_%T)" >> $OUT/e48_run_T512.txt
  bash $M/scripts/run_e38p.sh 512 p2D:def p2s3os:def p2ctopk:def p2ctopkg:def p2tcadd:def p2tcsl:def > $OUT/e48_T512_profiles.txt 2>&1
fi
$PY $M/scripts/e47_summary.py $F/e46 $F/e47 $OUT > $OUT/e48_summary.txt 2>&1; cat $OUT/e48_summary.txt
echo E48_DONE
