#!/bin/bash
# E47: the rest of the full-model ladder and of the capacity attribution re-timed with the elementwise head-group sum (E45), so
# that every head-parallel program runs the fixed attention. Per chunk length, all programs *_hpos with the KV cache retained:
#   s3sos ladder step 3a: the tree prefix scans only (routing metadata), export's dense combine
#   s3os  step 3: + zero-read removal and the tiled lane sum (the dense-path repairs; E42's R1)
#   s4os  step 4: token-owned combine with the Einsum-tile final sum (E42's R2)     s5os step 5, the anchor as in E46
#   Aos   E38's A (fixed card-major order, capacity T, static weights) at its tuned tile
#   Dos   E38's D (run-time rank, two tiers at T and T/8) at its tuned tile; it also stands for ladder step 6, which used the
#         older build_full_dyncard.py with the same tiers and tile
#   Bos   E38's B (fixed order through the bank gathers) at its tuned tile
# Sessions per T, each holding the anchor s5os, every program timed twice in mirrored order with telemetry and (gathered programs)
# a capacity check: R1 {s5os, s3sos, s3os, s4os, Aos}, R2 {s5os, Dos}, R3 {s5os, Bos}. The session names differ from E46's so that
# e47_summary.py fits both experiments together through the shared anchor. Gathered programs are compiled one at a time and
# deleted after their session (disk gate: 100 GiB stays free). The board is shared: a pass whose load finds a card taken by
# another job waits for the board and is retried. Graphs: the scans-only stack from build_full_stack.py --rewrite-set scan,
# then headpar_graph.py --osum addtree for every source (weights_hp linked to native_c128_hp's after an exact comparison).
# Usage: run_e47.sh [T ...]   (default 512 256 128; a T whose E47_T<T>_DONE marker is in its log is skipped, and on a relaunch a
# session whose passes all have results is kept)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e47; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e47_${T}_$1; }
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
build() { local src=$1 out=${1}_hpos; [ -f $F/$out/model.onnx ] && return
  [ -f $F/$src/model.onnx ] || { echo "### missing source graph $src"; exit 1; }
  $PY $M/scripts/headpar_graph.py $src $out --osum addtree > $OUT/build_$out.log 2>&1 || { echo "### build failed: $out"; tail -3 $OUT/build_$out.log; exit 1; }
  if diff -rq $out/weights_hp native_c128_hp/weights_hp > /dev/null; then rm -rf $out/weights_hp; ln -s $F/native_c128_hp/weights_hp $out/weights_hp; else echo "### $out: weights_hp differ from native_c128_hp, kept"; fi
  echo "built $out: $(tail -1 $OUT/build_$out.log | cut -c1-160)"; }
for T in ${@:-512 256 128}; do
  grep -q "E47_T${T}_DONE" $OUT/e47_run_T$T.txt 2>/dev/null && continue
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; st=stack_native_rw_ret_dense; p=full; pc=full128; ta=""; tb=1024; td=512;;
    256) spec=specializations_T256.json; ref=$F/ref256; st=stack256_native_rw_ret_dense; p=full256; pc=full256; ta=512; tb=512; td=512;;
    512) spec=specializations_T512.json; ref=$F/ref512; st=stack512_native_rw_ret_dense; p=full512; pc=full512; ta=512; tb=1024; td=1024;;
  esac
  declare -A CAP=([s5os]="" [s3sos]="" [s3os]="" [s4os]="" [Aos]="" [Dos]="16x$T,16x$((T / 8))" [Bos]="16x$T,16x$T")
  {
  echo "================ T=$T; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  for src in stack${T}_native_scan_ret_dense $st ${p}_naive ${pc}_e38A ${pc}_e38B ${pc}_e38D; do build $src; done
  if sdone R1 s5os s3sos s3os s4os Aos; then echo "== R1 complete, kept"; [ -f $(qp s5os)/programqpc.bin ] || gate 125; compile s5os ${p}_naive_tileadd_hpos ""
  else gate 225; compile s5os ${p}_naive_tileadd_hpos "" & compile s3sos stack${T}_native_scan_ret_dense_hpos "" & compile s3os ${st}_hpos "" &
    compile s4os ${p}_naive_hpos "" & compile Aos ${pc}_e38A_hpos "${ta:+-size-split-granularity=$ta}" & wait
    echo "== R1 {s5os, s3sos, s3os, s4os, Aos}"; session R1 s5os s3sos s3os s4os Aos; rm -rf $(qp s3sos) $(qp s3os) $(qp s4os) $(qp Aos); fi
  for job in "R2 Dos ${pc}_e38D_hpos $td" "R3 Bos ${pc}_e38B_hpos $tb"; do
    read -r S id graph tile <<< "$job"; sdone $S s5os $id && { echo "== $S complete, kept"; continue; }
    [ -f $(qp $id)/programqpc.bin ] || gate 190; compile $id $graph "-size-split-granularity=$tile"
    echo "== $S {s5os, $id}"; session $S s5os $id; rm -rf $(qp $id)
  done
  rm -rf $(qp s5os)
  echo "$(date +%F_%T) E47_T${T}_DONE"
  } >> $OUT/e47_run_T$T.txt 2>&1
done
$PY $M/scripts/e47_summary.py $F/e46 $OUT > $OUT/e47_summary.txt 2>&1; cat $OUT/e47_summary.txt
echo E47_DONE
