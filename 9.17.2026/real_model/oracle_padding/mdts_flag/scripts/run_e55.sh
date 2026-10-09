#!/bin/bash
# E55: the production compilation re-timed against the anchor of E46-E54 (D, ladder step 5), so that the paper's speedups over
# production come from the same program + session fit as every other number. Production = the exported graph (native_c128) with
# the compiler's default partitioning (no -mdts-mos=1), as in E36 step 1; the anchor s5os keeps -mdts-mos=1. Session per T:
# P1 {s5os, prod}, each program twice in mirrored order. Usage: run_e55.sh [T ...] (default 512 256 128)
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e55; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e55_${T}_$1; }
compile() { local id=$1 graph=$2 extra=$3; local out=$(qp $id); [ -f $out/programqpc.bin ] && return; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out $extra > $out.log 2>&1
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
  grep -q "E55_T${T}_DONE" $OUT/e55_run_T$T.txt 2>/dev/null && continue
  case $T in
    128) spec=specializations_flat.json; ref=$F/ref; p=full;;
    256) spec=specializations_T256.json; ref=$F/ref256; p=full256;;
    512) spec=specializations_T512.json; ref=$F/ref512; p=full512;;
  esac
  declare -A CAP=([s5os]="" [prod]="")
  {
  echo "================ T=$T; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
  if sdone P1 s5os prod; then echo "== P1 complete, kept"
  else gate 150; compile s5os ${p}_naive_tileadd_hpos "-mdts-mos=1" & compile prod native_c128 "" & wait
    echo "== P1 {s5os, prod}"; session P1 s5os prod; fi
  rm -rf $(qp s5os) $(qp prod)
  echo "$(date +%F_%T) E55_T${T}_DONE"
  } >> $OUT/e55_run_T$T.txt 2>&1
done
echo E55_DONE
