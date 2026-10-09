#!/bin/bash
# E54: the TokenCombine custom op in the full model at T=512, repeated (assessment §4): one session holding the anchor D (s5os) and
# the custom-op program, three rounds of passes in alternating order (s5os, tcc, tcc, s5os, s5os, tcc) after the host tool's warmup,
# plus the direct logits comparison with D. The program is compiled ahead of time into qpc_e54_512_tcc (about 11 hours) and kept
# afterwards. Usage: run_e54.sh
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e54; mkdir -p $OUT
cd $F || exit 1
gate() { local need=$1 free; free=$(df -B1G $F | tail -1 | awk '{print $4}'); [ $free -ge $need ] || { echo "$(date +%F_%T) stop: $free GiB free, $need needed"; exit 1; }; }
qp() { echo $F/qpc_e54_${T}_$1; }
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
T=512; spec=specializations_T512.json; ref=$F/ref512
declare -A CAP=([s5os]="" [tcc]="")
{
echo "================ T=$T; $(date +%F_%T); df $(df -B1G $F | tail -1 | awk '{print $4}') GiB"
while pgrep -f "qpc_e54_${T}_tcc" > /dev/null; do sleep 60; done
[ -f $(qp tcc)/programqpc.bin ] || { echo "### no custom-op program"; exit 1; }
[ -f $(qp s5os)/programqpc.bin ] || { gate 125; compile s5os full512_naive_tileadd_hpos ""; }
echo "== Y1 {s5os, tcc} x3"
for n in s5os tcc tcc s5os s5os tcc; do i=$(( ${i:-0} + 1 )); hostrun T${T}_Y1_${n}_p$i $n "${CAP[$n]}"; done
$PY - "$OUT" <<'EOP'
import sys, glob, json, numpy as np
O = sys.argv[1]; L = {}
for d in sorted(glob.glob(f'{O}/run_T512_Y1_*')):
    try: L.setdefault(d.split('_Y1_')[1].rsplit('_p', 1)[0], []).append(np.fromfile(f'{d}/logits_f32.bin', np.float32).astype(np.float64))
    except OSError: pass
d, c = L['s5os'], L['tcc']
print('   reruns bit-identical: D', all(np.array_equal(d[0], x) for x in d), ', custom op', all(np.array_equal(c[0], x) for x in c))
diff = c[0] - d[0]
print(f'   custom op vs D: rel L2 {np.linalg.norm(diff) / np.linalg.norm(d[0]):.3e}, max abs {np.abs(diff).max():.4f} (max |D| {np.abs(d[0]).max():.2f}), same argmax {int(c[0].argmax()) == int(d[0].argmax())}')
EOP
rm -rf $(qp s5os)
echo "$(date +%F_%T) E54_DONE"
} >> $OUT/e54_run_T$T.txt 2>&1
