#!/bin/bash
# E40b: overflow detection and fallback on the full 48-layer model (RANKTIER_EXPERIMENT_PLAN.md, experiment 4a).
# One RankTier program at T=128 with capacity inputs and three specializations (tiered 8x128+8x48+16x16, full 8x128+8x128+
# 16x128, tight 8x128+8x48+16x4), ctx_len 512, 512 KiB tiles; and the static full-capacity program A (card-major, capacity
# T, default tiles) with ctx_len 512 as the reference implementation at matched precision. Four consecutive 128-token chunks
# of the 512-token held-out prompt through one retained KV cache, each schedule repeated 5 times:
#   R tiered (fallback full)  F chunk 1 forced tight, detected, rerun full  G chunk 0 forced tight  U every chunk full  A static
# Checks: accepted logits of every chunk bit-identical across schedules and to A, accuracy of every chunk against the FP32
# logits at positions 127/255/383/511; reports per-attempt latency (detection, replay), load time and program size.
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; C=$F/configs; OUT=$F/e40; mkdir -p $OUT
cd $F || exit 1; qc=$F/qpc_e40_128_tier3cap; qa=$F/qpc_e40_128_A_ctx512
comp() { local out=$1 graph=$2 spec=$3 extra=$4; [ -f $out/programqpc.bin ] && return; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$F/$graph/model.onnx -retained-state -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$C/mdp_ts_4.json -network-specialization-config=$C/$spec -custom-IO-list-file=$C/custom_io.yaml -compile-only -aic-binary-dir=$out -mdts-mos=1 $extra > $out.log 2>&1
  echo "### $(basename $out) compile rc=$? $(( ($(date +%s)-t0)/60 )) min, $(du -sh $out 2>/dev/null | cut -f1), programqpc.bin $(stat -c %s $out/programqpc.bin 2>/dev/null) bytes"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $out.log | sed -n 3,10p | cut -c1-220; }
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
echo "df before: $(df -h $F | tail -1 | awk '{print $4}')"
comp $qc full128_tier3cap_hp specializations_T128_cap_ctx512.json "-size-split-granularity=512" &
comp $qa full128_e38A_hp specializations_T128_ctx512.json "" &
wait; echo "df after compiles: $(df -h $F | tail -1 | awk '{print $4}')"
[ -x $OUT/multichunk_host ] || g++ -O2 -std=c++17 -Wall -Wextra -Werror -I/opt/qti-aic/dev/inc $M/scripts/moe_qwen3_multichunk_host.cpp -o $OUT/multichunk_host -L/opt/qti-aic/dev/lib/x86_64 -lQAic -Wl,-rpath,/opt/qti-aic/dev/lib/x86_64
P="--tiers 8x128,8x48,16x16 --profile tiered:48,16 --profile full:128,128 --profile tight:48,4"
$PY $M/scripts/e40_plan.py $OUT/plan_b_R.txt $P --primary tiered --fallback full
$PY $M/scripts/e40_plan.py $OUT/plan_b_F.txt $P --primary tiered --fallback full --force 1:tight
$PY $M/scripts/e40_plan.py $OUT/plan_b_G.txt $P --primary tiered --fallback full --force 0:tight
$PY $M/scripts/e40_plan.py $OUT/plan_b_U.txt $P --primary full
$PY $M/scripts/e40_plan.py $OUT/plan_b_A.txt --tiers 16x128,16x128 --profile own: --no-inputs --primary own
for s in A R F G U; do q=$qc; [ $s = A ] && q=$qa; rm -rf $OUT/b_$s; idle
  $OUT/multichunk_host $q $F/ref512/input_ids_i64.bin $OUT/b_$s 128 5 $OUT/plan_b_$s.txt > $OUT/b_$s.log 2>&1; echo "=== schedule $s: $(tail -1 $OUT/b_$s.log); setup $(tr '\n' ' ' < $OUT/b_$s/setup.csv 2>/dev/null)"; done
$PY - "$OUT" "$F/ref512" <<'EOP'
import sys, os, csv, numpy as np, collections
O, ref = sys.argv[1:3]; R = np.load(f'{ref}/logits.npy', mmap_mode='r')
L = lambda s, c: np.fromfile(f'{O}/b_{s}/logits_chunk{c}.bin', np.float32) if os.path.exists(f'{O}/b_{s}/logits_chunk{c}.bin') else None
for s in 'ARFGU':
    if not os.path.exists(f'{O}/b_{s}/attempts.csv'): print(f'schedule {s}: no attempts'); continue
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{O}/b_{s}/attempts.csv')): by[(int(r['chunk']), int(r['attempt']), r['profile'], r['accepted'], r['overflow_lanes'], r['first_overflow_layer'], r['max_excess'])].append(float(r['host_ms']))
    print(f'schedule {s}:')
    for k, v in sorted(by.items()): print(f'   chunk {k[0]} attempt {k[1]} {k[2]}: accepted {k[3]}, overflow lanes {k[4]} (first layer {k[5]}, max excess {k[6]}), median {np.median(v):.2f} ms over {len(v)}')
for c in range(4):
    r = np.asarray(R[128 * (c + 1) - 1], np.float64); y = L('R', c)
    acc = f'rel L2 vs FP32 {np.linalg.norm(y - r) / np.linalg.norm(r):.4f}, argmax {int(y.argmax())} (FP32 {int(r.argmax())})' if y is not None else ''
    print(f'chunk {c}: ' + ', '.join(f'{s} vs R bit-identical {np.array_equal(L(s, c), y)}' for s in 'AFGU' if L(s, c) is not None and y is not None) + f'; R {acc}')
for s, c in (('F', 1), ('G', 0)):
    p = f'{O}/b_{s}/logits_chunk{c}_attempt0.bin'
    if os.path.exists(p): bad = np.fromfile(p, np.float32); ok = L('R', c); print(f'rejected tight attempt, schedule {s} chunk {c}: rel L2 vs accepted {np.linalg.norm(bad - ok) / np.linalg.norm(ok):.3e}')
EOP
[ -z "$E40_KEEP" ] && rm -rf $qc $qa
echo "df after cleanup: $(df -h $F | tail -1 | awk '{print $4}')"
echo E40B_DONE
