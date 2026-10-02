#!/bin/bash
# E40c: two consecutive prefill chunks through one retained KV cache on full-model programs (experiment 4a), no fallback.
# Usage: run_e40c.sh <T> <program dir> [<program dir> ...]
# Chunk 0 = tokens 0..T-1 and chunk 1 = tokens T..2T-1 of a 2T-token held-out prompt with an FP32 reference for every
# position (T=128: ref256, T=256: ref512, T=512: ref1024); the programs have ctx_len 2T. Each program runs the pair 5 times
# (accepted logits must repeat exactly); lane counts are checked against the program's own capacities (overflow is
# detected and reported, no fallback). Reports per-chunk latency, accuracy of both chunks against FP32, and bit-identity
# between the programs (the first program is the full-capacity reference).
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model; OUT=$F/e40; mkdir -p $OUT
T=$1; shift; cd $F || exit 1
case $T in 128) ref=$F/ref256; tiers="8x128,8x48,16x16";; 256) ref=$F/ref512; tiers="8x256,8x64,16x32";; 512) ref=$F/ref1024; tiers="8x512,8x128,16x64";; esac
idle() { local w; for w in $(seq 360); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && return; sleep 10; done; }
[ -x $OUT/multichunk_host ] || g++ -O2 -std=c++17 -Wall -Wextra -Werror -I/opt/qti-aic/dev/inc $M/scripts/moe_qwen3_multichunk_host.cpp -o $OUT/multichunk_host -L/opt/qti-aic/dev/lib/x86_64 -lQAic -Wl,-rpath,/opt/qti-aic/dev/lib/x86_64
names=()
for q in "$@"; do
  n=T${T}_$(basename $q | sed "s/^qpc_e38_${T}_//"); names+=($n)
  if [[ $n == *_D_* ]]; then plan="--tiers 16x$T,16x$((T/8)) --profile own:"; elif [[ $n == *_E_* ]]; then plan="--tiers $tiers --profile own:"; else plan="--tiers 16x$T,16x$T --profile own:"; fi
  $PY $M/scripts/e40_plan.py $OUT/plan_c_$n.txt $plan --no-inputs --primary own
  rm -rf $OUT/c_$n; idle
  $OUT/multichunk_host $q $ref/input_ids_i64.bin $OUT/c_$n $T 5 $OUT/plan_c_$n.txt > $OUT/c_$n.log 2>&1; echo "=== $n: $(tail -1 $OUT/c_$n.log)"
done
$PY - "$OUT" "$T" "$ref" "${names[@]}" <<'EOP'
import sys, csv, numpy as np
O, T, ref = sys.argv[1], int(sys.argv[2]), sys.argv[3]; names = sys.argv[4:]
R = np.load(f'{ref}/logits.npy', mmap_mode='r')
L = lambda n, c: np.fromfile(f'{O}/c_{n}/logits_chunk{c}.bin', np.float32) if __import__('os').path.exists(f'{O}/c_{n}/logits_chunk{c}.bin') else None
for n in names:
    rows = list(csv.DictReader(open(f'{O}/c_{n}/attempts.csv'))) if __import__('os').path.exists(f'{O}/c_{n}/attempts.csv') else []
    if not rows: print(f'{n}: no attempts'); continue
    parts = []
    for c in (0, 1):
        ms = [float(r['host_ms']) for r in rows if int(r['chunk']) == c]; over = max(int(r['overflow_lanes']) for r in rows if int(r['chunk']) == c)
        y = L(n, c).astype(np.float64); r = np.asarray(R[(c + 1) * T - 1], np.float64)
        top = lambda v: set(np.argsort(-v)[:5].tolist())
        parts.append(f'chunk {c}: median {np.median(ms):.2f} ms, overflow lanes {over}, rel L2 vs FP32 {np.linalg.norm(y - r) / np.linalg.norm(r):.4f}, argmax {int(y.argmax())} (FP32 {int(r.argmax())}), top-5 overlap {len(top(y) & top(r))}/5')
    print(f'{n}: ' + '; '.join(parts))
for n in names[1:]:
    print(f'{n} vs {names[0]}: ' + ', '.join(f'chunk {c} bit-identical {np.array_equal(L(n, c), L(names[0], c))}' for c in (0, 1) if L(n, c) is not None and L(names[0], c) is not None))
EOP
echo E40C_DONE
