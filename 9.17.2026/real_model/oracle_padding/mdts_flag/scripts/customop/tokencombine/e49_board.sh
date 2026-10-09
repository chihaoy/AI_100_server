#!/bin/bash
# E49 board runs, scheduled into a gap of a longer experiment: waits until the board is idle while another experiment is only
# compiling (no host run in flight), then runs (1) the small four-SoC programs (TokenCombine with data in DDR, cacheable DDR or
# VTCM, and the graph combine) for output and latency, and (2) the two-layer cuts of program D with TokenCombine in the three
# memory placements and D itself, against the two-layer FP32 reference, each compared with D's logits.
# Usage: e49_board.sh <small dir> <qpc dir> <out dir>
SMALL=$1; Q=$2; OUT=$3; mkdir -p $OUT
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model
idle4() { /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$'; }
gap() { for w in $(seq 2160); do if idle4 && ! pgrep -f moe_qwen3_baseline_T.py > /dev/null && ! pgrep -x qaic-runner > /dev/null && pgrep -x qaic-compile > /dev/null; then return 0; fi; sleep 10; done; return 1; }
gap || { echo "no gap within 6 h"; exit 1; }
echo "== gap found $(date +%F_%T)"
for tag in ddr cddr vtcm graph; do
  q=$SMALL/custom/qpc_$tag; [ $tag = ddr ] && q=$SMALL/custom/qpc; [ $tag = graph ] && q=$SMALL/graph/qpc; d=$OUT/small_$tag; rm -rf $d; mkdir -p $d
  idle4 || { echo "   board taken, stop"; exit 1; }
  timeout 300 /opt/qti-aic/exec/qaic-runner -t $q -D 0:1:2:3 --aic-batch-json-input $SMALL/custom/io.json -n 50 --write-output-dir $d > $d/run.log 2>&1
  $PY - "$d" "$SMALL" "$tag" <<'EOP'
import sys, re, numpy as np
d, small, tag = sys.argv[1:4]; log = open(f'{d}/run.log').read(); m = re.search(r'InferenceCnt (\d+) TotalDuration (\d+)us', log)
ref = np.fromfile(f'{small}/custom/reference_out_f32.raw', np.float32).astype(np.float64)
try: y = np.fromfile(f'{d}/out-activation-0-inf-0.bin', np.float16).astype(np.float64); err = np.linalg.norm(y - ref) / np.linalg.norm(ref)
except Exception as ex: err = float('nan')
print(f'   small {tag:5s}: {int(m.group(2)) / int(m.group(1)) / 1000:.3f} ms per inference, rel L2 vs reference {err:.2e}' if m else f'   small {tag}: run failed')
EOP
done
case 512 in 512) ref=$F/ref512;; esac
for tag in D tcc_ddr tcc_cddr tcc_vtcm; do
  q=$Q/qpc_2l_$tag; idle4 || { echo "   board taken, stop"; exit 1; }
  rm -rf $OUT/run2_$tag $OUT/run2_$tag.log
  $PY $M/scripts/moe_qwen3_baseline_T.py run --out $OUT/run2_$tag --precision mxfp6 --qpc $q --reference $ref > $OUT/run2_$tag.stdout 2>&1
  $PY - "$OUT" "$tag" "$ref" <<'EOP'
import sys, json, os, numpy as np
O, n, ref = sys.argv[1:4]
try: r = json.load(open(f'{O}/run2_{n}/result.json'))
except Exception as ex: print(f'   {n}: run failed ({ex})'); sys.exit()
y = np.fromfile(f'{O}/run2_{n}/logits_f32.bin', np.float32).astype(np.float64); fr = np.load(f'{ref}/trunc2_logits_last.npy').astype(np.float64)
s = f"   two-layer {n:9s}: median {r['timing']['median_ms']:.3f} ms; rel L2 vs FP32 {np.linalg.norm(y - fr) / np.linalg.norm(fr):.4f}, argmax {int(y.argmax())} (FP32 {int(fr.argmax())})"
if n != 'D' and os.path.exists(f'{O}/run2_D/logits_f32.bin'):
    x = np.fromfile(f'{O}/run2_D/logits_f32.bin', np.float32).astype(np.float64); s += f'; vs D: bit-identical {np.array_equal(x, y)}, rel L2 {np.linalg.norm(y - x) / np.linalg.norm(x):.2e}'
print(s)
EOP
done
echo "== done $(date +%F_%T)"
