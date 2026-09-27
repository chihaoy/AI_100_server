#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; d=$S/e20/ifprobe
rm -rf $d/qpc; t0=$(date +%s); /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$d/model.onnx -convert-to-fp16 -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$d/qpc -mdts-mos=1 > $d/compile.log 2>&1; echo "### ifprobe compile rc=$? $(( $(date +%s)-t0 )) s"; [ -f $d/qpc/programqpc.bin ] || { grep -vE '^\s*$' $d/compile.log | sed -n 3,14p | cut -c1-220; echo E20A_DONE; exit; }
for i in $(seq 90); do /opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep -c 'Nsp Free:16' | grep -q '^4$' && break; sleep 20; done
for sel in 0 1; do rm -rf $d/out$sel; mkdir -p $d/out$sel; head -c $((128*768*2)) /dev/zero > $d/y_ph.bin
  cat > $d/io$sel.json <<EOJ
{"IO-files": [[{"path": "$d/input_sel$sel.bin", "dims": [128, 2049], "elem-size": 2, "io-direction": "in", "map-to": "probe_input"},
 {"path": "$d/y_ph.bin", "dims": [128, 768], "elem-size": 2, "io-direction": "out", "map-to": "y"}]]}
EOJ
  /opt/qti-aic/exec/qaic-runner -t $d/qpc -D 0:1:2:3 --aic-batch-json-input $d/io$sel.json -n 200 -S 1 -T 1 -c --write-output-start-iter 199 --write-output-num-samples 1 --write-output-dir $d/out$sel > $d/runner$sel.log 2>&1
  echo "sel=$sel rc=$?: $(grep -E 'Inf/Sec|TotalDuration' $d/runner$sel.log | tr '\n' ' ' | cut -c1-160)"; grep -iE "error|fail" $d/runner$sel.log | head -2; done
$PY - <<EOP
import numpy as np, glob
d="$d"
for sel in (0,1):
    fs=glob.glob(f"{d}/out{sel}/*y*"); 
    if not fs: print("sel", sel, "no output"); continue
    y=np.fromfile(fs[0], np.float16).astype(np.float32); r0=np.fromfile(f"{d}/ref_sel0.bin", np.float32); r1=np.fromfile(f"{d}/ref_sel1.bin", np.float32)
    print(f"sel={sel}: rel L2 vs else-branch ref {np.linalg.norm(y-r0)/np.linalg.norm(r0):.3e}, vs then-branch ref {np.linalg.norm(y-r1)/np.linalg.norm(r1):.3e}")
EOP
echo E20A_DONE
