#!/bin/bash
# E21b: one dynamic-split QPC (T=128, capacities 128/16) over 200 prompts of four workloads
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e20
cd /home/wentao/workspace/AI_100_server/tools
for w in mmlu swe humaneval; do $PY $E/make_inputs_dyn.py $S/e21/routing_$w $S/e21/inputs128_$w 128 > $S/e21/inputs128_$w.log 2>&1 & done
out=$E/qpc/T128_dynsplit128; rm -rf $out; t0=$(date +%s)
/opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$E/T128_dynsplit128_to/model.onnx -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=0 -mdts-mos=1 > $E/qpc/T128_dynsplit128.log 2>&1; echo "### T128_dynsplit128 rc=$? $(( $(date +%s)-t0 )) s"
wait
$PY - <<EOP
import json, os, glob
S="$S"; E="$E"; cases=[]
for w, d in (("gsm8k", f"{E}/inputs128"), ("mmlu", f"{S}/e21/inputs128_mmlu"), ("swe", f"{S}/e21/inputs128_swe"), ("humaneval", f"{S}/e21/inputs128_humaneval")):
    names=sorted(os.path.basename(f)[:-len('_input_f16.bin')] for f in glob.glob(f"{d}/T128_*_input_f16.bin"))
    cases += [dict(label=n, qpc=f"{E}/qpc/T128_dynsplit128", input=f"{d}/{n}_input_f16.bin", counts=f"{d}/{n}_counts_sorted_i32.bin", ref_y=f"{d}/{n}_y_ref_f16.bin") for n in names if os.path.isfile(f"{d}/{n}_y_ref_f16.bin")]
json.dump(dict(cases=cases), open(f"{S}/e21/spec_all.json","w"), indent=1); print(len(cases), "cases")
EOP
$PY $S/e7_timing.py $S/e21/spec_all.json $S/e21/sweep_all.json 1 2>&1 | grep -E "round 0|FAILED|Traceback" | awk '{print $3, $5, $NF, $(NF-6)}' > $S/e21/sweep_all.txt
$PY - <<EOP
import numpy as np, collections, os, glob
S="$S"; rows=[l.split() for l in open(f"{S}/e21/sweep_all.txt")]
by=collections.defaultdict(list)
for n, lat, rel, ok in rows:
    w = 'gsm8k' if ('nat' in n or 'cat' in n) else n.split('_')[1].rstrip('0123456789'); by[w].append((float(lat), float(rel), ok))
for w, v in by.items():
    lat=np.array([a for a,_,_ in v]); rel=np.array([b for _,b,_ in v]); print(f"{w:10s} {len(v):3d} prompts  latency median {np.median(lat):.3f} ms (min {lat.min():.3f} max {lat.max():.3f})  rel L2 vs FP32 median {np.median(rel):.2e} max {rel.max():.2e}  counts exact: {sum(c=='True' for _,_,c in v)}/{len(v)}")
# per-token error per workload
for w, d in (("gsm8k", f"{S}/e20/inputs128"), ("mmlu", f"{S}/e21/inputs128_mmlu"), ("swe", f"{S}/e21/inputs128_swe"), ("humaneval", f"{S}/e21/inputs128_humaneval")):
    meds=[]; mx=0
    for f in sorted(glob.glob(f"{d}/T128_*_y_ref_f16.bin")):
        n=os.path.basename(f)[:-len('_y_ref_f16.bin')]; yf=f"{S}/e21/sweep_all_runs/{n}_r0/y.bin"
        if not os.path.exists(yf): continue
        y=np.fromfile(yf,np.float16).astype(np.float64).reshape(128,2048); r=np.fromfile(f,np.float16).astype(np.float64).reshape(128,2048)
        e=np.linalg.norm(y-r,axis=1)/np.maximum(np.linalg.norm(r,axis=1),1e-6); meds.append(np.median(e)); mx=max(mx,e.max())
    print(f"  {w:10s} per-token rel error: median {np.median(meds):.3e}, worst token {mx:.3e}")
EOP
echo E21B_DONE
