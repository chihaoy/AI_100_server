#!/bin/bash
# E22e: per-card dynamic split (native chains, global gathers) at T=256/512 vs static anchors and the global dynamic; sweeps
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e22; mkdir -p $E/qpc
for i in $(seq 120); do grep -q "E22C_DONE" $S/run_e22d.out && break; sleep 15; done
comp() { local name=$1 model=$2 out=${3:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=0 -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 3,12p | cut -c1-220; }
comp T256_dyncard4 $E/T256_dyncard4_to/model.onnx & comp T512_dyncard4 $E/T512_dyncard4_to/model.onnx & wait
$PY - <<EOP
import json, os, glob, numpy as np
S="$S"; E="$E"
for T in (512,):
    for w in ("mmlu", "swe", "humaneval"):
        for f in glob.glob(f"{S}/e21/inputs{T}_{w}/T{T}_*_counts_native_i32.bin"):
            c = np.fromfile(f, np.int32); parts = [np.sort(c[32*k:32*k+32])[::-1] for k in range(4)]
            np.concatenate([p[:16] for p in parts] + [p[16:] for p in parts]).astype(np.int32).tofile(f.replace('_counts_native_i32.bin', '_counts_percard_i32.bin'))
anc={"T256": (f"{S}/e18/qpc/T256_anchor", f"{S}/e7/T256_hc_158_4", f"{S}/e8/timing_D_runs/T256_hc_158_4__mx_r0/y.bin"), "T512": (f"{S}/e18/qpc/T512_anchor", f"{S}/e7/T512_hc_296_32", f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin")}
A=[]
for T in ("T256", "T512"):
    q, d, ry = anc[T]; I=f"{S}/e20/inputs{T[1:]}"; p="cat0"
    A += [dict(label=f"{T}_anchor", qpc=q, input=f"{d}/input_f16.bin", counts=f"{d}/expected_counts_i32.bin", ref_y=ry),
          dict(label=f"{T}_dynsplit", qpc=f"{S}/e20/qpc/{T}_dynsplit", input=f"{I}/{T}_{p}_input_f16.bin", counts=f"{I}/{T}_{p}_counts_sorted_i32.bin", ref_y=f"{I}/{T}_{p}_y_ref_f16.bin"),
          dict(label=f"{T}_dyncard4", qpc=f"{E}/qpc/{T}_dyncard4", input=f"{I}/{T}_{p}_input_f16.bin", counts=f"{I}/{T}_{p}_counts_percard_i32.bin", ref_y=f"{I}/{T}_{p}_y_ref_f16.bin")]
A=[c for c in A if os.path.isfile(f"{c['qpc']}/programqpc.bin")]; json.dump(dict(cases=A), open(f"{E}/spec_e.json","w"), indent=1); print(len(A), "timing cases")
for T in (256, 512):
    cases=[]
    for d in [f"{S}/e20/inputs{T}"] + [f"{S}/e21/inputs{T}_{w}" for w in ("mmlu", "swe", "humaneval")]:
        for f in sorted(glob.glob(f"{d}/T{T}_*_input_f16.bin")):
            n=os.path.basename(f)[:-len('_input_f16.bin')]
            if os.path.isfile(f"{d}/{n}_y_ref_f16.bin") and os.path.isfile(f"{d}/{n}_counts_percard_i32.bin"): cases.append(dict(label=n, qpc=f"{E}/qpc/T{T}_dyncard4", input=f, counts=f"{d}/{n}_counts_percard_i32.bin", ref_y=f"{d}/{n}_y_ref_f16.bin"))
    json.dump(dict(cases=cases), open(f"{E}/spec_sweep{T}.json","w"), indent=1); print(T, len(cases), "sweep cases")
EOP
echo "===== timing T=256/512: anchor / global dynamic / per-card dynamic (native chains) ====="
$PY $S/e7_timing.py $E/spec_e.json $E/timing_e.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error"
for T in 256 512; do $PY $S/e7_timing.py $E/spec_sweep$T.json $E/sweepT$T.json 1 2>&1 | grep -E "round 0|FAILED|Traceback" | awk '{print $3, $5, $NF, $(NF-6)}' > $E/sweepT$T.txt; echo "T=$T sweep: $(wc -l < $E/sweepT$T.txt) prompts, counts exact $(grep -c True $E/sweepT$T.txt), latency median $(awk '{print $2}' $E/sweepT$T.txt | sort -g | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')"; done
$PY - <<EOP
import numpy as np, glob, os
S="$S"; E="$E"
for T in (256, 512):
    meds=[]; mx=0; n_=0
    for d in [f"{S}/e20/inputs{T}"] + [f"{S}/e21/inputs{T}_{w}" for w in ("mmlu", "swe", "humaneval")]:
        for f in sorted(glob.glob(f"{d}/T{T}_*_y_ref_f16.bin")):
            n=os.path.basename(f)[:-len('_y_ref_f16.bin')]; yf=f"{E}/sweepT{T}_runs/{n}_r0/y.bin"
            if not os.path.exists(yf): continue
            y=np.fromfile(yf,np.float16).astype(np.float64).reshape(T,2048); r=np.fromfile(f,np.float16).astype(np.float64).reshape(T,2048)
            e=np.linalg.norm(y-r,axis=1)/np.maximum(np.linalg.norm(r,axis=1),1e-6); meds.append(np.median(e)); mx=max(mx,e.max()); n_+=1
    print(f"  T={T}: {n_} prompts, per-token rel error median {np.median(meds):.3e}, worst {mx:.3e}")
EOP
echo E22E_DONE
