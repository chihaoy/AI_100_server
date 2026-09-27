#!/bin/bash
# E22: per-card dynamic split — correctness on 200 prompts, timing vs static anchor and global dynamic, DRAM per card, profile
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e22; mkdir -p $E/qpc; Q20=$S/e20/qpc
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1
  echo "### $name rc=$? $(( $(date +%s)-t0 )) s qpc=$(du -sh $out 2>/dev/null | cut -f1)"; [ -f $out/programqpc.bin ] || grep -vE '^\s*$' $E/qpc/$name.log | sed -n 3,12p | cut -c1-220; }
comp T128_dyncard $E/T128_dyncard_to/model.onnx & comp T128_dyncard_s70 $E/T128_dyncard_to/model.onnx 70 $E/T128_dyncard_to/qpc_s70 & wait
[ -f $E/qpc/T128_dyncard/programqpc.bin ] || { echo E22_DONE; exit; }
$PY - <<EOP
import json, os, glob
S="$S"; E="$E"; Q20="$Q20"; I=f"{S}/e20/inputs128"
A=[dict(label="T128_static_mx", qpc=f"{Q20}/T128_static_mx", input=f"{S}/e8/T128_hc_92_2/input_f16.bin", counts=f"{S}/e8/T128_hc_92_2/expected_counts_i32.bin", ref_y=f"{S}/e8/timing_D_runs/T128_hc_92_2__mx_r0/y.bin"),
   dict(label="T128_dynsplit128", qpc=f"{Q20}/T128_dynsplit128", input=f"{I}/T128_nat41_input_f16.bin", counts=f"{I}/T128_nat41_counts_sorted_i32.bin", ref_y=f"{I}/T128_nat41_y_ref_f16.bin"),
   dict(label="T128_dyncard", qpc=f"{E}/qpc/T128_dyncard", input=f"{I}/T128_nat41_input_f16.bin", counts=f"{I}/T128_nat41_counts_percard_i32.bin", ref_y=f"{I}/T128_nat41_y_ref_f16.bin")]
json.dump(dict(cases=A), open(f"{E}/spec_a.json","w"), indent=1)
cases=[]
for d in (I, f"{S}/e21/inputs128_mmlu", f"{S}/e21/inputs128_swe", f"{S}/e21/inputs128_humaneval"):
    for f in sorted(glob.glob(f"{d}/T128_*_input_f16.bin")):
        n=os.path.basename(f)[:-len('_input_f16.bin')]
        if os.path.isfile(f"{d}/{n}_y_ref_f16.bin"): cases.append(dict(label=n, qpc=f"{E}/qpc/T128_dyncard", input=f, counts=f"{d}/{n}_counts_percard_i32.bin", ref_y=f"{d}/{n}_y_ref_f16.bin"))
json.dump(dict(cases=cases), open(f"{E}/spec_sweep.json","w"), indent=1); print(len(cases), "sweep cases")
EOP
echo "===== timing: static anchor / global dynamic / per-card dynamic ====="
$PY $S/e7_timing.py $E/spec_a.json $E/timing_a.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error|round 0"
echo "===== sweep: 200 prompts, four workloads ====="
$PY $S/e7_timing.py $E/spec_sweep.json $E/sweep.json 1 2>&1 | grep -E "round 0|FAILED|Traceback" | awk '{print $3, $5, $NF, $(NF-6)}' > $E/sweep.txt
$PY - <<EOP
import numpy as np, collections, glob, os
S="$S"; E="$E"; rows=[l.split() for l in open(f"{E}/sweep.txt")]; by=collections.defaultdict(list)
for n, lat, rel, ok in rows:
    w='gsm8k' if ('nat' in n or 'cat' in n) else n.split('_')[1].rstrip('0123456789'); by[w].append((float(lat), float(rel), ok))
for w, v in by.items():
    lat=np.array([a for a,_,_ in v]); print(f"  {w:10s} {len(v):3d} prompts  latency median {np.median(lat):.3f} ms  counts exact {sum(c=='True' for _,_,c in v)}/{len(v)}")
for w, d in (("gsm8k", f"{S}/e20/inputs128"), ("mmlu", f"{S}/e21/inputs128_mmlu"), ("swe", f"{S}/e21/inputs128_swe"), ("humaneval", f"{S}/e21/inputs128_humaneval")):
    meds=[]; mx=0
    for f in sorted(glob.glob(f"{d}/T128_*_y_ref_f16.bin")):
        n=os.path.basename(f)[:-len('_y_ref_f16.bin')]; yf=f"{E}/sweep_runs/{n}_r0/y.bin"
        if not os.path.exists(yf): continue
        y=np.fromfile(yf,np.float16).astype(np.float64).reshape(128,2048); r=np.fromfile(f,np.float16).astype(np.float64).reshape(128,2048)
        e=np.linalg.norm(y-r,axis=1)/np.maximum(np.linalg.norm(r,axis=1),1e-6); meds.append(np.median(e)); mx=max(mx,e.max())
    print(f"  {w:10s} per-token rel error median {np.median(meds):.3e}, worst {mx:.3e}")
EOP
echo "===== DRAM per card while running (MiB in use above idle) ====="
base=$(/opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep 'Dram Free' | grep -oE '[0-9]+' | tr '\n' ' ')
for v in T128_static_mx:$Q20:$S/e8/T128_hc_92_2/input_f16.bin T128_dynsplit128:$Q20:$S/e20/inputs128/T128_nat41_input_f16.bin T128_dyncard:$E/qpc:$S/e20/inputs128/T128_nat41_input_f16.bin; do n=${v%%:*}; r=${v#*:}; q=${r%%:*}; inp=${r#*:}; d=$E/mem_$n; rm -rf $d; mkdir -p $d; head -c $((128*2048*2)) /dev/zero > $d/y.bin; head -c 512 /dev/zero > $d/c.bin
  printf '{"IO-files": [[{"path": "%s", "dims": [128, 2176], "elem-size": 2, "io-direction": "in", "map-to": "probe_input"}, {"path": "%s/y.bin", "dims": [128, 2048], "elem-size": 2, "io-direction": "out", "map-to": "y"}, {"path": "%s/c.bin", "dims": [128], "elem-size": 4, "io-direction": "out", "map-to": "counts"}]]}' $inp $d $d > $d/io.json
  (/opt/qti-aic/exec/qaic-runner -t $q/$n -D 0:1:2:3 --aic-batch-json-input $d/io.json -n 3000 -S 1 -T 1 > $d/runner.log 2>&1 &); sleep 7; used=$(/opt/qti-aic/tools/qaic-util -q 2>/dev/null | grep 'Dram Free' | grep -oE '[0-9]+' | tr '\n' ' ')
  echo "$n: $($PY -c "b='$base'.split(); u='$used'.split(); print(' '.join(f'card{i} {(int(b[i])-int(u[i]))//1024} MiB' for i in range(4)))")"; sleep 12; done; sleep 3
echo "===== profile: per-card dynamic (stats70) ====="
d=$E/T128_dyncard_to; rm -f $d/input_f16.bin $d/expected_counts_i32.bin; ln -s $S/e20/inputs128/T128_nat41_input_f16.bin $d/input_f16.bin; ln -s $S/e20/inputs128/T128_nat41_counts_percard_i32.bin $d/expected_counts_i32.bin
bash $S/profile_case2.sh $d 128 "T128 dyncard MXFP6" | head -1; $PY $S/tail_breakdown.py $d/analysis/sample0 128 "T128_dyncard" 2>&1 | sed -n 1,7p
$PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); t0=min(float(r['start']) for r in rows)
def short(n): return n.split('/mlp/')[-1] if '/mlp/' in n else n
fam=collections.defaultdict(lambda: [1e18,-1e18,0.0])
for r in rows:
    n=short(r['node']); b=n.split('_rc')[0]
    if n.startswith('nat/') or n.startswith('nat_'): key='native chain (mask/scan/slots)'
    elif n.startswith('dc_topk') or n.startswith('dc_cnt') or n.startswith('dc_counts') or n.startswith('dc_routed'): key='per-card sort'
    elif n.startswith('dc_bank'): key='bank gathers'
    elif n.startswith('dc_rows') or n=='probe_input_routes': key='row/route permutations'
    elif b in ('MatMul','MatMul_1','MatMul_2'): key='hot GEMM/DMA'
    elif b in ('MatMul_3','MatMul_4','MatMul_5'): key='cold GEMM/DMA'
    elif b.startswith('to_'): key='token-owned combine'
    else: continue
    if int(r['card'])<0: continue
    f=fam[key]; f[0]=min(f[0],float(r['start'])); f[1]=max(f[1],float(r['end'])); f[2]+=float(r['dur'])
print("  family                          first start  last end  busy(ms)")
for k,(s,e,b) in sorted(fam.items(), key=lambda kv: kv[1][0]): print(f"  {k:30s} {(s-t0)/1e3:8.3f} {(e-t0)/1e3:9.3f} {b/1e3:8.2f}")
EOP
echo E22_DONE
