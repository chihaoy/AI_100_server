#!/bin/bash
S=/tmp/claude-1001/-home-wentao-workspace-AI-100-server/4254e3ca-77da-4b86-8915-e365bd9e745f/scratchpad; PY=/home/chihao/qeff-venv/bin/python; E=$S/e22
comp() { local name=$1 model=$2 stats=${3:-0} out=${4:-$E/qpc/$1}; [ -f $out/programqpc.bin ] && { echo "### $name cached"; return; }; rm -rf $out; local t0=$(date +%s)
  /opt/qti-aic/exec/qaic-compile -aic-hw -aic-hw-version=ai100 -m=$model -convert-to-fp16 -mxfp6-matmul -aic-num-cores=16 -mos=1 -aic-enable-depth-first -mdp-load-partition-config=$S/mdp_ts_4.json -compile-only -aic-binary-dir=$out -stats-level=$stats -mdts-mos=1 > $E/qpc/$name.log 2>&1; echo "### $name rc=$? $(( $(date +%s)-t0 )) s"; }
comp T256_dyncard4t $E/T256_dyncard4t_to/model.onnx & comp T512_dyncard4t $E/T512_dyncard4t_to/model.onnx & comp T512_dyncard4t_s70 $E/T512_dyncard4t_to/model.onnx 70 $E/T512_dyncard4t_to/qpc_s70 & comp T512_anchor_s70 $S/e15/T512_tokenowned/model.onnx 70 $S/e15/T512_tokenowned/qpc_s70b & wait
$PY - <<EOP
import json, os
S="$S"; E="$E"
anc={"T256": (f"{S}/e18/qpc/T256_anchor", f"{S}/e7/T256_hc_158_4", f"{S}/e8/timing_D_runs/T256_hc_158_4__mx_r0/y.bin"), "T512": (f"{S}/e18/qpc/T512_anchor", f"{S}/e7/T512_hc_296_32", f"{S}/e8/timing_D_runs/T512_hc_296_32__mx_r0/y.bin")}
A=[]
for T in ("T256", "T512"):
    q, d, ry = anc[T]; I=f"{S}/e20/inputs{T[1:]}"; p="cat0"
    A += [dict(label=f"{T}_anchor", qpc=q, input=f"{d}/input_f16.bin", counts=f"{d}/expected_counts_i32.bin", ref_y=ry),
          dict(label=f"{T}_dynsplit", qpc=f"{S}/e20/qpc/{T}_dynsplit", input=f"{I}/{T}_{p}_input_f16.bin", counts=f"{I}/{T}_{p}_counts_sorted_i32.bin", ref_y=f"{I}/{T}_{p}_y_ref_f16.bin"),
          dict(label=f"{T}_dyncard4", qpc=f"{E}/qpc/{T}_dyncard4", input=f"{I}/{T}_{p}_input_f16.bin", counts=f"{I}/{T}_{p}_counts_percard_i32.bin", ref_y=f"{I}/{T}_{p}_y_ref_f16.bin"),
          dict(label=f"{T}_dyncard4t", qpc=f"{E}/qpc/{T}_dyncard4t", input=f"{I}/{T}_{p}_input_f16.bin", counts=f"{I}/{T}_{p}_counts_percard_i32.bin", ref_y=f"{I}/{T}_{p}_y_ref_f16.bin")]
json.dump(dict(cases=A), open(f"{E}/spec_f.json","w"), indent=1); print(len(A), "cases")
EOP
$PY $S/e7_timing.py $E/spec_f.json $E/timing_f.json 3 2>&1 | grep --line-buffered -E "pooled|FAILED|Traceback|Error"
for v in T512_anchor:$S/e15/T512_tokenowned:qpc_s70b T512_dyncard4t:$E/T512_dyncard4t_to:qpc_s70; do n=${v%%:*}; r=${v#*:}; d=${r%%:*}; q=${r#*:}; rm -f $d/qpc_s70_use; ln -s $d/$q $d/qpc_s70_use
  if [ $n = T512_anchor ]; then rm -f $d/input_f16.bin $d/expected_counts_i32.bin; ln -s $S/e7/T512_hc_296_32/input_f16.bin $d/input_f16.bin; ln -s $S/e7/T512_hc_296_32/expected_counts_i32.bin $d/expected_counts_i32.bin; else rm -f $d/input_f16.bin $d/expected_counts_i32.bin; ln -s $S/e20/inputs512/T512_cat0_input_f16.bin $d/input_f16.bin; ln -s $S/e20/inputs512/T512_cat0_counts_percard_i32.bin $d/expected_counts_i32.bin; fi
  bash $S/profile_case3.sh $d 512 "$n" | head -1; $PY $S/tail_breakdown.py $d/analysis/sample0 512 "$n" 2>&1 | sed -n 1,7p; $PY $S/e18/e18_hot.py $d/analysis/sample0 "$n engines" | sed -n 2,4p
  $PY - <<EOP
import csv, collections
d="$d/analysis/sample0"; rows=list(csv.DictReader(open(f"{d}/work.csv"))); t0=min(float(r['start']) for r in rows)
def short(n): return n.split('/mlp/')[-1] if '/mlp/' in n else n
fam=collections.defaultdict(lambda: [1e18,-1e18,0.0])
for r in rows:
    n=short(r['node']); b=n.split('_rc')[0]
    if n.startswith('nat/') or n.startswith('nat_'): key='native chain (mask/scan/slots)'
    elif n.startswith('dc_topk') or n.startswith('dc_cnt') or n.startswith('dc_counts') or n.startswith('dc_routed'): key='per-card sort'
    elif n.startswith('dc_gather'): key='bank gathers'
    elif n.startswith('dc_rows') or n=='probe_input_routes': key='row/route permutations'
    elif n.startswith('CumSum') or n in ('Greater','Greater_1','Where_1','Where_5','CtxScatter3DInt','CtxScatter3DInt_1'): key='static routing chain'
    elif b in ('MatMul','MatMul_1','MatMul_2'): key='hot GEMM/DMA'
    elif b in ('MatMul_3','MatMul_4','MatMul_5'): key='cold GEMM/DMA'
    elif b.startswith('CtxGather3D'): key='token gathers'
    elif b.startswith('to_'): key='token-owned combine'
    else: continue
    if int(r['card'])<0: continue
    f=fam[key]; f[0]=min(f[0],float(r['start'])); f[1]=max(f[1],float(r['end'])); f[2]+=float(r['dur'])
print("  family                          first start  last end  busy(ms)")
for k,(s,e,b) in sorted(fam.items(), key=lambda kv: kv[1][0]): print(f"  {k:30s} {(s-t0)/1e3:8.3f} {(e-t0)/1e3:9.3f} {b/1e3:8.2f}")
EOP
done
echo E22F_DONE
