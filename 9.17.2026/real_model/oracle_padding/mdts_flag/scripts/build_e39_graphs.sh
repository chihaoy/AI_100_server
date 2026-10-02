#!/bin/bash
# E39 graphs: fixed-size block scheduling for every block size and budget kind at one chunk length, each as a full 48-layer
# graph (build_full_tiered.py --blocks, from the same retained-state stack as RankTier), head-parallel attention, and the
# two-layer cut used for timing. Budgets from e39_block_budget.py. Usage: build_e39_graphs.sh <T> [<B> ...]
PY=/home/chihao/qeff-venv/bin/python; M=/home/wentao/workspace/AI_100_server/9.17.2026/real_model/oracle_padding/mdts_flag; F=$M/full_model
T=$1; shift; Bs=${@:-16 32 64 128}; cd $F || exit 1
DG=${E39_DG:+dg}; DGARG=${E39_DG:+--direct-gather}   # E39_DG=1: the direct-gather variants (suffix dg)
case $T in 128) st=stack_native_rw_ret_dense;; *) st=stack${T}_native_rw_ret_dense;; esac
for B in $Bs; do for kind in "" cal orc; do
  o=full${T}_blk$B$kind$DG; bud=""
  case $kind in cal) bud="--block-budget e39/budget_T${T}_B${B}_calibrated.npy";; orc) bud="--block-budget e39/budget_T${T}_B${B}_oracle.npy";; esac
  $PY $M/scripts/build_full_tiered.py $st $o --T $T --tiers 32x$B --final tileadd --blocks $B $bud $DGARG > $o.build.log 2>&1 || { echo "build failed $o"; tail -3 $o.build.log; continue; }
  rm -rf ${o}_hp; $PY $M/scripts/headpar_graph.py $o ${o}_hp > /dev/null 2>&1 || { echo "headpar failed $o"; continue; }
  rm -rf ${o}_hp/weights_hp; ln -s ../native_c128_hp/weights_hp ${o}_hp/weights_hp; ln -sf $(realpath configs/custom_io.yaml) ${o}_hp/custom_io.yaml
  rm -rf trunc2_${T}_blk$B$kind$DG; $PY $M/scripts/truncate_layers.py ${o}_hp trunc2_${T}_blk$B$kind$DG 2 configs/custom_io.yaml > /dev/null 2>&1 || echo "truncate failed $o"
  echo "$o: $(tail -1 $o.build.log | sed 's/;.*//' | cut -d: -f2-)"
done; done
