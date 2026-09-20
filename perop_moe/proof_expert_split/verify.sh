#!/bin/bash
# 从 SDK trace 重新生成本目录的证据表(只读,不碰卡)
set -e
source /home/chihao/qeff-venv/bin/activate 2>/dev/null || true
R=/home/chihao/mllm/perop_moe
NT=$(ls $R/decode_realtok/out/*inf-1*-merged*.trace.json | head -1)
FT=$(ls $R/decode_forced/out/*inf-1*-merged*.trace.json  | head -1)
[ -f "$NT" ] || { echo "缺 natural trace:重跑 4card/pass_c_profile.sh"; exit 1; }
[ -f "$FT" ] || { echo "缺 forced  trace:重跑 4card/pass_d_forced.sh + opstats"; exit 1; }
python3 /home/chihao/mllm/research/tools/moe_proof_percard_ops.py \
  --label "NATURAL router (token 9707, router 未改)" \
  --label "FORCED  router (token 9707, router 权重全零 -> 强制只选 expert 0-7)" \
  --csv "$(dirname "$0")/per_card_moe_ops.csv" "$NT" "$FT"
