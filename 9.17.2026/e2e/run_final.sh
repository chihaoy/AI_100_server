#!/bin/bash
export PATH=/opt/qti-aic/exec:$PATH; source /home/chihao/qeff-venv/bin/activate
E=/home/chihao/mllm/9.17.2026/e2e; S=/tmp/claude-1002/-home-chihao-mllm/94739cc2-e8eb-4973-a4dd-5aba4d1aa56e/scratchpad
echo "### verify layers 0-19 (final cold QPCs)"; python3 /home/chihao/mllm/research/tools/moe_e2e_verify_layers.py --layers 0-19 2>&1 | grep -v "^\[" | cut -c1-118
echo "### swap breakdown on 2 layers"; E2E_WARM=0 $S/moe_e2e_host chain $E/chain_test/plan_2.txt $E/chain_test/h_L20_f16.bin $E/chain_test 1 --swap 2>&1 | grep -v "^\[" | cut -c1-400
echo "### full 48-layer chain, swap mode, 1 pass, dump per-layer outputs"; E2E_WARM=0 $S/moe_e2e_host chain $E/plan.txt $E/ref/h_L0_f16.bin $E/chain 1 --swap --dump-layers 2>&1 | grep -v "^\[" | cut -c1-1500
echo "### logits"; python3 /home/chihao/mllm/research/tools/moe_e2e_logits.py --h $E/chain/h_final_f32.bin --name "path-2 chain 48L" 2>&1 | grep -v Warning
echo "### DONE"
