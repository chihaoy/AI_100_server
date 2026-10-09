#!/bin/bash
# all chunk lengths in one queue: checks + timing (8 arms, one session per T), vLLM reference, kernel profile
cd ~/workspace/trt_h100
for T in 512 256 128; do
  PY_TRT=$HOME/workspace/trt_h100/.venv_trt/bin/python PY_VLLM=$HOME/workspace/trt_h100/.venv_vllm/bin/python GPU=0 ./run_all.sh $T > run_all_$T.out 2>&1
  .venv_trt/bin/python profile_arms.py --engines engines/L20 --data data/L20 --T $T --tags S01_T$T,S00_T$T,G00_T$T,G01_T$T,U00_T${T}_3t_32_32_64,U01_T${T}_3t_32_32_64,G10_T${T}_3t_32_32_64,G11_T${T}_3t_32_32_64 --out results/L20_T$T > results/L20_T$T/profile.out 2>&1
done
echo ALL_DONE
