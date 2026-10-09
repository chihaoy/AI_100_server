#!/bin/bash
# venv for the native reference: vLLM (brings its own PyTorch)
set -e
cd ~/workspace/trt_h100
python3 -m venv .venv_vllm
.venv_vllm/bin/pip install -q --upgrade pip
.venv_vllm/bin/pip install -q vllm
.venv_vllm/bin/python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, torch.version.cuda)"
echo SETUP_VLLM_DONE
