#!/bin/bash
# venv for the TensorRT arms: PyTorch (CUDA 12.8 wheels), TensorRT 10 (cu12 wheels), onnx
set -e
cd ~/workspace/trt_h100
python3 -m venv .venv_trt
.venv_trt/bin/pip install -q --upgrade pip
.venv_trt/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128
.venv_trt/bin/pip install -q tensorrt-cu12 onnx numpy
.venv_trt/bin/python -c "import torch, tensorrt, onnx; print(\"torch\", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name()); print(\"tensorrt\", tensorrt.__version__, \"onnx\", onnx.__version__)"
echo SETUP_TRT_DONE
