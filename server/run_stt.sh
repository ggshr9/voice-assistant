#!/bin/bash
cd ~/voice-svc
export PATH=$HOME/.local/bin:$PATH
CUBLAS=$(dirname $(find .venv -name "libcublas.so*" 2>/dev/null | head -1))
CUDNN=$(dirname $(find .venv -name "libcudnn.so*" 2>/dev/null | head -1))
export LD_LIBRARY_PATH=$CUBLAS:$CUDNN
export HF_ENDPOINT=https://huggingface.co
exec .venv/bin/python stt_server_cuda.py
