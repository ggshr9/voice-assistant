#!/bin/bash
cd ~/voice-svc/web
export PATH=$HOME/.local/bin:$PATH
CUBLAS=$(dirname $(find ~/voice-svc/.venv -name libcublas.so* | head -1))
CUDNN=$(dirname $(find ~/voice-svc/.venv -name libcudnn.so* | head -1))
export LD_LIBRARY_PATH=$CUBLAS:$CUDNN
export HF_ENDPOINT=https://huggingface.co
source ~/voice-svc/web/secrets.env
source ~/voice-svc/hf.env
exec ~/voice-svc/.venv/bin/python web_caption.py
