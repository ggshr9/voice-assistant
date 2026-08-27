# voice-assistant 网页版(实时字幕/上传纪要/手记合并/跨会议问答)
# 单镜像双口味:
#   --build-arg TORCH_FLAVOR=cpu    无卡机器/试用(转写走 faster-whisper int8,慢但全功能)
#   --build-arg TORCH_FLAVOR=cu121  NVIDIA GPU(需 nvidia-container-toolkit)
FROM python:3.12-slim

ARG TORCH_FLAVOR=cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg openssl && rm -rf /var/lib/apt/lists/*

# torch 按口味装(cpu 轮子小得多);其余依赖两种口味相同
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_FLAVOR}
# webrtcvad 是 C 扩展且没有官方预编译轮子:同一层装编译器、编完卸掉,不进镜像
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
 && pip install --no-cache-dir \
      aiohttp webrtcvad numpy faster-whisper \
      "pyannote.audio>=3.1" qwen-asr \
 && apt-get purge -y gcc python3-dev && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 目录结构与裸机一致:web/ 在根下,共享模块(prompts/llm_chain/noise_filter/recall_core)在根
COPY prompts.py llm_chain.py noise_filter.py recall_core.py ./
COPY bin/_voiceprint.py ./_voiceprint.py
COPY web/ ./web/

ENV MEETING_PY_STT=python3 \
    MEETING_PY_DIA=python3 \
    CAPTION_TLS_SELFSIGN=1 \
    CAPTION_TLS_DIR=/data/tls \
    PORT=8443
# 会话与模型缓存都放 /data,volume 挂出去才不会随容器消失
ENV HF_HOME=/data/hf
VOLUME /data
EXPOSE 8443

WORKDIR /app/web
CMD ["python3", "web_caption.py"]
