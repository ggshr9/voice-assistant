#!/bin/bash
# 统一对比测试：同一参考音 + 同一句话，跑不同克隆模型，量化对比
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com

REF=~/会议录音/降噪_我的声音2.wav
REFTEXT="大家好我是这台电脑的主人今天天气晴朗阳光明媚我正在测试声音克隆技术希望这段录音能准确地复刻出我的音色和语调"
TARGET="${1:-你好呀，今天过得怎么样？要不要跟我聊聊最近发生的事情？}"
OUT=~/Desktop/克隆对比
mkdir -p "$OUT"

dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null; }
chars=$(echo -n "$TARGET" | python3 -c "import sys;print(len(sys.stdin.read()))")
echo "目标文本($chars字): $TARGET"
echo "参考音: $REF"
echo "======================================"

# IndexTTS
if [ -f ~/models/IndexTTS-1.5/model.safetensors ]; then
  echo "▶ IndexTTS-1.5 …"
  t0=$(python3 -c "import time;print(time.time())")
  mlx_audio.tts.generate --model ~/models/IndexTTS-1.5 --ref_audio "$REF" --ref_text "$REFTEXT" \
    --text "$TARGET" --file_prefix idx --output_path "$OUT" >/dev/null 2>&1
  t1=$(python3 -c "import time;print(time.time())")
  d=$(dur "$OUT/idx_000.wav")
  echo "  耗时 $(python3 -c "print(round($t1-$t0,1))")s | 音频时长 ${d}s | $([ "$(python3 -c "print(1 if float('${d:-0}')>$chars*0.8+5 else 0)")" = "1" ] && echo '⚠️可能退化' || echo '✅正常')"
fi

# Qwen3-TTS
if [ -f ~/models/Qwen3-TTS-1.7B/model.safetensors ]; then
  echo "▶ Qwen3-TTS-1.7B …"
  t0=$(python3 -c "import time;print(time.time())")
  mlx_audio.tts.generate --model ~/models/Qwen3-TTS-1.7B --ref_audio "$REF" --ref_text "$REFTEXT" \
    --text "$TARGET" --file_prefix qwen --output_path "$OUT" >/dev/null 2>&1
  t1=$(python3 -c "import time;print(time.time())")
  d=$(dur "$OUT/qwen_000.wav")
  echo "  耗时 $(python3 -c "print(round($t1-$t0,1))")s | 音频时长 ${d}s | $([ "$(python3 -c "print(1 if float('${d:-0}')>$chars*0.8+5 else 0)")" = "1" ] && echo '⚠️可能退化' || echo '✅正常')"
fi

echo "======================================"
echo "结果在: $OUT/  （idx_000.wav=IndexTTS, qwen_000.wav=Qwen3-TTS）"
ls "$OUT"/*.wav 2>/dev/null
