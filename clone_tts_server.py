# /// script
# requires-python = ">=3.10"
# dependencies = ["flask", "soundfile", "numpy"]
# ///
"""常驻 Qwen3-TTS 克隆服务（单线程，避开 MLX 多线程 stream 崩溃）。
模型常驻内存，POST 文本秒回克隆语音。监听 127.0.0.1:8083
"""
import io
import os
import sys

sys.path.insert(0, os.path.expanduser(
    "~/.local/share/uv/tools/mlx-audio/lib/python3.11/site-packages"))

from flask import Flask, request, send_file
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

MODEL = os.path.expanduser("~/models/Qwen3-TTS-1.7B")
REF = os.environ.get("VA_CLONE_REF", os.path.expanduser("~/会议录音/降噪_我的声音2.wav"))
REF_TEXT = os.environ.get("VA_CLONE_TEXT",
    "大家好我是这台电脑的主人今天天气晴朗阳光明媚我正在测试声音克隆技术希望这段录音能准确地复刻出我的音色和语调")

app = Flask(__name__)
print("加载 Qwen3-TTS（含 speech tokenizer）…", flush=True)
model = load_model(MODEL)
print("✅ 克隆 TTS 服务就绪 :8083", flush=True)


def _synth_wav(text, ref=None, ref_text=None):
    results = list(model.generate(
        text=text, ref_audio=ref or REF, ref_text=ref_text or REF_TEXT))
    audio = np.array(results[0].audio)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    buf.seek(0)
    return buf


@app.post("/speak")
def speak():
    text = (request.json.get("text") or "").strip()
    if not text:
        return {"error": "empty"}, 400
    buf = _synth_wav(text, request.json.get("ref_audio"), request.json.get("ref_text"))
    return send_file(buf, mimetype="audio/wav")


@app.post("/v1/audio/speech")
def openai_speech():
    """OpenAI 兼容端点，供 wechat-cc 等调用。固定用女朋友克隆音。"""
    data = request.json or {}
    text = (data.get("input") or data.get("text") or "").strip()
    if not text:
        return {"error": "empty input"}, 400
    buf = _synth_wav(text)
    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8083, threaded=False)
