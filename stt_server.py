# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-whisper", "flask"]
# ///
"""常驻 STT 服务：模型保持在内存，POST 音频文件路径即可秒回转写。
启动：uv run stt_server.py   监听 127.0.0.1:8082/transcribe
"""
import sys
from flask import Flask, request, jsonify
import mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
app = Flask(__name__)

# 预热加载
print("加载 whisper 模型…", flush=True)
mlx_whisper.transcribe("/System/Library/Sounds/Tink.aiff", path_or_hf_repo=MODEL)
print("✅ STT 服务就绪 :8082", flush=True)


@app.post("/transcribe")
def transcribe():
    path = request.json.get("path")
    lang = request.json.get("language", "zh")
    if lang in (None, "auto", ""):
        lang = None                      # None → whisper 自动检测语种
    try:
        r = mlx_whisper.transcribe(
            path, path_or_hf_repo=MODEL, language=lang,
            condition_on_previous_text=False,
        )
        return jsonify({"text": r["text"].strip(),
                        "language": r.get("language", lang or "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8082, threaded=False)
