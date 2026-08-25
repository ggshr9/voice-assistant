"""CUDA STT (faster-whisper, 4090) — OpenAI-compatible for wechat-cc.

POST /v1/audio/transcriptions  (multipart: file=<audio>, model=<ignored>) -> {"text","language"}
POST /transcribe               (legacy: audio=<audio>)                     -> {"text","language"}
GET  /health                                                              -> {"ok","model"}

faster-whisper decodes via PyAV, so any container (amr/mp3/ogg/wav) is accepted
by content — the field/extension are hints only. Bound to the tailscale IP by
default (STT_HOST), reached only via the VPS nginx /stt/ passthrough.
"""
import os, tempfile
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
HOST = os.environ.get("STT_HOST", "127.0.0.1")   # 部署时用 STT_HOST 指定   # tailscale IP only (not 0.0.0.0)
PORT = int(os.environ.get("STT_PORT", "8090"))
print(f"loading {MODEL} on CUDA ...", flush=True)
m = WhisperModel(MODEL, device="cuda", compute_type="float16")
print(f"STT(CUDA) ready {HOST}:{PORT}", flush=True)

app = Flask(__name__)


def _run(f, lang):
    lang = None if lang in (None, "auto", "") else lang
    tmp = tempfile.mktemp(suffix=".bin")
    f.save(tmp)
    try:
        segs, info = m.transcribe(tmp, language=lang, condition_on_previous_text=False)
        text = "".join(s.text for s in segs).strip()
        return {"text": text, "language": info.language}
    finally:
        try: os.remove(tmp)
        except Exception: pass


@app.post("/v1/audio/transcriptions")
def transcriptions():
    # OpenAI shape: field name `file`; `model` form field is accepted+ignored
    # (the loaded model is fixed). Tolerate `audio` too for the legacy client.
    f = request.files.get("file") or request.files.get("audio")
    if f is None:
        return jsonify({"error": "no audio file (field name: file)"}), 400
    lang = request.form.get("language") or request.args.get("language") or "auto"
    return jsonify(_run(f, lang))


@app.post("/transcribe")
def transcribe():
    f = request.files.get("audio") or request.files.get("file")
    if f is None:
        return jsonify({"error": "no audio file (field name: audio)"}), 400
    lang = request.form.get("language") or request.args.get("language") or "auto"
    return jsonify(_run(f, lang))


@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
