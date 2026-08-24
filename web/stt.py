"""实时 STT(faster-whisper / CUDA) + 翻译(litellm 网关)。"""
import os, json, asyncio, urllib.request
import numpy as np
from config import MODEL, LLM_URL, LLM_KEY, LLM_MODEL

STT_LOCK = asyncio.Lock()   # 串行化 GPU 转写,避免并发会话同时调 model.transcribe 互相串
_HALLU = {"you", "thank you", "thanks", ".", "", "请订阅", "谢谢观看", "字幕"}

if os.environ.get("SKIP_MODEL"):           # 测试/导入冒烟用,不占 GPU
    model = None
    print("STT: 跳过模型加载(SKIP_MODEL)", flush=True)
else:
    from faster_whisper import WhisperModel
    print(f"加载 {MODEL} 到 CUDA ...", flush=True)
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    print("会议工作台后端就绪", flush=True)


def transcribe_pcm(pcm):
    audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    segs, info = model.transcribe(audio, language=None, condition_on_previous_text=False)
    return "".join(s.text for s in segs).strip(), info.language


def is_noise(text):
    c = text.strip().strip("。.,，!！?？ ").lower()
    return (not c) or len(c) < 2 or c in _HALLU


def translate(text, lang):
    if lang == "zh" or not text:
        return text
    body = json.dumps({
        "model": LLM_MODEL, "max_tokens": 200, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是同声传译。把这句话翻成简洁、口语化的简体中文,只输出译文本身,不要解释、不要原文、不要引号。"},
            {"role": "user", "content": text}],
    }).encode()
    h = {"Content-Type": "application/json"}
    if LLM_KEY:
        h["Authorization"] = f"Bearer {LLM_KEY}"
    req = urllib.request.Request(LLM_URL, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def translate_stream(text, lang):
    """流式翻译:逐块 yield 中文增量(打字机效果)。lang==zh 或空则直接 yield 原文。"""
    if lang == "zh" or not text:
        yield text
        return
    body = json.dumps({
        "model": LLM_MODEL, "max_tokens": 200, "temperature": 0.2, "stream": True,
        "messages": [
            {"role": "system", "content": "你是同声传译。把这句话翻成简洁、口语化的简体中文,只输出译文本身,不要解释、不要原文、不要引号。"},
            {"role": "user", "content": text}],
    }).encode()
    h = {"Content-Type": "application/json"}
    if LLM_KEY:
        h["Authorization"] = f"Bearer {LLM_KEY}"
    req = urllib.request.Request(LLM_URL, data=body, headers=h)
    got = False
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:                                  # 逐行读 SSE
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                tok = (json.loads(data)["choices"][0].get("delta", {}) or {}).get("content") or ""
            except Exception:
                tok = ""
            if tok:
                got = True
                yield tok
    if not got:                                        # 流式没拿到(网关不支持等)→回退整句
        yield translate(text, lang)
