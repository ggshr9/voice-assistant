"""实时字幕网页后端:HTTPS 静态页 + WSS 音频流。
浏览器共享标签页声音 → 推 16k int16 PCM → 服务端 VAD 切句 → faster-whisper(CUDA)转写
→ litellm 网关 Qwen3.6 译中文 → 推回 {orig, zh}。复用今晚验证过的 STT/翻译。"""
import asyncio, json, os, ssl, uuid, urllib.request
import numpy as np
import webrtcvad
from aiohttp import web
from faster_whisper import WhisperModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
LLM_URL = os.environ.get("CAPTION_LLM_URL", "http://10.0.0.1:4000/v1/chat/completions")
LLM_KEY = os.environ.get("CAPTION_LLM_KEY", "")
LLM_MODEL = os.environ.get("CAPTION_LLM_MODEL", "Qwen3.6")
SR, FRAME = 16000, 480           # 30ms 帧
SILENCE_TAIL, MIN_SPEECH = 0.8, 0.4
ACCESS_PW = os.environ.get("CAPTION_ACCESS_PW", "")   # 设了就要口令

print(f"加载 {MODEL} 到 CUDA ...", flush=True)
model = WhisperModel(MODEL, device="cuda", compute_type="float16")
print("网页字幕后端就绪", flush=True)

_HALLU = {"you", "thank you", "thanks", ".", "", "请订阅", "谢谢观看", "字幕"}


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


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    if ACCESS_PW and request.query.get("pw", "") != ACCESS_PW:
        await ws.send_json({"error": "口令错误"})
        await ws.close()
        return ws
    await ws.send_json({"status": "ready"})       # 通知前端:已连接就绪
    vad = webrtcvad.Vad(2)
    buf, seg = bytearray(), bytearray()
    triggered, silence, nspeech = False, 0.0, 0
    loop = asyncio.get_event_loop()
    async for msg in ws:
        if msg.type != web.WSMsgType.BINARY:
            continue
        buf.extend(msg.data)
        while len(buf) >= FRAME * 2:
            frame = bytes(buf[:FRAME * 2]); del buf[:FRAME * 2]
            speech = vad.is_speech(frame, SR)
            if not triggered:
                if speech:
                    triggered, seg, silence, nspeech = True, bytearray(frame), 0.0, 1
            else:
                seg.extend(frame)
                if speech:
                    silence, nspeech = 0.0, nspeech + 1
                else:
                    silence += 0.03
                    if silence >= SILENCE_TAIL:
                        if nspeech * 0.03 >= MIN_SPEECH:
                            pcm = bytes(seg)
                            try:
                                text, lang = await loop.run_in_executor(None, transcribe_pcm, pcm)
                                if not is_noise(text):
                                    zh = await loop.run_in_executor(None, translate, text, lang)
                                    await ws.send_json({"orig": text, "zh": zh})
                            except Exception as e:
                                await ws.send_json({"error": "服务异常: " + str(e)[:60]})
                        triggered, silence, nspeech = False, 0.0, 0
    return ws


async def index(request):
    return web.FileResponse(os.path.join(HERE, "index.html"))


# ---------- 会议纪要(批处理上传)----------
JOBS = {}
MEETING_PY = os.path.join(HERE, "meeting", "meeting_pipeline.py")
PY_MEETING = os.path.expanduser("~/voice-svc/.venv-meeting/bin/python")
_KEEP = ("准备", "转写", "分人", "合并", "生成", "纪要", "记录", "完成", "transcribed", "diarized")


async def meeting_page(request):
    return web.FileResponse(os.path.join(HERE, "meeting.html"))


async def meeting_upload(request):
    job = uuid.uuid4().hex[:8]
    outdir = f"/tmp/meeting_job_{job}"
    os.makedirs(outdir, exist_ok=True)
    fields, audio = {}, None
    reader = await request.multipart()
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "audio":
            audio = os.path.join(outdir, "in_" + os.path.basename(field.filename or "rec"))
            with open(audio, "wb") as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
        else:
            fields[field.name] = (await field.read()).decode("utf-8", "ignore")
    if ACCESS_PW and fields.get("pw", "") != ACCESS_PW:
        return web.json_response({"error": "口令错误"})
    if not audio:
        return web.json_response({"error": "没收到音频"})
    JOBS[job] = {"progress": [], "done": False}
    asyncio.create_task(run_meeting_job(job, audio, outdir, fields.get("language", "zh"), fields.get("me", "")))
    return web.json_response({"job": job})


async def run_meeting_job(job, audio, outdir, lang, me):
    try:
        proc = await asyncio.create_subprocess_exec(
            PY_MEETING, MEETING_PY, audio, outdir, me or "-", lang or "zh",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PATH": "/usr/bin:" + os.environ.get("PATH", "")})
        async for line in proc.stdout:
            s = line.decode("utf-8", "ignore").strip()
            if s and any(k in s for k in _KEEP):
                JOBS[job]["progress"].append(s)
        await proc.wait()
        m, r = os.path.join(outdir, "会议纪要.md"), os.path.join(outdir, "会议记录.md")
        if os.path.exists(m) and os.path.exists(r):
            JOBS[job]["minutes"] = open(m, encoding="utf-8").read()
            JOBS[job]["record"] = open(r, encoding="utf-8").read()
        else:
            JOBS[job]["error"] = "处理未产出结果"
    except Exception as e:
        JOBS[job]["error"] = str(e)[:120]
    JOBS[job]["done"] = True


async def meeting_status(request):
    j = JOBS.get(request.match_info["job"])
    if not j:
        return web.json_response({"error": "无此任务"}, status=404)
    return web.json_response({"all": j["progress"], "done": j["done"],
                              "minutes": j.get("minutes"), "record": j.get("record"), "error": j.get("error")})


def main():
    app = web.Application(client_max_size=512 * 1024 * 1024)   # 允许大录音上传(512MB)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/meeting", meeting_page)
    app.router.add_post("/meeting/upload", meeting_upload)
    app.router.add_get("/meeting/status/{job}", meeting_status)
    cert = os.environ.get("CAPTION_CERT",
                          os.path.expanduser("~/voice-svc/le/live/caption.example.com/fullchain.pem"))
    key = os.environ.get("CAPTION_KEY",
                         os.path.expanduser("~/voice-svc/le/live/caption.example.com/privkey.pem"))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8443")), ssl_context=ctx)


if __name__ == "__main__":
    main()
