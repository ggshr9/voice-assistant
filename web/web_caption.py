"""会议工作台后端:HTTPS 静态页 + WSS 实时字幕 + 路由。
实时:浏览器推 16k int16 PCM → VAD 切句 → faster-whisper(CUDA)转写 → Qwen3.6 译中文 → 推回 {orig,zh}。
落盘:每场会议是一个 session,录音(recording.pcm/.wav)与实时转写(live.jsonl)边走边存。
生成:停止后拿 recording.wav 跑 meeting_pipeline.py 出 会议纪要/会议记录。
模块:config(配置) / stt(转写翻译) / sessions(会议存储) / jobs(纪要任务) / 本文件(路由)。"""
import asyncio, json, os, shutil, ssl, time, uuid, glob
import numpy as np
import webrtcvad
from aiohttp import web

from config import (HERE, SESSIONS, SR, FRAME, SILENCE_TAIL, MIN_SPEECH,
                    RETENTION_DAYS, SESSIONS_WARN_GB, MAX_UPLOAD_BYTES,
                    MAX_SEG_SEC)
from stt import STT_LOCK, transcribe_pcm, is_noise, translate, translate_stream
from sessions import (check_pw, new_session, sess_dir, read_meta, write_meta,
                      finalize_wav, wav_seconds, recording_path, dir_size_bytes, prune_old_audio)
from jobs import JOBS, prune_jobs, run_meeting_job


_COMPRESS = {}   # sid -> 正在进行的压缩任务,供生成纪要前等待,避免读到正被删除的 wav


async def compress_recording(d):
    """停止后台把 recording.wav 转 Opus(~10x 小),成功后删 wav。pipeline/播放都能读 opus。"""
    wav = os.path.join(d, "recording.wav")
    opus = os.path.join(d, "recording.opus")
    tmp = opus + ".tmp"
    if not os.path.exists(wav):
        return
    try:
        proc = await asyncio.create_subprocess_exec(   # 原子写:先写 .tmp 再 rename;-f opus 显式指定格式(.tmp 扩展名 ffmpeg 猜不出)
            "ffmpeg", "-y", "-i", wav, "-c:a", "libopus", "-b:a", "24k", "-application", "voip", "-f", "opus", tmp,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if proc.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, opus)
            os.remove(wav)
    except Exception:
        pass


async def _retention_loop(app):
    """每 6 小时:按保留期清理超期录音(保留文字),并在录音目录过大时告警。"""
    while True:
        try:
            n = prune_old_audio(RETENTION_DAYS)
            if n:
                print(f"保留策略:清理 {n} 个超期录音文件(保留文字)", flush=True)
            gb = dir_size_bytes(SESSIONS) / 2 ** 30
            if gb > SESSIONS_WARN_GB:
                print(f"⚠️ 会议录音目录 {gb:.1f}GB 超过告警阈值 {SESSIONS_WARN_GB}GB", flush=True)
        except Exception:
            pass
        await asyncio.sleep(6 * 3600)


async def _on_startup(app):
    app["_retention"] = asyncio.create_task(_retention_loop(app))


# ---------- 实时字幕 + 录音 ----------
async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    if not check_pw(request.query.get("pw", "")):
        await ws.send_json({"error": "口令错误"})
        await ws.close()
        return ws
    d = sess_dir(request.query.get("sid", ""))   # 有合法 sid 才落盘;否则纯显示
    ch = 2 if request.query.get("ch") == "2" else 1   # 双声道:L=对方,R=我
    if d:                                          # 记录声道数,封 wav 时用
        m = read_meta(d) or {}
        if m.get("channels") != ch:
            m["channels"] = ch
            write_meta(d, m)
    # 界面上选的语言要真的用上 —— 从前它存进 meta.json 就没人再看,
    # live 字幕永远走自动检测。实测安静的中文被判成 en 的置信度只有 0.51,
    # 于是「哈喽哈喽」被音译成 "Honey, honey"。
    sess_lang = ((read_meta(d) or {}).get("lang") if d else None) or ""
    stt_lang = sess_lang if sess_lang in ("zh", "en") else None
    pcm_f = open(os.path.join(d, "recording.pcm"), "ab") if d else None
    jsonl_f = open(os.path.join(d, "live.jsonl"), "a", encoding="utf-8") if d else None
    await ws.send_json({"status": "ready"})
    vad = webrtcvad.Vad(2)
    buf, seg = bytearray(), bytearray()
    triggered, silence, nspeech, since_partial = False, 0.0, 0, 0
    cur_sid = 0                       # 句子编号在主循环里分配:草稿与定稿同号,前端对得上
    loop = asyncio.get_event_loop()
    seg_q = asyncio.Queue()           # (sid, pcm) 定稿整句 → 消费者顺序转写+翻译
    partial = {"sid": 0, "pcm": None} # 只保留"最新草稿"(说到一半的当前句),边说边转写
    partial_ev = asyncio.Event()
    last_final = {"v": 0}             # 已定稿的最大 sid:迟到的草稿不再覆盖
    state = {"stop": False}

    async def safe_send(obj):
        try:
            await ws.send_json(obj)
        except Exception:
            pass

    async def drafter_task():
        """边说边出字:把当前没说完的句子(最新快照)转成草稿推前端,不翻译。
        只认最新快照、给定稿让路,避免堆积拖慢 GPU。"""
        while not state["stop"]:
            try:
                await asyncio.wait_for(partial_ev.wait(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            partial_ev.clear()
            sid, pcm = partial["sid"], partial["pcm"]
            if pcm is None or sid <= last_final["v"]:
                continue
            if not seg_q.empty():                 # 有定稿排队 → 先让定稿走
                partial_ev.set()
                await asyncio.sleep(0.03)
                continue
            try:
                async with STT_LOCK:
                    text, lang = await loop.run_in_executor(
                        None, transcribe_pcm, pcm, stt_lang)
            except Exception:
                continue
            if sid > last_final["v"] and text and not is_noise(text):
                await safe_send({"seg": sid, "ip": text, "iz": lang == "zh"})

    async def consume():
        """定稿句:转写 →(外语)流式翻译推前端 → 落定 → 写 jsonl。sid 与草稿同源。"""
        while True:
            item = await seg_q.get()
            if item is None:
                break
            sid, pcm = item
            try:
                async with STT_LOCK:
                    text, lang = await loop.run_in_executor(
                        None, transcribe_pcm, pcm, stt_lang)
                last_final["v"] = sid                          # 定稿,迟到草稿失效
                if is_noise(text):
                    await safe_send({"seg": sid, "done": True, "zh": ""})  # 撤掉草稿行
                    continue
                if lang == "zh":                               # 中文:草稿已是中文,直接落定
                    await safe_send({"seg": sid, "done": True, "zh": text})
                    top, zh = "", text
                else:
                    top = text
                    await safe_send({"seg": sid, "orig": top})
                    q = asyncio.Queue()                        # 流式翻译 token

                    def _w(t=text, lg=lang):
                        try:
                            for tok in translate_stream(t, lg):
                                loop.call_soon_threadsafe(q.put_nowait, tok)
                        except Exception:
                            pass
                        loop.call_soon_threadsafe(q.put_nowait, None)

                    loop.run_in_executor(None, _w)
                    parts = []
                    while True:
                        tok = await q.get()
                        if tok is None:
                            break
                        parts.append(tok)
                        await safe_send({"seg": sid, "d": tok})
                    zh = "".join(parts).strip()
                    await safe_send({"seg": sid, "done": True, "zh": zh})
                if jsonl_f:                                      # jsonl 必写(即便 ws 已关,送失败也不影响)
                    jsonl_f.write(json.dumps(
                        {"t": time.strftime("%H:%M:%S"), "orig": top, "zh": zh},
                        ensure_ascii=False) + "\n")
                    jsonl_f.flush()
            except Exception as e:
                await safe_send({"error": "服务异常: " + str(e)[:60]})

    consumer = asyncio.create_task(consume())
    drafter = asyncio.create_task(drafter_task())
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:   # 客户端"音频发完了"(停止录制)→ 跳出去收尾尾句
                break
            if msg.type != web.WSMsgType.BINARY:
                continue
            if pcm_f:                       # 整段录音边收边写(可能是双声道交织)
                pcm_f.write(msg.data)
            if ch == 2:                     # VAD/STT 用单声道:L+R 钳位求和(L/R 很少同时大声,保电平)
                s = np.frombuffer(msg.data, dtype="<i2")
                mono = np.clip(s[0::2].astype(np.int32) + s[1::2].astype(np.int32), -32768, 32767).astype("<i2").tobytes()
            else:
                mono = bytes(msg.data)
            buf.extend(mono)
            while len(buf) >= FRAME * 2:
                frame = bytes(buf[:FRAME * 2]); del buf[:FRAME * 2]
                speech = vad.is_speech(frame, SR)
                if not triggered:
                    if speech:
                        cur_sid += 1
                        triggered, seg, silence, nspeech, since_partial = True, bytearray(frame), 0.0, 1, 0
                else:
                    seg.extend(frame)
                    if speech:
                        silence, nspeech = 0.0, nspeech + 1
                    else:
                        silence += 0.03
                    since_partial += 1
                    if nspeech * 0.03 >= MIN_SPEECH and since_partial * 0.03 >= 0.7:
                        since_partial = 0                        # 每 ~0.7s 推一版草稿(只留最新)
                        partial["sid"], partial["pcm"] = cur_sid, bytes(seg)
                        partial_ev.set()
                    # 断句只靠静音是不够的:有人一口气说两分钟,就攒出一个两分钟的段。
                    # 那既拖垮延迟,也让模型更容易跑飞。本机版早就加了 MAX_SPEECH=12,
                    # 服务端一直没有 —— 实测真会议里出现过 130 秒的整段。
                    too_long = len(seg) >= MAX_SEG_SEC * SR * 2
                    if silence >= SILENCE_TAIL or too_long:
                        if nspeech * 0.03 >= MIN_SPEECH:
                            seg_q.put_nowait((cur_sid, bytes(seg)))  # 交给消费者,主循环立刻继续读音频
                        if too_long and silence < SILENCE_TAIL:
                            cur_sid += 1                 # 硬断:话没说完,下一段接着算新句
                            seg, since_partial = bytearray(), 0
                            silence, nspeech = 0.0, 1
                        else:
                            triggered, silence, nspeech = False, 0.0, 0
    finally:
        if triggered and nspeech * 0.03 >= MIN_SPEECH:   # 收尾:停止时没等到静音的最后一句也补上
            seg_q.put_nowait((cur_sid, bytes(seg)))
        seg_q.put_nowait(None)                            # 让消费者把剩余队列处理完再退出
        state["stop"] = True
        partial_ev.set()
        try:
            await asyncio.wait_for(consumer, timeout=30)
        except Exception:
            consumer.cancel()
        try:
            await asyncio.wait_for(drafter, timeout=3)
        except Exception:
            drafter.cancel()
        if pcm_f:
            pcm_f.close()
        if jsonl_f:
            jsonl_f.close()
    return ws


# ---------- 页面 ----------
async def index(request):
    return web.FileResponse(os.path.join(HERE, "index.html"))


# ---------- session 路由 ----------
async def session_start(request):
    body = await request.json()
    if not check_pw(body.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    meta = new_session(body.get("scene"), body.get("lang"), body.get("title"))
    return web.json_response({"id": meta["id"], "meta": meta})


async def session_finish(request):
    if not check_pw(request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.json_response({"error": "无此会议"}, status=404)
    finalize_wav(d)
    meta = read_meta(d) or {}
    meta["ended"] = time.time()
    meta["duration"] = wav_seconds(d) or int(meta["ended"] - meta.get("started", meta["ended"]))
    meta["status"] = "recorded"
    write_meta(d, meta)
    sid = meta.get("id") or os.path.basename(d)   # 后台压成 Opus,不阻塞响应;登记任务供生成时等待
    t = asyncio.create_task(compress_recording(d))
    _COMPRESS[sid] = t
    t.add_done_callback(lambda _: _COMPRESS.pop(sid, None))
    return web.json_response({"ok": True, "meta": meta})


async def session_rename(request):
    if not check_pw(request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.json_response({"error": "无此会议"}, status=404)
    body = await request.json() if request.can_read_body else {}
    title = (body.get("title") or "").strip()
    if not title:
        return web.json_response({"error": "标题不能为空"}, status=400)
    meta = read_meta(d) or {}
    meta["title"] = title
    write_meta(d, meta)
    return web.json_response({"ok": True, "title": title})


async def session_delete(request):
    if not check_pw(request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.json_response({"error": "无此会议"}, status=404)
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return web.json_response({"ok": True})


async def session_minutes(request):
    # 口令要排在查会话之前 —— 否则 404/403 的差别会向未鉴权者泄漏「这个会话存不存在」。
    # 其余写接口(rename/delete)本来就是先验口令,这里从前是反的。
    body = await request.json() if request.can_read_body else {}
    if not check_pw(body.get("pw", "") or request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.json_response({"error": "无此会议"}, status=404)
    finalize_wav(d)                  # pcm→wav(若刚停还没封);已压成 opus 时无操作
    t = _COMPRESS.get(os.path.basename(d))   # 若正在压成 opus,先等它完成,避免拿到正被删的 wav
    if t and not t.done():
        try:
            await asyncio.wait_for(asyncio.shield(t), timeout=30)
        except Exception:
            pass
    audio = recording_path(d)        # opus 或 wav,pipeline 的 ffmpeg 都能读
    if not audio:
        return web.json_response({"error": "没有录音可生成"}, status=400)
    meta = read_meta(d) or {}
    meta["status"] = "processing"
    write_meta(d, meta)
    job = uuid.uuid4().hex[:8]
    prune_jobs(); JOBS[job] = {"progress": [], "done": False, "sid": meta.get("id")}
    asyncio.create_task(run_meeting_job(
        job, audio, d, meta.get("lang", "auto"), body.get("me", "")))
    return web.json_response({"job": job})


async def job_status(request):
    if not check_pw(request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    j = JOBS.get(request.match_info["job"])
    if not j:
        return web.json_response({"error": "无此任务"}, status=404)
    return web.json_response({"all": j["progress"], "done": j["done"],
                              "minutes": j.get("minutes"), "record": j.get("record"),
                              "enhanced": j.get("enhanced"),
                              "error": j.get("error")})


# ---------- 上传已有录音(也是一种 session) ----------
async def session_upload(request):
    reader = await request.multipart()
    fields, audio, d = {}, None, None
    # 口令先于音频校验:从前是整个文件写完盘才判口令,任何人不带口令也能往磁盘里灌。
    # 前端已改成先 append('pw'),再 append('audio');curl 也可以走 ?pw= 。
    authed = check_pw(request.query.get("pw", "")) if request.query.get("pw") else False
    while True:
        field = await reader.next()
        if field is None:
            break
        if field.name == "audio":
            if not authed:
                return web.json_response(
                    {"error": "口令错误（口令字段须排在音频之前，或用 ?pw= 传）"}, status=403)
            d = os.path.join(SESSIONS, time.strftime("%Y%m%d_%H%M") + "_上传_" + uuid.uuid4().hex[:4])
            os.makedirs(d, exist_ok=True)
            ext = os.path.splitext(field.filename or "rec")[1].lower() or ".bin"
            audio = os.path.join(d, "recording" + ext)   # 存成 recording.<ext>,事后可下载
            size = 0
            try:
                with open(audio, "wb") as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:      # 免得一个请求把盘写满
                            raise ValueError("文件过大")
                        f.write(chunk)
            except ValueError:
                shutil.rmtree(d, ignore_errors=True)
                return web.json_response(
                    {"error": f"文件超过 {MAX_UPLOAD_BYTES // (1 << 30)}GB 上限"}, status=413)
        else:
            fields[field.name] = (await field.read()).decode("utf-8", "ignore")
            if field.name == "pw":
                authed = authed or check_pw(fields["pw"])
    if not authed:
        return web.json_response({"error": "口令错误"}, status=403)
    if not audio:
        return web.json_response({"error": "没收到音频"}, status=400)
    meta = {"id": os.path.basename(d), "title": fields.get("title", "").strip() or time.strftime("%m-%d %H:%M 上传"),
            "scene": "上传", "lang": fields.get("language", "auto"),
            "started": time.time(), "ended": time.time(), "status": "processing", "duration": 0}
    write_meta(d, meta)
    job = uuid.uuid4().hex[:8]
    prune_jobs(); JOBS[job] = {"progress": [], "done": False, "sid": meta["id"]}
    asyncio.create_task(run_meeting_job(job, audio, d, fields.get("language", "auto"), fields.get("me", "")))
    return web.json_response({"id": meta["id"], "job": job})


# ---------- 会议库 ----------
async def session_notes(request):
    """保存用户手记(录制中自动保存,会后也可改)。

    口令在查会话之前 —— 404/403 的差别会向未鉴权者泄漏会话存不存在(session_minutes
    犯过同样的错)。写入走 tmp+replace:自动保存每几秒打一次,截断式写法碰上
    刷新/断电就把笔记清空了,而那是用户唯一手打的东西,比转写更不可再生。
    """
    body = await request.json() if request.can_read_body else {}
    if not check_pw(body.get("pw", "") or request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.json_response({"error": "无此会议"}, status=404)
    text = body.get("text", "")
    if not isinstance(text, str):
        return web.json_response({"error": "text 需要字符串"}, status=400)
    if len(text) > 1_000_000:
        return web.json_response({"error": "笔记过大"}, status=413)
    tmp = os.path.join(d, ".notes.md.part")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, os.path.join(d, "notes.md"))
    return web.json_response({"ok": True, "chars": len(text)})


async def sessions_list(request):
    if not check_pw(request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    out = []
    for d in glob.glob(os.path.join(SESSIONS, "*")):
        m = read_meta(d)
        if m:
            out.append({k: m.get(k) for k in
                        ("id", "title", "scene", "lang", "started", "ended", "status", "duration")})
    out.sort(key=lambda x: x.get("started") or 0, reverse=True)
    return web.json_response({"sessions": out})


async def session_get(request):
    if not check_pw(request.query.get("pw", "")):
        return web.json_response({"error": "口令错误"}, status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.json_response({"error": "无此会议"}, status=404)
    m = read_meta(d) or {}
    live = []
    jp = os.path.join(d, "live.jsonl")
    if os.path.exists(jp):
        for ln in open(jp, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    live.append(json.loads(ln))
                except Exception:
                    pass

    def _read(name):
        p = os.path.join(d, name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else None
    rec = recording_path(d)
    return web.json_response({"meta": m, "live": live,
                              "minutes": _read("会议纪要.md"), "record": _read("会议记录.md"),
                              "notes": _read("notes.md"), "enhanced": _read("增强笔记.md"),
                              "has_audio": rec is not None,
                              "audio_name": os.path.basename(rec) if rec else None,
                              "has_pcm": os.path.exists(os.path.join(d, "recording.pcm"))})


async def session_audio(request):
    if not check_pw(request.query.get("pw", "")):
        return web.Response(status=403)
    d = sess_dir(request.match_info["id"])
    if not d:
        return web.Response(status=404)
    p = recording_path(d)
    if not p:
        return web.Response(status=404)
    return web.FileResponse(p)


def main():
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app.on_startup.append(_on_startup)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    # session API
    app.router.add_post("/session/start", session_start)
    app.router.add_post("/session/upload", session_upload)
    app.router.add_post("/session/{id}/finish", session_finish)
    app.router.add_post("/session/{id}/rename", session_rename)
    app.router.add_post("/session/{id}/minutes", session_minutes)
    app.router.add_post("/session/{id}/notes", session_notes)
    app.router.add_post("/session/{id}/delete", session_delete)
    app.router.add_get("/session/{id}/audio", session_audio)
    app.router.add_get("/session/{id}", session_get)
    app.router.add_get("/sessions", sessions_list)
    app.router.add_get("/job/{job}", job_status)
    cert = os.environ.get("CAPTION_CERT",
                          os.path.expanduser("~/voice-svc/le/live/caption.example.com/fullchain.pem"))
    key = os.environ.get("CAPTION_KEY",
                         os.path.expanduser("~/voice-svc/le/live/caption.example.com/privkey.pem"))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8443")), ssl_context=ctx)


if __name__ == "__main__":
    main()
