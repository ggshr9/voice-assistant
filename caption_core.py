"""实时字幕的纯逻辑:VAD 切句 / STT / 翻译 / 幻觉过滤。不碰 GUI、不直接开音频设备。"""
import os, json, re, wave, urllib.request, urllib.error
from collections import Counter

SAMPLE_RATE = 16000
FRAME_MS = 30

# 服务地址必须显式配置(见 README / ~/.config/caption.env),不设内网默认值
STT_URL = os.environ.get("CAPTION_STT_URL", "")
STT_MODE = os.environ.get("CAPTION_STT_MODE", "upload")   # upload=传字节(远端) / path=传路径(本机 mlx 服务)
LLM_URL = os.environ.get("CAPTION_LLM_URL", "")
LLM_KEY = os.environ.get("CAPTION_LLM_KEY", "")
LLM_MODEL = os.environ.get("CAPTION_LLM_MODEL", "Qwen3.6")

HALLU_PHRASES = ["点赞", "订阅", "转发", "打赏", "字幕", "明镜", "点点栏目",
                 "感谢观看", "谢谢观看", "谢谢大家", "下期再见", "志愿者",
                 "请不吝", "关注我", "Amara", "字幕组"]
_HALLU_EXACT = {"字幕", "谢谢观看", "请订阅", "by", "you", ".", "", "thank you", "thanks"}


def _require(url, env_name):
    """地址没配就当场报清楚,别静默打到错的主机。"""
    if not url:
        raise RuntimeError(
            f"{env_name} 未配置。在 ~/.config/caption.env 里设好服务地址后重试(见 README「配置」)。")
    return url


def segment_frames(frames, vad, sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS,
                   silence_tail=0.8, min_speech=0.4, max_speech=None):
    """输入 30ms int16 PCM 帧迭代器,按停顿切句,产出每段拼接 PCM bytes(丢弃过短段)。

    时长门限只算【语音帧】(不含末尾静音);产出时裁掉末尾静音、留 ~150ms 自然收尾。

    Args:
        max_speech: 单段语音的秒数上限;超了就地断开,不等停顿。默认 None=不限。
            字幕场景按停顿切就够(默认参数就是为它调的);但会议里有人一口气说两分钟
            很常见 —— 实测真实会议音频切出过 130 秒一整段,靠它给录音反馈就等于没反馈。
    """
    pad = 5
    buf, triggered, silence, nspeech, last_speech = [], False, 0.0, 0, -1

    def _emit():
        if nspeech * frame_ms / 1000 >= min_speech:
            end = min(len(buf), last_speech + 1 + pad)
            return b"".join(buf[:end])
        return None

    for chunk in frames:
        speech = vad.is_speech(chunk, sample_rate)
        if not triggered:
            if speech:
                triggered = True; buf = [chunk]; silence = 0.0
                nspeech, last_speech = 1, 0
        else:
            buf.append(chunk)
            if speech:
                silence = 0.0; nspeech += 1; last_speech = len(buf) - 1
                if max_speech and nspeech * frame_ms / 1000 >= max_speech:
                    seg = _emit()           # 说太久了,就地断开
                    if seg is not None:
                        yield seg
                    buf, triggered, silence, nspeech, last_speech = [], False, 0.0, 0, -1
            else:
                silence += frame_ms / 1000
                if silence >= silence_tail:
                    seg = _emit()
                    if seg is not None:
                        yield seg
                    buf, triggered, silence, nspeech, last_speech = [], False, 0.0, 0, -1
    if triggered:
        seg = _emit()
        if seg is not None:
            yield seg


def pcm_to_wav(pcm, path, sample_rate=SAMPLE_RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm)


def stt(path, url=None):
    """转写一段 wav。upload 模式:读字节 multipart 上传(远端服务);path 模式:传本地路径(本机服务)。"""
    url = _require(url or STT_URL, "CAPTION_STT_URL")
    if STT_MODE == "path":
        body = json.dumps({"path": path, "language": "auto"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    else:
        with open(path, "rb") as f:
            audio = f.read()
        b = "----captionboundary7c3f"
        data = (
            f"--{b}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nauto\r\n".encode()
            + f"--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
              f"Content-Type: audio/wav\r\n\r\n".encode()
            + audio + f"\r\n--{b}--\r\n".encode()
        )
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return d.get("text", "").strip(), d.get("language", "")


_TRANS_SYS = ("你是同声传译。把用户给的这句话翻成简洁、口语化的简体中文,"
              "只输出译文本身,不要解释、不要原文、不要引号。")


def model_chain(spec=None):
    """CAPTION_LLM_MODEL 支持逗号分隔的候选链,前面的挂了自动退到后面。

    **为什么需要**:网关上的模型会被下线。实测公司网关的 Qwen3.6 仍在 /models
    列表里,chat 却返回 404("Received Model Group=Qwen3.6")——
    单点配置意味着字幕直接哑掉,而且报错还看不出是模型没了。
    """
    raw = LLM_MODEL if spec is None else spec
    out = []
    for m in (raw or "").split(","):
        m = m.strip()
        if m and m not in out:
            out.append(m)
    return out or [""]


def translate(text, src_lang, url=None):
    if src_lang == "zh" or not text:
        return text
    url = _require(url or LLM_URL, "CAPTION_LLM_URL")
    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = f"Bearer {LLM_KEY}"
    body = {
        "model": None, "max_tokens": 200, "temperature": 0.2,   # model 由下面的候选链填
        # 必须显式关思考。开着的话本机大脑会把推理过程当译文吐出来 ——
        # 而且是明文、没有 <think> 标签，下面那行 re.sub 剥不掉。
        # 实测漏出来的是「Here's a thinking process: 1. **Analyze User Input:**…」，
        # 糊在字幕浮窗上等于整条链路废掉。设计文档当初就写了这条，实现漏了。
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": _TRANS_SYS},
                     {"role": "user", "content": text}],
    }
    def _post(payload):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    errs = []
    for model in model_chain():
        payload = dict(body, model=model)
        try:
            out = _post(payload)
        except urllib.error.HTTPError as e:
            # 严格端点(OpenAI/DeepSeek 等)不认 vLLM 扩展字段 —— 脱掉重试。
            # LLM_URL 是让用户随便填的，不能假设对端是 vLLM。bin/minutes 里有同样的处理。
            if e.code in (400, 422):
                try:
                    out = _post({k: v for k, v in payload.items()
                                 if k != "chat_template_kwargs"})
                except Exception as e2:            # noqa: BLE001
                    errs.append(f"{model}: {e2}")
                    continue
            elif e.code == 404:                    # 这个模型下线了,换下一个
                errs.append(f"{model}: 404 模型不存在")
                continue
            else:
                raise
        except Exception as e:                     # noqa: BLE001
            errs.append(f"{model}: {type(e).__name__} {e}")
            continue
        if "choices" not in out:                   # 网关的错误体也可能是 200
            errs.append(f"{model}: {str(out.get('error', out))[:120]}")
            continue
        zh = (out["choices"][0]["message"].get("content") or "").strip()
        zh = re.sub(r"<think>.*?</think>", "", zh, flags=re.S).strip()
        if zh:
            return zh
        errs.append(f"{model}: 空译文")
    raise RuntimeError("翻译无可用模型 —— " + " | ".join(errs[-3:]))


def is_noise(text):
    """空串 / whisper 幻觉 / 字幕台词 → True(丢弃)。"""
    c = text.strip("。.，,！!？? ").lower()
    if not c or c in _HALLU_EXACT or len(c) < 2:
        return True
    if any(p in text for p in HALLU_PHRASES):
        return True
    t = re.sub(r"[，。！？、\s,.!?]", "", text)
    if len(t) >= 4 and Counter(t).most_common(1)[0][1] >= max(5, len(t) * 0.5):
        return True
    return False
