# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice", "webrtcvad", "numpy", "setuptools<81", "opencc"]
# ///
"""
朋友式连续语音对话「小麦」：
免按键，VAD 自动检测你说话/停顿，说完它自动接话；用克隆音回答。
你说话时它会停下（可打断）。按 Ctrl+C 退出。

依赖三个本地服务：STT(8082) / LLM(8080) / TTS(8081)
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

import numpy as np
import sounddevice as sd
import webrtcvad

STT_URL = "http://127.0.0.1:18091/transcribe"   # 统一到共享 ASR 服务(按需加载)
LLM_URL = "http://127.0.0.1:18080/v1/chat/completions"  # 共享 35B,闲时自动释放
TTS_URL = "http://127.0.0.1:18083/v1/audio/speech"      # 共享 TTS

# 声音：默认 Qwen3-TTS 克隆(女朋友的声音，长句稳定)；VA_TTS=kokoro 用预设音
TTS_MODE = os.environ.get("VA_TTS", "clone")
QWEN_TTS = os.path.expanduser("~/models/Qwen3-TTS-1.7B")
INDEXTTS = os.path.expanduser("~/models/IndexTTS-1.5")
KOKORO = os.path.expanduser("~/models/Kokoro-82M-bf16")
KOKORO_VOICE = os.environ.get("VA_VOICE", "zf_xiaoxiao")
CLONE_REF = os.environ.get("VA_CLONE_REF", os.path.expanduser("~/会议录音/降噪_我的声音2.wav"))
CLONE_TEXT = os.environ.get("VA_CLONE_TEXT",
    "大家好我是这台电脑的主人今天天气晴朗阳光明媚我正在测试声音克隆技术希望这段录音能准确地复刻出我的音色和语调")

import datetime
_NOW = datetime.datetime.now()
_WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][_NOW.weekday()]
SYSTEM = (f"你叫小麦，是用户的语音聊天伙伴。现在是 {_NOW.year}年{_NOW.month}月{_NOW.day}日 {_WEEK} "
          f"{_NOW.hour}点{_NOW.minute}分，被问到时间/日期就用这个。"
          "像朋友一样自然聊天：口语化、简短、有温度，可以反问、接梗。"
          "必须用简体中文回答，不要用繁体字。"
          "不要用 markdown、不要列清单，一般一两句话。")

# Whisper 常见成句幻觉（静音/噪音时蹦出来的字幕台词），命中即丢弃
HALLU_PHRASES = ["点赞", "订阅", "转发", "打赏", "字幕", "明镜", "点点栏目",
                 "感谢观看", "谢谢观看", "谢谢大家", "下期再见", "志愿者",
                 "请不吝", "关注我", "Amara", "字幕组"]

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME = SAMPLE_RATE * FRAME_MS // 1000
SILENCE_TAIL = 0.8      # 停顿多久算说完（秒）
MIN_SPEECH = 0.4        # 最短有效语音
HALLU = {"değil", "字幕", "谢谢观看", "请订阅", "by", "you", ".", ""}


def transcribe(wav_path):
    body = json.dumps({"path": wav_path, "language": "zh"}).encode()
    req = urllib.request.Request(STT_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("text", "").strip()


def is_hallucination(text):
    """检测 Whisper 幻觉：某个短词/字反复出现(如'如果如果如果')。"""
    import re
    t = re.sub(r"[，。！？、\s,.!?]", "", text)
    if len(t) < 4:
        return False
    # 单字重复过多(如"呢呢呢呢")
    from collections import Counter
    most = Counter(t).most_common(1)[0][1]
    if most >= max(5, len(t) * 0.5):
        return True
    # 任意 1-4 字片段重复占比过高 → 幻觉
    for n in (1, 2, 3, 4):
        for i in range(0, min(len(t), 8)):
            seg = t[i:i + n]
            if len(seg) == n and t.count(seg) >= 4:
                return True
    return False


def llm(history):
    body = json.dumps({
        "model": "qwen3.6", "messages": history,
        "max_tokens": 200, "temperature": 0.8, "top_p": 0.9,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


try:
    import opencc
    _t2s = opencc.OpenCC("t2s")  # 繁→简
except Exception:
    _t2s = None


def clean(t):
    import re
    t = re.sub(r"[*#`_~>\-]", "", t)
    t = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if _t2s:                      # 转简体，防繁体字崩 Kokoro
        t = _t2s.convert(t)
    return t


def split_clauses(text, maxlen=20):
    """按中文标点切成短句，避免 IndexTTS 长句退化。过短的合并。"""
    import re
    parts = re.split(r"(?<=[。！？!?；;，,])", text)
    clauses, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) <= maxlen:
            buf += p
        else:
            if buf:
                clauses.append(buf)
            buf = p
    if buf:
        clauses.append(buf)
    return clauses or [text]


CLONE_URL = "http://127.0.0.1:18083/speak"


def _synth(clause, mode):
    """kokoro 走 8081 服务；clone(Qwen3-TTS)走 8083 常驻克隆服务(模型热，秒回)。"""
    if mode != "clone":
        payload = {"model": KOKORO, "input": clause, "voice": KOKORO_VOICE, "lang_code": "z"}
        req = urllib.request.Request(TTS_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    payload = {"text": clause, "ref_audio": CLONE_REF, "ref_text": CLONE_TEXT}
    req = urllib.request.Request(CLONE_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def synth_clause(clause, idx):
    """合成一个短句；检测退化(音频时长远超正常)则用 Kokoro 兜底。返回音频文件路径。"""
    p = os.path.join(tempfile.gettempdir(), f"chat_reply_{idx}.mp3")
    # 正常中文 ~0.3s/字；超过 字数*0.8+4 秒判为退化
    limit = len(clause) * 0.8 + 4.0
    if TTS_MODE == "clone":
        try:
            audio = _synth(clause, "clone")
            with open(p, "wb") as f:
                f.write(audio)
            if _duration(p) <= limit:
                return p           # 克隆音正常
            print("  (克隆音退化，本句改用稳定音)")
        except Exception as e:
            print(f"  (克隆失败:{e}，改用稳定音)")
    # 兜底：Kokoro
    try:
        audio = _synth(clause, "kokoro")
        with open(p, "wb") as f:
            f.write(audio)
        return p
    except Exception:
        return None


def _synth_clone_retry(clause, tries=3):
    """clone(Qwen3-TTS)稳，偶发网络抖动重试即可。"""
    last = None
    for _ in range(tries):
        try:
            return _synth(clause, "clone")
        except Exception as e:
            last = e
            time.sleep(0.3)
    raise last


def _synth_chunks_kokoro(clause, depth=0):
    """Kokoro 的 broadcast_shapes/IncompleteRead 崩按音频长度确定性触发，
    重试同串必崩；但切短~一半就能绕开(实测深度1即可全恢复)。
    崩了就二分递归，返回可播放的音频块列表(尽力而为，极端小段仍崩则丢)。"""
    try:
        return [_synth(clause, "kokoro")]
    except Exception:
        if len(clause) < 4 or depth >= 5:
            return []
        mid = len(clause) // 2
        return (_synth_chunks_kokoro(clause[:mid], depth + 1)
                + _synth_chunks_kokoro(clause[mid:], depth + 1))


def speak(text):
    """clone=Qwen3-TTS(女朋友声音，长句稳)；kokoro=预设音。按句切+兜底，单句失败不中断。"""
    mode = "clone" if TTS_MODE == "clone" else "kokoro"
    ext = "wav" if mode == "clone" else "mp3"   # clone 返回 wav，kokoro 返回 mp3
    outdir = os.path.expanduser("~/.cache/va_tts")
    os.makedirs(outdir, exist_ok=True)
    for i, c in enumerate(split_clauses(text, maxlen=40)):
        try:
            chunks = [_synth_clone_retry(c)] if mode == "clone" else _synth_chunks_kokoro(c)
        except Exception as e:
            print(f"  (本句合成失败,跳过: {str(e)[:30]})")
            continue
        if not chunks:
            print(f"  (本句 Kokoro 二分仍失败,跳过: {c[:20]})")
            continue
        for j, audio in enumerate(chunks):
            p = os.path.join(outdir, f"chat_reply_{i}_{j}.{ext}")
            with open(p, "wb") as f:
                f.write(audio)
            subprocess.run(["afplay", p])


def save_wav(frames, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(frames))


def main():
    for name, url in [("STT", "http://127.0.0.1:18091/transcribe"),
                      ("LLM", "http://127.0.0.1:18080/v1/models")]:
        try:
            if name == "STT":
                urllib.request.urlopen(urllib.request.Request(url, data=b'{}',
                    headers={"Content-Type": "application/json"}), timeout=5)
            else:
                urllib.request.urlopen(url, timeout=5)
        except Exception:
            if name == "LLM":
                print(f"❌ {name} 服务未就绪（{url}）。先 llm start")
                sys.exit(1)

    vad = webrtcvad.Vad(2)
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(bytes(indata))

    voice = "女朋友的克隆音" if TTS_MODE == "clone" else f"Kokoro/{KOKORO_VOICE}"
    print(f"🎙️ 连续对话已开始（声音：{voice}）。直接说话即可，Ctrl+C 退出。")
    history = [{"role": "system", "content": SYSTEM}]
    tmp = os.path.join(tempfile.gettempdir(), "chat_in.wav")

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME,
                           dtype="int16", channels=1, callback=cb):
        while True:
            # 等待说话开始
            frames, triggered, silence = [], False, 0
            while True:
                chunk = q.get()
                is_speech = vad.is_speech(chunk, SAMPLE_RATE)
                if not triggered:
                    if is_speech:
                        triggered = True
                        frames = [chunk]
                        print("🔴 听到你说话…", end="", flush=True)
                else:
                    frames.append(chunk)
                    if is_speech:
                        silence = 0
                    else:
                        silence += FRAME_MS / 1000
                        if silence >= SILENCE_TAIL:
                            break
            dur = len(frames) * FRAME_MS / 1000
            if dur < MIN_SPEECH:
                print(" (太短，忽略)")
                continue
            save_wav(frames, tmp)
            print(" 转写中…", end="", flush=True)
            text = transcribe(tmp)
            c = text.strip("。.，,！!？? ").lower()
            hit_phrase = any(p in text for p in HALLU_PHRASES)
            if not c or c in HALLU or len(c) < 2 or is_hallucination(text) or hit_phrase:
                print(" (没听清)")
                continue
            print(f"\n🗣️  你: {text}")
            history.append({"role": "user", "content": text})
            reply = llm(history)
            history.append({"role": "assistant", "content": reply})
            print(f"🤖 小麦: {reply}")
            speak(clean(reply))
            # 播完清空麦克风缓冲，避免把自己的声音/回声当成新输入
            time.sleep(0.2)
            with q.mutex:
                q.queue.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 拜拜")
