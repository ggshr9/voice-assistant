# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-whisper"]
# ///
"""
全本地语音助手：你说话 → mlx-whisper 转写 → 本地 Qwen3.6 → 你的克隆声音念出来。
按回车开始说话，说完再按回车；输 q 退出。
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import termios
import wave
import urllib.request

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
TTS_URL = "http://127.0.0.1:8081/v1/audio/speech"
WHISPER = "mlx-community/whisper-large-v3-turbo"
MIC = os.environ.get("VA_MIC", "MacBook Pro Microphone")

# 声音：clone=用你的克隆音(IndexTTS)  kokoro=Kokoro预设音
TTS_MODE = os.environ.get("VA_TTS", "clone")
KOKORO = os.path.expanduser("~/models/Kokoro-82M-bf16")
KOKORO_VOICE = os.environ.get("VA_VOICE", "zf_xiaoxiao")
INDEXTTS = os.path.expanduser("~/models/IndexTTS-1.5")
CLONE_REF = os.environ.get("VA_CLONE_REF", os.path.expanduser("~/会议录音/降噪_我的声音2.wav"))
CLONE_TEXT = os.environ.get(
    "VA_CLONE_TEXT",
    "大家好我是这台电脑的主人今天天气晴朗阳光明媚我正在测试声音克隆技术希望这段录音能准确地复刻出我的音色和语调",
)

SYSTEM = (
    "你是一个本地语音助手，名叫小麦。回答口语化、简洁，适合朗读——"
    "不要用 markdown、不要列清单、控制在两三句话内。"
)

# Whisper 对静音常见的幻觉，命中就当没听清
HALLUCINATIONS = {"değil", "değil değil", "字幕", "请不吝点赞", "谢谢观看",
                  "谢谢大家", "请订阅", "by", "you", "thank you", "."}

import mlx_whisper  # noqa: E402


def flush_input():
    """清空键盘缓冲，避免排队的回车把流程跳过。"""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def wav_duration(path):
    try:
        with wave.open(path) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def record(path):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "avfoundation", "-i", f":{MIC}",
         "-ac", "1", "-ar", "16000", path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    flush_input()
    input("🔴 录音中… 说完按回车结束")
    proc.send_signal(signal.SIGINT)
    proc.wait()


def transcribe(path):
    r = mlx_whisper.transcribe(
        path, path_or_hf_repo=WHISPER, language="zh",
        condition_on_previous_text=False,
    )
    return r["text"].strip()


def ask_llm(history, model_id):
    body = json.dumps({
        "model": model_id, "messages": history,
        "max_tokens": 250, "temperature": 0.7, "top_p": 0.9,
        "repetition_penalty": 1.1,             # 防止量化模型重复退化
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(LLM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return (data["choices"][0]["message"].get("content") or "").strip()


def clean_for_tts(text):
    """去掉 markdown 符号和表情，避免念成“星号星号”。"""
    import re
    text = re.sub(r"[*#`_~>\-]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _synth_once(clause):
    if TTS_MODE == "clone":
        payload = {"model": INDEXTTS, "input": clause,
                   "ref_audio": CLONE_REF, "ref_text": CLONE_TEXT, "lang_code": "z"}
    else:
        payload = {"model": KOKORO, "input": clause,
                   "voice": KOKORO_VOICE, "lang_code": "z"}
    req = urllib.request.Request(TTS_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def _synth_chunks(clause, depth=0):
    """Kokoro 的 broadcast_shapes/IncompleteRead 崩按音频长度确定性触发，重试同串必崩，
    但切短~一半即可绕开(实测深度1全恢复)。崩了二分递归，返回可播放音频块(尽力而为)。"""
    try:
        return [_synth_once(clause)]
    except Exception:
        if TTS_MODE == "clone" or len(clause) < 4 or depth >= 5:
            return []
        mid = len(clause) // 2
        return _synth_chunks(clause[:mid], depth + 1) + _synth_chunks(clause[mid:], depth + 1)


def speak(text):
    # 切句逐句合成：单句崩不拖垮整段；kokoro 崩了二分递归绕开，全崩才回退系统音
    clauses = [c.strip() for c in re.split(r"(?<=[。！？!?；;\n])", text) if c.strip()] or [text]
    for c in clauses:
        chunks = _synth_chunks(c)
        if not chunks:
            print(f"(本句合成失败，改用系统音: {c[:20]})")
            subprocess.run(["say", "-v", "Tingting", "-r", "190", c])
            continue
        for audio in chunks:
            p = os.path.join(tempfile.gettempdir(), "va_reply.mp3")
            with open(p, "wb") as f:
                f.write(audio)
            subprocess.run(["afplay", p])


def get_model_id():
    with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5) as r:
        return json.load(r)["data"][0]["id"]


def main():
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5)
    except Exception:
        print("❌ 本地大模型未启动。先运行: llm start")
        sys.exit(1)

    model_id = get_model_id()
    mode = "你的克隆音" if TTS_MODE == "clone" else f"Kokoro/{KOKORO_VOICE}"
    print(f"🤖 语音助手「小麦」已就绪（声音：{mode}）。按回车说话，输 q 退出。")
    history = [{"role": "system", "content": SYSTEM}]
    tmp = os.path.join(tempfile.gettempdir(), "va_input.wav")

    while True:
        flush_input()
        try:
            cmd = input("\n🎙️  按回车说话（或输 q 退出）…")
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.strip().lower() == "q":
            break

        record(tmp)
        dur = wav_duration(tmp)
        if dur < 0.8:
            print("（没录到声音，请按回车后再说话）")
            continue

        print("✍️  转写中…")
        user = transcribe(tmp)
        cleaned = user.strip().strip("。.，,！!？? ")
        if not cleaned or cleaned.lower() in HALLUCINATIONS or len(cleaned) < 2:
            print("（没听清，再来一次）")
            continue

        print(f"🗣️  你: {user}")
        history.append({"role": "user", "content": user})
        print("🧠 思考中…")
        reply = ask_llm(history, model_id)
        history.append({"role": "assistant", "content": reply})
        print(f"🤖 小麦: {reply}")
        print("🔊 合成语音中…")
        speak(clean_for_tts(reply))

    print("👋 再见")


if __name__ == "__main__":
    main()
