"""会议 session:目录/元数据/录音封装。每场会议一个目录,内含 meta.json、recording.*、live.jsonl 等。"""
import os, re, time, json, uuid, wave, glob
from config import SESSIONS, SR, ACCESS_PW

_SLUG_RE = re.compile(r"[^\w一-鿿-]+")


def check_pw(pw):
    return (not ACCESS_PW) or pw == ACCESS_PW


def _slug(title):
    s = _SLUG_RE.sub("", (title or "").strip())[:24]
    return s or "会议"


def new_session(scene, lang, title):
    sid = time.strftime("%Y%m%d_%H%M") + "_" + _slug(title) + "_" + uuid.uuid4().hex[:4]
    d = os.path.join(SESSIONS, sid)
    os.makedirs(d, exist_ok=True)
    meta = {"id": sid, "title": (title or "").strip() or time.strftime("%m-%d %H:%M 会议"),
            "scene": scene or "线上", "lang": lang or "auto",
            "started": time.time(), "ended": None, "status": "recording", "duration": 0}
    write_meta(d, meta)
    return meta


def sess_dir(sid):
    """校验 sid 安全且存在,返回目录;非法返回 None。"""
    if not sid:
        return None
    d = os.path.realpath(os.path.join(SESSIONS, sid))
    if not d.startswith(os.path.realpath(SESSIONS) + os.sep):
        return None
    return d if os.path.isdir(d) else None


def read_meta(d):
    try:
        return json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    except Exception:
        return None


def write_meta(d, meta):
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def finalize_wav(d):
    """把流式 recording.pcm 封成可播放的 recording.wav(幂等),封好后删 pcm 省空间。"""
    pcm, wav = os.path.join(d, "recording.pcm"), os.path.join(d, "recording.wav")
    if not os.path.exists(pcm):
        return os.path.exists(wav)
    ch = (read_meta(d) or {}).get("channels", 1)
    w = wave.open(wav, "wb")
    w.setnchannels(ch); w.setsampwidth(2); w.setframerate(SR)
    with open(pcm, "rb") as f:                 # 流式封装,超长会议不一次性进内存
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            w.writeframes(chunk)
    w.close()
    try:
        os.remove(pcm)
    except OSError:
        pass
    return True


def wav_seconds(d):
    """录音的真实秒数(比墙钟时间准:不含开始录制前的等待)。"""
    p = os.path.join(d, "recording.wav")
    if not os.path.exists(p):
        return 0
    try:
        w = wave.open(p)
        n, sr = w.getnframes(), w.getframerate()
        w.close()
        return int(n / sr) if sr else 0
    except Exception:
        return 0


def recording_path(d):
    """当前可用的录音文件:优先 opus/wav(实时录的),否则任意 recording.*(上传的原始文件);无则 None。"""
    for name in ("recording.opus", "recording.wav"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    g = [p for p in glob.glob(os.path.join(d, "recording.*")) if not p.endswith(".pcm")]
    return g[0] if g else None


def dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prune_old_audio(retention_days):
    """删除超过保留期会议的录音文件(保留 meta/字幕/纪要)。返回删除文件数。"""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    n = 0
    for d in glob.glob(os.path.join(SESSIONS, "*")):
        m = read_meta(d)
        if not m or (m.get("started") or 0) >= cutoff:
            continue
        for p in glob.glob(os.path.join(d, "recording.*")) + [os.path.join(d, "audio.wav")]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    n += 1
                except OSError:
                    pass
    return n
