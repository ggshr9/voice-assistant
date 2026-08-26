"""实时 STT(faster-whisper / CUDA) + 翻译(litellm 网关)。"""
import os, re, json, asyncio, urllib.request
import numpy as np
from config import MODEL, LLM_URL, LLM_KEY, LLM_MODEL

STT_LOCK = asyncio.Lock()   # 串行化 GPU 转写,避免并发会话同时调 model.transcribe 互相串
# 精确匹配只能挡住那几个固定短语。真实的幻觉是【长句和重复】——
# 实测同一段音频上 whisper 吐出过:
#   auto  → "Honey, honey, honey, honey."          (把「哈喽哈喽」按英文音译)
#   zh    → "优优独播剧场——YoYo Television Series Exclusive"  (训练数据里的字幕污染)
#   还有  → "I'm using a trombone." × 4             (静音上的循环)
# 这些一个都不在下面的集合里。所以除了精确匹配,还要看【是否整句就是幻觉短语】
# 和【是否在原地打转】。
# 整句就是这些 → 幻觉。刻意用【明确列表】而不是长度规则:
# 中文的「好的」「对的」也是两个字,那是真话;英文两个字母才多半是语气填充。
_HALLU = {
    "you", "thank you", "thanks", ".", "", "请订阅", "谢谢观看", "字幕",
    # whisper 在安静段落上最常吐的单词填充,单独成句时没有任何信息量
    "so", "uh", "um", "mm", "hmm", "hm", "oh", "ah", "eh", "er",
    "bye", "okay.", "the", "and",
}

# 出现即判定为幻觉的片段(这些是 whisper 训练集里的字幕/水印污染,真会议里不会说)
_HALLU_SUBSTR = (
    "优优独播剧场", "yoyo television", "请订阅", "谢谢观看", "谢谢大家观看",
    "字幕由", "字幕组", "amara.org", "本字幕", "请不吝点赞", "订阅转发",
    "打赏支持明镜", "明镜与点点栏目",
)

MIN_RMS = 0.004        # ≈-48dBFS。低于此不送模型 —— 静音正是幻觉的温床


def rms(pcm):
    """一段 int16 PCM 的 rms(0~1)。"""
    a = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    return float(np.sqrt((a * a).mean())) if a.size else 0.0


# 判定「在原地打转」的门槛。**刻意定得保守**:
# 误删真话的代价高于显示一行垃圾 —— 垃圾用户可以无视,删掉的话找不回来。
# 血的教训:第一版设成「重复 3 次即判幻觉」,结果把用户真实说的
# "Hello. Hello, hello, hello, hello."(他就是在连着打招呼)整条丢掉了。
# 真正能区分「人在重复」和「模型跑飞」的不是次数,是**重复单元的长度**:
# 人会把一个词连说几遍(「哈喽哈喽哈喽」「对对对对」),但不会把一个十几字的
# 句子一字不差地连说四遍 —— 那只可能是生成循环。
# 第二版把门槛设成「重复 5 次」,结果用户真的说了 5 声 hello,又被误杀了一次。
MIN_REPS = 4          # 连续 4 段完全相同
MIN_UNIT_LEN = 10     # 且单元长到是个「句子」而不是「词」


def is_repetitive(text, min_reps=MIN_REPS, min_unit=MIN_UNIT_LEN):
    """同一片段连续重复 ≥min_reps 次,且片段本身不短 → 模型在打转。

    要抓的是 "I'm using a trombone." × 4 这类跑飞的生成循环,
    不是 "哈喽，哈喽，哈喽" 这种真人真的会说的话。
    """
    t = text.strip()
    if len(t) < min_unit * min_reps:
        return False
    # 整串就是同一个长片段的复读
    for size in range(min_unit, len(t) // min_reps + 1):
        unit = t[:size]
        if unit.strip() and t.startswith(unit * min_reps):
            return True
    # 以标点切开后,连续 min_reps 句完全相同
    parts = [p.strip() for p in re.split(r"[。.!！?？,，;；]+", t) if p.strip()]
    if len(parts) >= min_reps:
        for i in range(len(parts) - min_reps + 1):
            win = parts[i:i + min_reps]
            if len(set(win)) == 1 and len(win[0]) >= min_unit:
                return True
    return False

# 实时字幕的模型。**2026-08-26 从 faster-whisper 换成 Qwen3-ASR** ——
# 同一段真实录音上的实测对照:
#   faster-whisper large-v3 (auto) → "Honey, honey, honey, honey."  (把「哈喽」按英文音译)
#   faster-whisper large-v3 (zh)   → "优优独播剧场——YoYo Television Series Exclusive"
#                                     (训练集里混进的盗版字幕水印,业内有名的幻觉)
#   Qwen3-ASR-1.7B                 → "Hello。哈喽，哈喽，哈喽。嗯。咱们呢是孩子嘛…"  ✅
# 批处理管线(asr_diarize_step.py)本来就用它,模型早在机器上、注释里就写着
# 「中文 SOTA,无 whisper 幻觉」—— 只有实时这条路一直落在 whisper 上。
ASR_MODEL = os.environ.get("CAPTION_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ASR_DEVICE = os.environ.get("CAPTION_ASR_DEVICE", "cuda:1")   # 避开批处理占的 cuda:0

# 语言代码:界面用 zh/en,Qwen3-ASR 要全称
_LANG = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese"}
_LANG_BACK = {v: k for k, v in _LANG.items()}

# 每秒音频允许生成多少 token。**这是延迟护栏,不是质量参数**:
# 实测某个 10 秒片段上模型进入生成循环,跑满 max_new_tokens=256 用了 5.67 秒,
# 而正常片段只要 0.13 秒。把上限按时长给,最坏延迟就跟着时长走而不是失控。
# 中文正常语速约 5 字/秒,给 8 token/秒 是宽松的两倍余量。
TOKENS_PER_SEC = 8
MIN_NEW_TOKENS, MAX_NEW_TOKENS = 24, 256

if os.environ.get("SKIP_MODEL"):           # 测试/导入冒烟用,不占 GPU
    model = None
    print("STT: 跳过模型加载(SKIP_MODEL)", flush=True)
else:
    import torch
    from qwen_asr import Qwen3ASRModel
    print(f"加载 {ASR_MODEL} 到 {ASR_DEVICE} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(ASR_MODEL, dtype=torch.bfloat16,
                                          device_map=ASR_DEVICE,
                                          max_new_tokens=MAX_NEW_TOKENS)
    print("会议工作台后端就绪", flush=True)


def transcribe_pcm(pcm, lang=None):
    """转写一段 PCM。

    Args:
        lang: "zh" / "en" / None(自动)。**这个参数从前不存在** —— 界面上选的语言
            存进了 meta.json 却从没传到这里,live 字幕永远走自动检测。
            实测在安静的中文上自动检测判成 en 的置信度只有 0.51(等于抛硬币),
            于是「哈喽哈喽」被音译成 "Honey, honey"。
    """
    if rms(pcm) < MIN_RMS:                  # 静音不送模型,它会开始编
        return "", lang or ""
    audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    secs = len(audio) / 16000.0
    cap = max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS, int(secs * TOKENS_PER_SEC)))
    name = _LANG.get(lang or "", None)
    # transcribe() 的签名里没有 max_new_tokens —— 它读的是实例属性 self.max_new_tokens
    # (qwen3_asr.py:379)。第一版按参数传,静默走了 TypeError 兜底,上限从没生效,
    # 那个 5.8 秒的生成循环也就一直在。
    model.max_new_tokens = cap
    r = model.transcribe(audio=(audio, 16000), language=name)
    if not r:
        return "", lang or ""
    item = r[0]
    text = (getattr(item, "text", "") or "").strip()
    # 模型能自己报语种;报不出就退回调用方指定的那个
    got = getattr(item, "language", None) or name
    return text, _LANG_BACK.get(got, lang or "")


def is_noise(text):
    c = text.strip().strip("。.,，!！?？ ").lower()
    if (not c) or len(c) < 2 or c in _HALLU:
        return True
    if any(k in c for k in _HALLU_SUBSTR):
        return True
    return is_repetitive(c)


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
