"""实时 STT(faster-whisper / CUDA) + 翻译(litellm 网关)。"""
import os, re, json, asyncio, urllib.request
import numpy as np
from config import LLM_URL, LLM_KEY, LLM_MODEL   # MODEL 已不再用:改 Qwen3-ASR 了

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
# 模型私有的知识(加载方式、调用怪癖、语言映射)全在 asr_backends.py ——
# 换模型只动那一个文件,这里保持模型无关:能量门、幻觉过滤、(pcm,lang)->(text,lang) 契约。
# 选择后端: CAPTION_ASR_BACKEND=qwen3|whisper  (默认 qwen3)
if os.environ.get("SKIP_MODEL"):           # 测试/导入冒烟用,不占 GPU
    model = None
    print("STT: 跳过模型加载(SKIP_MODEL)", flush=True)
else:
    from asr_backends import load_backend
    model = load_backend()
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
    return model.transcribe(audio, lang)


def is_noise(text):
    c = text.strip().strip("。.,，!！?？ ").lower()
    if (not c) or len(c) < 2 or c in _HALLU:
        return True
    if any(k in c for k in _HALLU_SUBSTR):
        return True
    return is_repetitive(c)


_TRANS_SYS = ("你是同声传译。把这句话翻成简洁、口语化的简体中文,"
              "只输出译文本身,不要解释、不要原文、不要引号。")


def model_chain(spec=None):
    """CAPTION_LLM_MODEL 支持逗号分隔的候选链,前面的挂了自动退到后面。

    **这里曾经是坏的**:minutes_lib 和 caption_core 都加过 fallback,唯独
    实时字幕的 translate 还是单点 —— 于是配置改成 "Qwen3.6,qwen3.5-...,DeepSeek"
    之后,它把整串当成一个模型名发出去,网关回 404,外语字幕的中文翻译全线不工作。
    改一处漏两处比不改更糟:配置看起来是对的,功能却是死的。
    """
    out = []
    for m in ((LLM_MODEL if spec is None else spec) or "").split(","):
        m = m.strip()
        if m and m not in out:
            out.append(m)
    return out or [""]


def _headers():
    h = {"Content-Type": "application/json"}
    if LLM_KEY:
        h["Authorization"] = f"Bearer {LLM_KEY}"
    return h


def _try_next(e):
    """这个错误值不值得换下一个候选模型。

    分界线是**错在模型上还是错在网关上**:
      404 模型下线、429 限流、5xx 后端故障 —— 都是【这个模型】的问题,换一个很可能就好
      连不上/DNS/超时                    —— 是【网关本身】的问题,再试几个只是白等
    第一版只认 404,结果实测撞上 429 就整个放弃了,而换个模型立刻能用。
    """
    code = getattr(e, "code", None)
    if code is not None:
        return code in (404, 408, 409, 429) or code >= 500
    return False           # 没有 HTTP 状态码 = 连接层面的问题,别再耗时间


def translate(text, lang):
    if lang == "zh" or not text:
        return text
    errs = []
    for model in model_chain():
        body = json.dumps({
            "model": model, "max_tokens": 200, "temperature": 0.2,
            "messages": [{"role": "system", "content": _TRANS_SYS},
                         {"role": "user", "content": text}],
        }).encode()
        try:
            req = urllib.request.Request(LLM_URL, data=body, headers=_headers())
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            if "choices" not in out:               # 网关的错误体也可能是 200
                errs.append(f"{model}: {str(out.get('error', out))[:100]}")
                continue
            zh = (out["choices"][0]["message"].get("content") or "").strip()
            if zh:
                return zh
            errs.append(f"{model}: 空译文")
        except Exception as e:                     # noqa: BLE001
            errs.append(f"{model}: {type(e).__name__} {str(e)[:80]}")
            if not _try_next(e):
                break                              # 不是"模型没了"就别把候选挨个试一遍
    print(f"翻译失败: {' | '.join(errs[-3:])}", flush=True)
    return ""                                      # 字幕宁可只显示原文,也别整条消失


def translate_stream(text, lang):
    """流式翻译:逐块 yield 中文增量(打字机效果)。lang==zh 或空则直接 yield 原文。"""
    if lang == "zh" or not text:
        yield text
        return
    # 流式这条也要走候选链。**只在第一个 token 到达之前**允许换模型 ——
    # 已经吐出半句再切换,前端会看到两段拼接的乱译文。
    model = None
    r = None
    errs = []
    for cand in model_chain():
        body = json.dumps({
            "model": cand, "max_tokens": 200, "temperature": 0.2, "stream": True,
            "messages": [{"role": "system", "content": _TRANS_SYS},
                         {"role": "user", "content": text}],
        }).encode()
        try:
            req = urllib.request.Request(LLM_URL, data=body, headers=_headers())
            r = urllib.request.urlopen(req, timeout=60)
            model = cand
            break
        except Exception as e:                     # noqa: BLE001
            errs.append(f"{cand}: {type(e).__name__} {str(e)[:80]}")
            if not _try_next(e):
                break
    if r is None:
        print(f"流式翻译失败: {' | '.join(errs[-3:])}", flush=True)
        return
    got = False
    with r:
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
