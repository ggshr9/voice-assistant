"""实时 STT(faster-whisper / CUDA) + 翻译(litellm 网关)。"""
import os, re, sys, json, asyncio, urllib.request
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
# 幻觉判定与候选链语义在仓库根的共享模块(sync-web 推到服务器根,与 prompts.py 同机制)。
# 曾各写一遍导致两端防线各缺一半、翻译死了一周(改一处漏两处)。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from noise_filter import is_noise, is_repetitive  # noqa: F401,E402
from llm_chain import parse_chain, try_next_model  # noqa: E402

_try_next = try_next_model                       # 兼容旧名(内部引用与测试)


def model_chain(spec=None):
    """薄壳:链解析语义在 llm_chain.parse_chain(唯一一份)。"""
    return parse_chain(LLM_MODEL if spec is None else spec)


MIN_RMS = 0.004        # ≈-48dBFS。低于此不送模型 —— 静音正是幻觉的温床


def rms(pcm):
    """一段 int16 PCM 的 rms(0~1)。能量门是音频层的事,留在这里,不进 noise_filter。"""
    a = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    return float(np.sqrt((a * a).mean())) if a.size else 0.0



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


_TRANS_SYS = ("你是同声传译。把这句话翻成简洁、口语化的简体中文,"
              "只输出译文本身,不要解释、不要原文、不要引号。")


def _headers():
    h = {"Content-Type": "application/json"}
    if LLM_KEY:
        h["Authorization"] = f"Bearer {LLM_KEY}"
    return h


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
