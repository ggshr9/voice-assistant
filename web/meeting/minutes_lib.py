"""合并转写+分人 → 说话人A/B/C 标注稿;调网关 Qwen3.6 出会议纪要 + 会议记录。
LLM 走 litellm 网关(env: CAPTION_LLM_URL / CAPTION_LLM_KEY / CAPTION_LLM_MODEL)。"""
import os, re, json, sys, time, string, urllib.request

# 纪要 prompt 与本机 CLI 共用同一份,别在这里另存 —— 两边曾各存各的,实测已漂移。
# 真源在仓库根 prompts.py,由 sync-web 推到 ~/voice-svc/prompts.py。
# dirname x3: web/meeting -> web -> voice-svc(本机则是仓库根),两边同一份代码都成立。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from prompts import NOTE_SYS, FINAL_SYS, RECORD_SYS, ENHANCE_SYS, me_instruction, template_sys
from llm_chain import parse_chain  # noqa: E402  链解析语义唯一一份
from collections import Counter

LLM_URL = os.environ.get("CAPTION_LLM_URL", "")   # 必须显式配置(线上由 secrets.env 注入)
LLM_KEY = os.environ.get("CAPTION_LLM_KEY", "")
# 可以写成逗号分隔的多个,前面的挂了自动退到后面。
# **为什么需要**:网关上的模型会被下线 —— 实测配置里的 Qwen3.6 已在 /models 列表里
# 却返回 404("Received Model Group=Qwen3.6"),整个纪要功能因此静默失效,
# 而错误只被报成一句"网关无返回",查不出是模型没了。
LLM_MODEL = os.environ.get("CAPTION_LLM_MODEL", "Qwen3.6")


def model_chain(spec=None):
    """薄壳:链解析语义在 llm_chain.parse_chain(唯一一份)。"""
    return parse_chain(LLM_MODEL if spec is None else spec)


def _call(model, system, user, max_tokens, temperature):
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    h = {"Content-Type": "application/json"}
    if LLM_KEY:
        h["Authorization"] = f"Bearer {LLM_KEY}"
    req = urllib.request.Request(LLM_URL, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    if "choices" not in out:                     # 网关的错误体也是 200,得自己看
        raise RuntimeError(json.dumps(out.get("error", out), ensure_ascii=False)[:300])
    c = (out["choices"][0]["message"].get("content") or "")
    return re.sub(r"<think>.*?</think>", "", c, flags=re.S).strip()


def ask(system, user, max_tokens=2600, temperature=0.3):
    """按候选模型依次尝试,每个重试 3 次。全挂时把每个模型的真实死因都带出来。"""
    errs = []
    for model in model_chain() or [LLM_MODEL]:
        for attempt in range(3):
            try:
                c = _call(model, system, user, max_tokens, temperature)
                if c:
                    return c
                errs.append(f"{model}: 返回空内容")
            except Exception as e:               # noqa: BLE001
                errs.append(f"{model}: {type(e).__name__} {str(e)[:160]}")
                if "404" in str(e) or "not found" in str(e).lower():
                    break                        # 模型没了,重试没意义,直接换下一个
            time.sleep(2)
    raise RuntimeError("网关无可用模型 —— " + " | ".join(errs[-4:]))


# ---------- 合并转写+分人 ----------
def _assign(seg, turns):
    best, bestov = None, 0.0
    mid = (seg["start"] + seg["end"]) / 2
    nearest, ndist = None, 1e9
    for t in turns:
        ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
        if ov > bestov:
            bestov, best = ov, t["speaker"]
        d = min(abs(mid - t["start"]), abs(mid - t["end"]))   # 无重叠时取时间最近的轮次
        if d < ndist:
            ndist, nearest = d, t["speaker"]
    return best or nearest          # 不再产生幽灵 SPEAKER_?


_HALLU = ["点赞", "订阅", "转发", "打赏", "明镜", "点点栏目", "字幕志愿者", "李宗盛",
          "感谢观看", "谢谢观看", "请不吝", "关注我", "下期再见", "Amara", "字幕组"]


def _is_hallu(text):
    """whisper 中文幻觉行(点赞订阅那类)/ 单字重复退化 → True 丢弃。"""
    t = re.sub(r"[，。！？、\s,.!?]", "", text)
    if len(t) < 2:
        return True
    if sum(p in text for p in _HALLU) >= 2:
        return True
    if len(t) >= 6 and Counter(t).most_common(1)[0][1] >= len(t) * 0.6:
        return True
    return False


def merge(segments, turns):
    """每段按时间重叠归到说话人,连续同人合并;过滤 whisper 幻觉;说话人按出现顺序映射 A/B/C。返回(标注文本, 人数)。"""
    rows = []
    for s in segments:
        if _is_hallu(s["text"]):
            continue
        spk = _assign(s, turns) or "SPEAKER_?"
        if rows and rows[-1][0] == spk:
            rows[-1][1] += s["text"]
        else:
            rows.append([spk, s["text"]])
    order = {}
    for spk, _ in rows:
        if spk not in order:
            i = len(order)
            order[spk] = f"说话人{string.ascii_uppercase[i]}" if i < 26 else f"说话人{i + 1}"
    lines = [f"{order[spk]}：{txt.strip()}" for spk, txt in rows if txt.strip()]
    return "\n".join(lines), len(order)


# ---------- 分块 ----------
def split_chunks(text, size):
    if len(text) <= size:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > size and cur:
            parts.append(cur); cur = ""
        cur += line + "\n"
        while len(cur) > size:
            parts.append(cur[:size]); cur = cur[size:]
    if cur.strip():
        parts.append(cur)
    return parts


# ---------- 会议纪要 ----------


def make_minutes(text, me=None, log=print, template=None):
    sysmsg = template_sys(template)          # 不认识的模板落回默认,老 meta 不崩
    if me:
        sysmsg += f"\n\n**重要**:「{me}」就是「我」本人,TA 名下待办负责人写 **我**。"
    chunks = split_chunks(text, 12000)
    if len(chunks) == 1:
        log("纪要:单次生成"); return ask(sysmsg, "会议转写如下:\n\n" + text, 2600)
    log(f"纪要:分 {len(chunks)} 块")
    notes = []
    for i, c in enumerate(chunks, 1):
        log(f"纪要块 {i}/{len(chunks)}")
        notes.append(f"【片段{i}】\n" + ask(NOTE_SYS, c, 1200))
    return ask(sysmsg, "以下是按时间顺序的要点笔记,汇总成完整纪要:\n\n" + "\n\n".join(notes), 2600)


def make_enhanced(notes, text, log=print):
    """增强笔记:用户手记为骨架,转写为血肉。

    与 make_minutes 是两种产物 —— 那个是 AI 从零写的纪要,这个是**用户自己的笔记**
    被补全后的样子(Granola 式)。铁律在 ENHANCE_SYS 里:不改用户的结构和措辞。
    转写太长时只压转写、不动笔记:笔记是主角,一个字都不能丢。
    """
    notes = (notes or "").strip()
    if not notes:
        return ""
    ctx = text
    if len(ctx) > 12000:                      # 超长会议:先把转写压成要点当上下文
        log("增强笔记:转写过长,先压缩")
        chunks = split_chunks(ctx, 12000)
        ctx = "\n\n".join(ask(NOTE_SYS, c, 1200) for c in chunks)
    log("增强笔记:合并手记与转写")
    return ask(ENHANCE_SYS,
               f"【用户手记】\n{notes}\n\n【会议转写】\n{ctx}", 3000)


# ---------- 会议记录(忠实还原)----------


def _denoise(text):
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^(说话人[A-Z0-9]+：)(.*)$", ln)
        prefix, content = (m.group(1), m.group(2)) if m else ("", ln)
        content = re.sub(r"(.)\1{2,}", r"\1", content)
        if re.sub(r"[，。！？、\s,.!?]", "", content):
            out.append(prefix + content)
    return "\n".join(out)


def make_record(text, log=print):
    text = _denoise(text)
    chunks = split_chunks(text, 3000)
    parts = []
    for i, c in enumerate(chunks, 1):
        log(f"记录块 {i}/{len(chunks)}")
        parts.append(ask(RECORD_SYS, c, 5000).strip())
    return "\n".join(parts)
