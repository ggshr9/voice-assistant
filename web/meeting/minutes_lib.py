"""合并转写+分人 → 说话人A/B/C 标注稿;调网关 Qwen3.6 出会议纪要 + 会议记录。
LLM 走 litellm 网关(env: CAPTION_LLM_URL / CAPTION_LLM_KEY / CAPTION_LLM_MODEL)。"""
import os, re, json, time, string, urllib.request
from collections import Counter

LLM_URL = os.environ.get("CAPTION_LLM_URL", "http://10.0.0.1:4000/v1/chat/completions")
LLM_KEY = os.environ.get("CAPTION_LLM_KEY", "")
LLM_MODEL = os.environ.get("CAPTION_LLM_MODEL", "Qwen3.6")


def ask(system, user, max_tokens=2600, temperature=0.3):
    body = json.dumps({
        "model": LLM_MODEL, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    h = {"Content-Type": "application/json"}
    if LLM_KEY:
        h["Authorization"] = f"Bearer {LLM_KEY}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(LLM_URL, data=body, headers=h)
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            c = (out["choices"][0]["message"].get("content") or "")
            c = re.sub(r"<think>.*?</think>", "", c, flags=re.S).strip()
            if c:
                return c
        except Exception as e:
            time.sleep(2)
    raise RuntimeError("网关无返回")


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
NOTE_SYS = ("你是会议记录助理。把这段会议转写片段提炼成要点笔记,保留:讨论的话题、各方观点、"
            "做出的决定、提到的待办和负责人、数字与时间。用简洁中文分条列出,不要寒暄。")
FINAL_SYS = ("你是资深会议纪要撰写者。基于提供的会议内容,输出结构化中文纪要 Markdown。严格按结构,"
             "不要寒暄、不要编造:\n\n## 一句话摘要\n\n## 关键决议\n- (没有写「本次无明确决议」)\n\n"
             "## 待办事项\n用表格:| 事项 | 负责人 | 期限 |,负责人是「我」标 **我**,没期限写「—」。\n\n"
             "## 讨论要点\n按话题分小节,各方观点要点列出。\n\n## 风险 / 待澄清\n- (没有就省略本节)\n")


def make_minutes(text, me=None, log=print):
    sysmsg = FINAL_SYS
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


# ---------- 会议记录(忠实还原)----------
RECORD_SYS = (
    "你在整理会议的【逐字记录】,不是摘要。输入有识别错误、噪音字、缺标点,每行以「说话人X：」标注。\n"
    "请尽量【还原】成可读对话:1) 逐行保留每人全部内容,不概括不删减不合并不同人;2) 补标点、修同音字、"
    "删「嘶嘶嘶」类噪音和口水重复;3) 听不清/拿不准处就地加「[?…]」写你的判断或存疑,鼓励多标,不算编造;"
    "4) 不凭空加没出现的句子;5) 严格按「说话人X：内容」逐行输出,不加标题。\n\n示例——\n"
    "输入:\n说话人A：好现在打给他OK嘶嘶嘶嘶\n说话人C：嘶嘶嗯是马内几次你吃\n输出:\n"
    "说话人A：好,现在打给他,OK。\n说话人C：嗯,是。[?听不清,疑为'某地区']你吃?")


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
