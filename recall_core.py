# -*- coding: utf-8 -*-
"""跨会议检索的平台无关内核 —— 全项目唯一的一份。

**为什么抽出来**:两段式检索(目录选会 → 按预算拼上下文作答)先在 bin/recall
(Mac,读 索引.json)跑通,网页版(服务器,读 sessions/*/meta.json)要同一套语义。
不抽就是再写一遍 —— 与 llm_chain / noise_filter 同一类教训:
重复的知识必然漂移,漂移的防线必然咬人。
由 sync-web 推到服务器根,与 prompts.py 同机制。

留在各端的:目录怎么来(索引.json vs sessions/)、正文怎么读、LLM 怎么调。
收进这里的:目录格式、LLM 选会输出的解析、关键词兜底、预算分配与掐中间截断。
"""
import re

MAX_CONTEXT_CHARS = 24000        # 所有会加起来喂给第二段的上限

_ELLIPSIS = "\n\n〔……中间略去 {n} 字……〕\n\n"


def format_catalog(entries):
    """把目录压成第一段检索用的文本(每场会一行)。"""
    lines = []
    for i, e in enumerate(entries, 1):
        lines.append(
            f"{i}. [{e.get('date') or '日期不明'}] {e.get('title') or e.get('id')}"
            f" —— {e.get('summary') or '(无摘要)'}"
        )
    return "\n".join(lines)


def parse_picks(raw, count):
    """从大脑的回答里抠出编号,转成 0 起的下标。

    容忍各种啰嗦写法(`1,3` / `会议 1 和 3` / `1、3`);`none` 或抠不出 → 空列表。
    """
    text = (raw or "").strip().lower()
    if not text or "none" in text:
        return []
    picks = []
    for tok in re.findall(r"\d+", text):
        n = int(tok)
        if 1 <= n <= count and (n - 1) not in picks:
            picks.append(n - 1)
    return picks


def keyword_fallback(question, entries, limit=2):
    """大脑没挑出东西时的兜底:按标题+摘要里的字符重合度排个序。

    中文没有空格分词,这里按 2-gram 重合算,够用即止 —— 它只是兜底,不是主路径。
    """
    q = "".join(ch for ch in (question or "") if ch.strip())
    grams = {q[i:i + 2] for i in range(len(q) - 1)} or {q}
    scored = []
    for i, e in enumerate(entries):
        hay = f"{e.get('title', '')}{e.get('summary', '')}"
        hits = sum(1 for g in grams if g in hay)
        if hits:
            scored.append((hits, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:limit]]


def trim_middle(text, max_chars):
    """超长时**掐掉中间**,保留开头和结尾。

    Returns:
        (正文, 是否被截断)

    会议的结论、待办、期限几乎总在最后("那就定 Redis""九月九号之前上灰度"),
    而 recall 最常见的问题正是「最后定的是什么」—— 只取开头恰好把最该被问到的
    部分丢掉。掐中间保两头,结尾给足(1/3 预算)。
    """
    if max_chars <= 0:                    # 预算被分光了,这场一个字也放不下
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    marker = _ELLIPSIS.format(n=len(text) - max_chars)
    budget = max_chars - len(marker)
    if budget <= 0:
        return text[:max_chars], True
    head = budget * 2 // 3
    tail = budget - head
    return text[:head] + marker + text[-tail:], True


def split_budget(sizes, total):
    """把总预算分给各场会:先按均分给,短会用不完的额度让给长会。

    这样两场「一短一长」不会让长会被无谓腰斩,而总量仍然守得住。
    """
    n = len(sizes)
    if n == 0:
        return []
    quota = [0] * n
    remaining, active = total, list(range(n))
    while active:
        share = remaining // len(active)
        if share <= 0:
            break
        done = [i for i in active if sizes[i] <= share]
        if not done:
            for i in active:
                quota[i] += share
            remaining -= share * len(active)
            break
        for i in done:
            quota[i] += sizes[i]
            remaining -= sizes[i]
        active = [i for i in active if i not in done]
    return quota


def build_context(entries, picks, load_text, max_chars=MAX_CONTEXT_CHARS):
    """把选中的会拼成第二段的输入。

    Args:
        load_text: ``entry -> str``,各端自己实现正文怎么读
            (Mac 读转写文件,服务器读会话目录里的 记录/纪要)。
        max_chars: **所有会加起来**的上限(曾按每场施加,3 场吐出 72055 字)。

    Returns:
        (上下文文本, 截断说明列表)
    """
    picked = [entries[i] for i in picks if 0 <= i < len(entries)]
    raw = [(e, load_text(e) or "") for e in picked]
    quota = split_budget([len(t) for _, t in raw], max_chars)

    blocks, notes = [], []
    for (e, text), q in zip(raw, quota):
        if not text:
            continue
        body, cut = trim_middle(text, q)
        head = f"### 会议：{e.get('title') or e.get('id')}（{e.get('date') or '日期不明'}）"
        blocks.append(f"{head}\n{body}")
        if cut:
            notes.append(f"{e.get('title') or e.get('id')}（{len(text)} 字，"
                         f"取了首尾共 {q} 字，中间略去）")
    return "\n\n".join(blocks), notes
