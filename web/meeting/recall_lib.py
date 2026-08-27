# -*- coding: utf-8 -*-
"""服务器侧跨会议检索:目录来自 sessions/*/meta.json,检索语义复用 recall_core。

与 bin/recall(Mac 端)的分工:两段式语义(目录格式/选会解析/预算/掐中间截断)
和两个 prompt 都在共享层(recall_core.py / prompts.py,唯一一份);
这里只实现服务器特有的:会话目录怎么变成 catalog、正文从哪读、走网关 LLM。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # voice-svc 根:共享模块在这
sys.path.insert(0, HERE)

from recall_core import (MAX_CONTEXT_CHARS, format_catalog, parse_picks,  # noqa: E402
                         keyword_fallback, build_context)
from prompts import PICK_SYS, ANSWER_SYS  # noqa: E402
import minutes_lib  # noqa: E402  (走网关,自带模型候选链)


def _summary(d):
    """catalog 用的一句话摘要:纪要的「一句话摘要」段,退而求其次取纪要开头。"""
    p = os.path.join(d, "会议纪要.md")
    if not os.path.exists(p):
        return ""
    text = open(p, encoding="utf-8").read()
    m = text.split("## 一句话摘要", 1)
    if len(m) == 2:
        first = m[1].strip().split("\n", 1)[0].strip()
        if first:
            return first[:120]
    return text.strip().replace("\n", " ")[:80]


def load_entries(sessions_dir):
    """扫会话目录出 catalog 条目。只收「有内容可查」的会(纪要/记录/实时字幕任一)。"""
    entries = []
    for name in sorted(os.listdir(sessions_dir), reverse=True):
        d = os.path.join(sessions_dir, name)
        mp = os.path.join(d, "meta.json")
        if not os.path.isfile(mp):
            continue
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:                      # noqa: BLE001  坏 meta 跳过,别拖垮整个检索
            continue
        has_text = any(os.path.exists(os.path.join(d, f))
                       for f in ("会议记录.md", "会议纪要.md", "live.jsonl"))
        if not has_text:
            continue
        import time as _t
        started = meta.get("started") or 0
        entries.append({
            "id": meta.get("id") or name,
            "title": meta.get("title") or name,
            "date": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(started)) if started else "",
            "summary": _summary(d),
            "dir": d,
        })
    return entries


def _load_text(entry):
    """第二段的正文:记录(逐字)优先,其次纪要,再不济拼实时字幕。"""
    d = entry["dir"]
    for name in ("会议记录.md", "会议纪要.md"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    p = os.path.join(d, "live.jsonl")
    if os.path.exists(p):
        lines = []
        for ln in open(p, encoding="utf-8"):
            try:
                o = json.loads(ln)
                lines.append(o.get("zh") or o.get("orig") or "")
            except Exception:                  # noqa: BLE001
                pass
        return "\n".join(x for x in lines if x)
    return ""


def ask_meetings(question, sessions_dir, top=3, log=lambda *a: None):
    """两段式:目录选会 → 按预算拼上下文 → 作答带出处。

    Returns:
        {"answer": str, "sources": [{"id","title","date"}], "notes": [截断说明]}
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    entries = load_entries(sessions_dir)
    if not entries:
        return {"answer": "会议库还是空的 —— 先录一场或上传一份录音。", "sources": [], "notes": []}

    log(f"检索:{len(entries)} 场会里选")
    raw = minutes_lib.ask(PICK_SYS, f"问题：{question}\n\n会议列表：\n{format_catalog(entries)}",
                          max_tokens=60, temperature=0.1)
    picks = parse_picks(raw, len(entries))[:top]
    if not picks:
        picks = keyword_fallback(question, entries, limit=top)
    if not picks:
        return {"answer": "没找到相关的会议。", "sources": [], "notes": []}

    ctx, notes = build_context(entries, picks, load_text=_load_text,
                               max_chars=MAX_CONTEXT_CHARS)
    log(f"作答:{len(picks)} 场会,{len(ctx)} 字上下文")
    answer = minutes_lib.ask(ANSWER_SYS, f"问题：{question}\n\n{ctx}", max_tokens=1200)
    sources = [{k: entries[i][k] for k in ("id", "title", "date")} for i in picks]
    return {"answer": answer, "sources": sources, "notes": notes}
