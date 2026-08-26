#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""meetings_mcp.py — 把本地会议档案暴露成 MCP server，让 agent 直接问过去的会。

设计取向（跟 wxvault_mcp.py 一致，少即是多）：
  - 不依赖 mcp SDK。直接用 stdlib 讲 JSON-RPC 2.0 over stdio，协议就几十行、可逐行审，
    任何 MCP 客户端（含 Claude Code）都能连。
  - 零外部依赖：只读 `~/会议录音/索引.json` 和转写文件，复用 `_index.py` / `recall`。
  - **全部同步秒回**。转写/纪要是 `meeting` 和 `minutes` 离线跑完的，MCP 只读成品，
    所以不需要异步 job + 轮询那套。唯一慢的是 `ask_meetings`（要过一遍本地大脑）。

暴露 4 个工具（够用即止）：
  list_meetings(query?, limit?)   —— 列会议（倒序；query 按标题/摘要过滤）
  get_meeting(id, part?)          —— 取某场会的纪要 / 逐字记录 / 转写原文
  search_meetings(query, limit?)  —— 字面检索：在转写全文里找关键词，返回命中上下文
  ask_meetings(question, top?)    —— 自然语言问答（两段式，同 recall），要本地大脑在跑

search 和 ask 是互补的，别只留一个：
  search 精确、秒回、不需要大脑，适合「谁提过某个专有名词」这种找原话；
  ask 要过大脑、十几秒，适合「当时结论是什么」这种需要读懂再归纳的问题。

注册（Claude Code）：
  claude mcp add meetings -- python3 ~/voice-assistant/bin/meetings_mcp.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

SERVER_NAME = "meetings"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

MAX_SNIPPET = 3000        # get_meeting 单次最多回多少字，超了截断并说明
CONTEXT_CHARS = 120       # search 命中处前后各留多少字


def _index():
    import _index as mod
    return mod


def _load_entries():
    entries = _index().load()
    if not entries:
        raise RuntimeError(
            "会议索引是空的。先跑一次 `_index.py rebuild`（或跑一次 minutes 自动登记）。")
    return entries


def _find(entries, meeting_id):
    for e in entries:
        if e.get("id") == meeting_id:
            return e
    # 允许用标题/日期片段模糊定位，agent 常常只记得个大概
    hits = [e for e in entries
            if meeting_id in (e.get("title") or "") or meeting_id in (e.get("date") or "")]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError("「%s」匹配到多场会：%s" % (
            meeting_id, "、".join(e["id"] for e in hits)))
    raise ValueError("找不到会议：%s（用 list_meetings 看有哪些）" % meeting_id)


def _brief(e):
    """索引条目压成给 agent 看的摘要，去掉内部字段。"""
    return {
        "id": e.get("id"),
        "title": e.get("title"),
        "date": e.get("date"),
        "duration": _index().format_duration(e.get("duration_sec")),
        "speakers": e.get("speakers") or None,
        "todos": e.get("todos", 0),
        "my_todos": e.get("mine", 0),
        "summary": e.get("summary"),
    }


def _read_part(entry, part):
    """读一场会的某个部分。返回 (正文, 文件名)。"""
    files = entry.get("files") or {}
    key = {"minutes": "minutes", "纪要": "minutes",
           "record": "record", "记录": "record",
           "transcript": "transcript", "转写": "transcript"}.get(part or "minutes")
    if key is None:
        raise ValueError("part 只能是 minutes / record / transcript，收到：%s" % part)
    name = files.get(key)
    if not name:
        raise ValueError("这场会没有「%s」（有的部分：%s）" % (
            part, "、".join(k for k, v in files.items() if v) or "无"))
    path = os.path.join(entry.get("dir", ""), name)
    if not os.path.exists(path):
        raise ValueError("文件不在了：%s" % path)
    with open(path, encoding="utf-8") as f:
        return f.read(), name


# ---------- 四个工具 ----------
def list_meetings(query="", limit=50):
    entries = _load_entries()
    q = (query or "").strip()
    if q:
        entries = [e for e in entries
                   if q in (e.get("title") or "") or q in (e.get("summary") or "")]
    shown = entries[:limit] if limit else []
    # count 从前给的是【匹配总数】,而 meetings 只有 limit 条 —— limit=0 时
    # 返回 count:4 / meetings:[] 自相矛盾。现在 count 就是本次返回的条数。
    return {"count": len(shown), "total": len(entries),
            "truncated": len(shown) < len(entries),
            "meetings": [_brief(e) for e in shown]}


def get_meeting(meeting_id, part="minutes", max_chars=MAX_SNIPPET):
    entry = _find(_load_entries(), meeting_id)
    text, name = _read_part(entry, part)
    truncated = len(text) > max_chars
    return {
        "meeting": _brief(entry),
        "part": part,
        "file": name,
        "truncated": truncated,
        "total_chars": len(text),
        "text": text[:max_chars] if truncated else text,
    }


def search_meetings(query, limit=20):
    """在所有会的转写全文里找关键词，返回命中处的上下文。

    字面检索，不过大脑 —— 用于「谁提过 X」这种要原话的问题。
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query 不能为空")
    hits = []
    for e in _load_entries():
        try:
            text, name = _read_part(e, "transcript")
        except ValueError:
            continue
        for m in re.finditer(re.escape(q), text, re.I):
            if len(hits) >= limit:
                break
            s = max(0, m.start() - CONTEXT_CHARS)
            t = min(len(text), m.end() + CONTEXT_CHARS)
            hits.append({
                "meeting_id": e.get("id"),
                "title": e.get("title"),
                "date": e.get("date"),
                "file": name,
                "context": ("…" if s else "") + text[s:t] + ("…" if t < len(text) else ""),
            })
        if len(hits) >= limit:
            break
    return {"query": q, "count": len(hits), "hits": hits}


def ask_meetings(question, top=2):
    """自然语言问答：复用 recall 的两段式检索。需要本地大脑在跑。"""
    q = (question or "").strip()
    if not q:
        raise ValueError("question 不能为空")
    import importlib.util
    from importlib.machinery import SourceFileLoader

    here = os.path.dirname(os.path.realpath(__file__))
    spec = importlib.util.spec_from_loader(
        "recall", SourceFileLoader("recall", os.path.join(here, "recall")))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    spec_m = importlib.util.spec_from_loader(
        "_minutes", SourceFileLoader("_minutes", os.path.join(here, "minutes")))
    mod = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(mod)

    if not mod.health_ok():
        raise RuntimeError("本地大脑没起来。先在终端跑 `llm start`（8bit 权重加载约 30-60 秒）。")

    entries = _load_entries()
    raw = mod.ask(rc.PICK_SYS,
                  "问题：%s\n\n会议列表：\n%s" % (q, rc.format_catalog(entries)),
                  max_tokens=60, temperature=0.1)
    picks = rc.parse_picks(raw, len(entries))[:top] or rc.keyword_fallback(q, entries, limit=top)
    if not picks:
        return {"question": q, "answer": "", "sources": [],
                "note": "索引里没有与问题相关的会议。"}

    context, notes = rc.build_context(entries, picks)
    if not context:
        return {"question": q, "answer": "", "sources": [_brief(entries[i]) for i in picks],
                "note": "选中的会议没有可读的转写文本。"}
    answer = mod.ask(rc.ANSWER_SYS, "问题：%s\n\n会议内容：\n\n%s" % (q, context),
                     max_tokens=1200, temperature=0.3)
    out = {"question": q, "answer": answer,
           "sources": [_brief(entries[i]) for i in picks]}
    if notes:
        out["note"] = "；".join(notes)
    return out


TOOLS = [
    {
        "name": "list_meetings",
        "description": "列出本地会议档案（按时间倒序）。可用 query 按标题/摘要过滤。"
                       "先用它看有哪些会，再用 get_meeting 取全文。",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "可选：标题或摘要里的关键词"},
            "limit": {"type": "integer", "description": "最多返回几场，默认 50"},
        }},
    },
    {
        "name": "get_meeting",
        "description": "取某场会的内容。part=minutes 结构化纪要（默认）/ record 逐字记录 / "
                       "transcript 转写原文。id 可以用 list_meetings 给的 id，也可以是标题或日期片段。",
        "inputSchema": {"type": "object", "properties": {
            "id": {"type": "string", "description": "会议 id、标题片段或日期"},
            "part": {"type": "string", "enum": ["minutes", "record", "transcript"],
                     "description": "要哪个部分，默认 minutes"},
            "max_chars": {"type": "integer", "description": "最多返回多少字，默认 3000"},
        }, "required": ["id"]},
    },
    {
        "name": "search_meetings",
        "description": "在所有会的转写全文里做字面检索，返回命中处上下文。秒回、不需要本地大脑。"
                       "适合「谁提过某个专有名词」这种要找原话的问题；"
                       "要归纳、要结论请用 ask_meetings。",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "关键词"},
            "limit": {"type": "integer", "description": "最多返回几条命中，默认 20"},
        }, "required": ["query"]},
    },
    {
        "name": "ask_meetings",
        "description": "用自然语言问过去的会议，两段式检索后由本地大脑作答并标出处。"
                       "适合「上次联调的结论是什么」这类需要读懂再归纳的问题。"
                       "**需要本地大脑在跑（llm start），一次约十几秒**；只要找原话就用 search_meetings。",
        "inputSchema": {"type": "object", "properties": {
            "question": {"type": "string", "description": "你的问题"},
            "top": {"type": "integer", "description": "最多取几场会作答，默认 2"},
        }, "required": ["question"]},
    },
]


# 调这个接口的是 **agent**,不是人。报错必须指名道姓说清「哪个参数、要什么」,
# 否则模型只能瞎猜。从前缺 id 会原样抛出 KeyError,前端看到的是「错误: \'id\'」;
# 传个数字当 query 则是「\'int\' object has no attribute \'strip\'」—— 全是泄漏的内部异常。
MAX_LIMIT = 500


def _int(v, default, name="limit", lo=0, hi=MAX_LIMIT):
    """可选整型参数;缺省或 null 回落默认值(客户端常发 "limit": null)。

    越界一律**夹紧**而不是报错 —— 对 agent 来说,把 limit=99999 直接夹到 500
    比让整个调用失败有用得多。但类型错了要明确报,那是调用方写错了。
    """
    if v is None or v == "":
        return default
    if isinstance(v, bool):                       # bool 是 int 的子类,但 limit=true 显然是笔误
        raise ValueError("%s 需要整数,收到布尔值 %r" % (name, v))
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError("%s 需要整数,收到 %r（示例：\"%s\": 20）" % (name, v, name))
    return max(lo, min(hi, n))


def _text(args, name, tool):
    """必填的字符串参数。"""
    if name not in args:
        raise ValueError("缺少必填参数 %s。%s 的调用示例：{\"%s\": \"...\"}" % (name, tool, name))
    v = args[name]
    if v is None:
        raise ValueError("%s 不能为 null" % name)
    if not isinstance(v, str):
        raise ValueError("%s 需要字符串,收到 %s（%r）" % (name, type(v).__name__, v))
    if not v.strip():
        raise ValueError("%s 不能为空" % name)
    return v


def call_tool(name, args):
    if not isinstance(args, dict):
        raise ValueError("arguments 必须是对象,收到 %s" % type(args).__name__)
    if name == "list_meetings":
        q = args.get("query") or ""
        if not isinstance(q, str):
            raise ValueError("query 需要字符串,收到 %s" % type(q).__name__)
        return list_meetings(q, _int(args.get("limit"), 50))
    if name == "get_meeting":
        part = args.get("part") or "minutes"
        return get_meeting(_text(args, "id", "get_meeting"), part,
                           _int(args.get("max_chars"), MAX_SNIPPET,
                                "max_chars", lo=1, hi=1_000_000))
    if name == "search_meetings":
        return search_meetings(_text(args, "query", "search_meetings"),
                               _int(args.get("limit"), 20))
    if name == "ask_meetings":
        return ask_meetings(_text(args, "question", "ask_meetings"),
                            _int(args.get("top"), 2, "top", lo=1, hi=20))
    raise ValueError("没有名为 %s 的工具（可用：list_meetings / get_meeting / "
                     "search_meetings / ask_meetings）" % name)


# ---------- JSON-RPC 2.0 over stdio ----------
def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def handle(req):
    if not isinstance(req, dict):   # JSON 数组/标量也是合法 JSON，但不是 JSON-RPC 请求
        return _err(None, -32600, "Invalid Request: expected an object")
    mid = req.get("id")
    method = req.get("method")
    if method == "initialize":
        pv = (req.get("params") or {}).get("protocolVersion") or PROTOCOL_VERSION
        return _ok(mid, {"protocolVersion": pv,
                         "capabilities": {"tools": {}},
                         "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}})
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return _ok(mid, {})
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        p = req.get("params") or {}
        if not isinstance(p, dict):   # params 传成数组时,p.get 会漏出 AttributeError
            return _err(mid, -32602, "params 必须是对象,收到 %s" % type(p).__name__)
        try:
            result = call_tool(p.get("name"), p.get("arguments") or {})
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:
            return _ok(mid, {"content": [{"type": "text", "text": "错误: %s" % e}],
                             "isError": True})
    if mid is not None:
        return _err(mid, -32601, "method not found: %s" % method)
    return None


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        n = len(_index().load())
    except Exception as e:
        sys.stderr.write("[meetings] 索引读不了：%s\n" % e)
        n = 0
    sys.stderr.write("[meetings] 就绪 · 索引里 %d 场会\n" % n)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = _err(None, -32700, "Parse error")   # 按 JSON-RPC 2.0 回错误而非静默丢弃
        else:
            try:
                resp = handle(req)
            except Exception as e:                     # 兜底：任何请求都不该拖垮 stdio 循环
                resp = _err(req.get("id") if isinstance(req, dict) else None,
                            -32603, "Internal error: %s" % e)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
