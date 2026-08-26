#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""会议索引 —— 让「上个月聊接口联调的是哪场会」有地方可查。

在此之前找一场会只能肉眼翻 `~/会议录音/` 下一堆时间戳目录。索引把每场会压成
一行(标题/日期/时长/人数/待办数/一句话摘要/路径),`recall` 的第一段检索就读它。

  ~/会议录音/索引.json   机器读(recall 与将来的 MCP 直接吃这个)
  ~/会议录音/索引.md     人读(Finder 里点开就能扫)

json 是唯一真源,md 每次由它渲染。

用法:
  _index.py upsert <转写目录> [--title 标题]   # 登记/更新一场会
  _index.py rebuild [会议根目录]               # 扫全部 转写_* 重建
  _index.py list                               # 打印索引表
"""
import json
import os
import re
import sys
from datetime import datetime

ROOT = os.path.expanduser("~/会议录音")
JSON_NAME = "索引.json"
MD_NAME = "索引.md"

_SENT_END = "。！？!?；;\n"


def parse_minutes(md):
    """从纪要 Markdown 里抽出摘要与待办统计。

    Args:
        md: 纪要文件全文。

    Returns:
        dict: ``summary``(一句话摘要)、``todos``(待办条数)、``mine``(其中标了「我」的)。
    """
    if md.startswith("---\n"):              # 有 frontmatter 就先剥掉，别把它当正文
        end = md.find("\n---", 4)
        if end >= 0:
            md = md[end + 4:]
    summary = ""
    m = re.search(r"##\s*一句话摘要\s*\n+(.+?)(?=\n\s*##|\Z)", md, re.S)
    if m:
        summary = " ".join(m.group(1).split()).strip()

    todos = mine = 0
    t = re.search(r"##\s*待办事项\s*\n(.*?)(?=\n\s*##|\Z)", md, re.S)
    if t:
        for line in t.group(1).splitlines():
            line = line.strip()
            if not line.startswith("|") or set(line) <= set("|: -"):
                continue
            if re.search(r"\|\s*事项\s*\|", line):        # 表头
                continue
            todos += 1
            if "**我**" in line:
                mine += 1
    return {"summary": summary, "todos": todos, "mine": mine}


def derive_title(summary, limit=24, fallback=""):
    """把一句话摘要压成便于扫读的短标题。摘要为空则回退到会议标识。"""
    s = (summary or "").strip()
    if not s:
        return fallback
    for i, ch in enumerate(s):
        if ch in _SENT_END or (ch in "，," and i >= limit - 4):
            s = s[:i]
            break
    s = s.strip("，,。 ")
    return s[:limit]


TITLE_SYS = ("你在给会议起标题。读下面的会议摘要，输出一个 8-14 字的中文标题，"
             "点明这场会真正在谈什么。只输出标题本身：不要引号、书名号、句号，"
             "不要写「会议」「讨论」这类废话开头。")


def llm_title(summary, ask, limit=24, fallback=""):
    """用本地大脑给会议起标题;失败或返回空就退回截断摘要。

    Args:
        summary: 一句话摘要。
        ask: ``ask(system, user) -> str``,注入以便测试。
        limit: 标题上限字数。
        fallback: 连摘要都没有时用的兜底。

    Returns:
        标题字符串。**任何异常都被吞掉** —— 起不出标题不该让整场会进不了索引。
    """
    plain = derive_title(summary, limit=limit, fallback=fallback)
    if not (summary or "").strip():
        return plain
    try:
        raw = ask(TITLE_SYS, (summary or "").strip()) or ""
    except Exception:
        return plain
    title = raw.strip().splitlines()[0] if raw.strip() else ""
    title = title.strip().strip("「」《》\"'' 。.·-—:：")
    return title[:limit] if title else plain


def parse_date(meeting_id):
    """从 `线上会议_20260619_2301` 这样的名字里认出 `2026-06-19 23:01`。"""
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})", meeting_id or "")
    if not m:
        return ""
    y, mo, d, hh, mm = m.groups()
    return f"{y}-{mo}-{d} {hh}:{mm}"


def format_duration(seconds):
    """按媒体时长惯例:`0:14` / `2:05` / `1:48:03`。0 或缺失显示 `—`。

    别用「时:分」——一场 14 秒的录音会显示成 `0:00`,看着像没录上。
    """
    sec = int(seconds or 0)
    if sec <= 0:
        return "—"
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_FM_FIELDS = [
    ("id", "id"), ("title", "title"), ("date", "date"),
    ("duration", None), ("speakers", "speakers"),
    ("todos", "todos"), ("my_todos", "mine"), ("summary", "summary"),
]


def _yaml_value(v):
    """YAML 标量：含冒号/引号/井号时必须加引号，否则解析会断在中间。"""
    s = str(v)
    if any(c in s for c in ':#"\'\n') or s.strip() != s:
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


def render_frontmatter(entry):
    """把索引条目渲染成 YAML frontmatter，贴在纪要 .md 最前面。

    **为什么要有**：结构化信息原本只在 索引.json 里，同一份事实存两处会不同步 ——
    手工改了纪要、或把 .md 拷到别处，元数据就对不上了。带上 frontmatter 之后
    单个文件自己就说得清自己是什么，json 退回成"检索用的聚合索引"。

    空值不输出：`speakers: 0` 会让人以为真的 0 个人说话，不如不写。
    """
    lines = ["---"]
    for key, src in _FM_FIELDS:
        v = format_duration(entry.get("duration_sec")) if key == "duration" else entry.get(src)
        if v in (None, "", 0, "—"):
            continue
        lines.append(f"{key}: {_yaml_value(v)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def parse_frontmatter(text):
    """从 .md 里读回 frontmatter；没有就返回空 dict。值一律当字符串。"""
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    out = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            v = v[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        out[k.strip()] = v
    return out


def upsert(entries, entry):
    """按 id 覆盖式写入,再按日期倒序排(没日期的沉底)。"""
    out = [e for e in entries if e.get("id") != entry.get("id")]
    out.append(entry)
    out.sort(key=lambda e: (e.get("date") or "", e.get("id") or ""), reverse=True)
    return out


def render_markdown(entries):
    """渲染人读的索引表。"""
    head = "# 会议索引\n\n> 由 `_index.py` 自动生成，勿手改；真源是 `索引.json`。\n\n"
    if not entries:
        return head + "暂无会议记录。\n"
    rows = ["| 日期 | 标题 | 时长 | 人数 | 待办(我的) | 摘要 |",
            "| :--- | :--- | ---: | ---: | ---: | :--- |"]
    for e in entries:
        rows.append(
            "| {date} | [{title}]({dir}) | {dur} | {spk} | {todos}({mine}) | {summary} |".format(
                date=e.get("date") or "—",
                title=e.get("title") or e.get("id"),
                dir=e.get("dir", ""),
                dur=format_duration(e.get("duration_sec")),
                spk=e.get("speakers") or "—",
                todos=e.get("todos", 0),
                mine=e.get("mine", 0),
                summary=(e.get("summary") or "").replace("|", "／"),
            ))
    return head + "\n".join(rows) + "\n"


# ---------- 落盘 ----------
def index_paths(root=ROOT):
    return os.path.join(root, JSON_NAME), os.path.join(root, MD_NAME)


def load(root=ROOT):
    jf, _ = index_paths(root)
    if not os.path.exists(jf):
        return []
    with open(jf, encoding="utf-8") as f:
        return json.load(f).get("meetings", [])


def save(entries, root=ROOT):
    jf, mf = index_paths(root)
    os.makedirs(root, exist_ok=True)
    with open(jf, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "meetings": entries}, f, ensure_ascii=False, indent=2)
    with open(mf, "w", encoding="utf-8") as f:
        f.write(render_markdown(entries))
    return jf, mf


# ---------- 扫描一个转写目录 ----------
def _pick(files, *patterns):
    for pat in patterns:
        hit = [f for f in files if re.search(pat, f)]
        if hit:
            return sorted(hit)[0]
    return ""


def scan_dir(path):
    """读一个 `转写_*` 目录,拼出索引条目。目录里没有纪要则返回 None。"""
    path = os.path.abspath(path)
    files = os.listdir(path) if os.path.isdir(path) else []
    minutes_f = _pick(files, r"^纪要_.*\.md$")
    if not minutes_f:
        return None

    meeting_id = re.sub(r"^转写_", "", os.path.basename(path))
    with open(os.path.join(path, minutes_f), encoding="utf-8") as f:
        parsed = parse_minutes(f.read())

    duration = speakers = 0
    chars = 0
    json_f = _pick(files, r"\.json$")
    if json_f:
        try:
            with open(os.path.join(path, json_f), encoding="utf-8") as f:
                data = json.load(f)
            segs = data.get("segments") or []
            if segs:
                duration = int(float(segs[-1].get("end") or 0))
            speakers = len({s.get("speaker") for s in segs if s.get("speaker")})
        except (ValueError, OSError):
            pass

    transcript_f = _pick(files, r"_annotated\.txt$", r"\.txt$")
    if transcript_f:
        try:
            with open(os.path.join(path, transcript_f), encoding="utf-8") as f:
                chars = len(f.read())
        except OSError:
            pass

    return {
        "id": meeting_id,
        "title": derive_title(parsed["summary"], fallback=meeting_id),
        "date": parse_date(meeting_id),
        "dir": path,
        "duration_sec": duration,
        "speakers": speakers,
        "chars": chars,
        "todos": parsed["todos"],
        "mine": parsed["mine"],
        "summary": parsed["summary"],
        "files": {
            "minutes": minutes_f,
            "record": _pick(files, r"^会议记录_.*\.md$"),
            "transcript": transcript_f,
            "json": json_f,
        },
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _brain_ask():
    """借用 minutes 里的 ask()(同目录),不另起一个 LLM 客户端。

    Returns:
        ``ask(system, user) -> str``;大脑不可用时返回 None。
    """
    try:
        import importlib.util
        from importlib.machinery import SourceFileLoader
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "minutes")
        spec = importlib.util.spec_from_loader("_minutes", SourceFileLoader("_minutes", path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not mod.health_ok():
            return None
        return lambda sysmsg, user: mod.ask(sysmsg, user, max_tokens=40, temperature=0.3)
    except Exception:
        return None


def _titled(entry, ask):
    """有大脑就用它起标题,没有就保留 scan_dir 的截断兜底。"""
    if ask and entry.get("summary"):
        entry["title"] = llm_title(entry["summary"], ask=ask, fallback=entry["id"])
    return entry


def main(argv):
    if not argv:
        print(__doc__.strip()); return 1
    cmd = argv[0]

    if cmd == "upsert":
        if len(argv) < 2:
            print("用法: _index.py upsert <转写目录> [--title 标题]", file=sys.stderr); return 1
        entry = scan_dir(argv[1])
        if entry is None:
            print(f"⚠️ {argv[1]} 里没有纪要，跳过索引", file=sys.stderr); return 0
        if "--title" in argv:
            t = argv[argv.index("--title") + 1].strip()
            if t:
                entry["title"] = t
        elif "--no-llm" not in argv:
            entry = _titled(entry, _brain_ask())
        jf, mf = save(upsert(load(), entry))
        print(f"🗂  已登记「{entry['title']}」→ {os.path.basename(jf)} / {os.path.basename(mf)}")
        return 0

    if cmd == "rebuild":
        root = next((a for a in argv[1:] if not a.startswith("--")), ROOT)
        ask = None if "--no-llm" in argv else _brain_ask()
        if ask:
            print("🧠 用本地大脑起标题 …")
        entries = []
        for name in sorted(os.listdir(root)):
            if not name.startswith("转写_"):
                continue
            e = scan_dir(os.path.join(root, name))
            if e:
                entries = upsert(entries, _titled(e, ask))
        jf, mf = save(entries, root)
        print(f"🗂  已重建索引：{len(entries)} 场会 → {jf}")
        return 0

    if cmd == "list":
        print(render_markdown(load()))
        return 0

    print(f"未知命令: {cmd}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
