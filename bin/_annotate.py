#!/usr/bin/env python3
"""把 Qwen3-ASR 的词级 segment 合成带说话人标注的稿子(_annotated.txt)。

Qwen3-ASR 只把 speaker 写进 json,txt/srt 是纯文本,minutes 读不到说话人 —— 这里补上。

**为什么拼接要讲究**:开了分人就会强制开词级时间戳,于是 segment 是一个词一个词的。
中文直接拼没问题,英文直接拼会糊成一坨。生产数据里真的发生了:

    说话人C：Yeahsososothemainmainmainquestionuhfirstlasttimeumwehadthecall...

而同一段在纯 txt 里是正常的 "So let's say if the users have...";中英混说的会议里,
LLM 一直在读这种糊掉的英文写纪要。

规则:只在【下一段以拉丁字母/数字开头】且【上一段末尾不是中文或全角标点】时补空格。
中↔英交界不补,跟 ASR 自己产出的纯 txt 风格一致(「呃，OK」而不是「呃， OK」)。

用法: _annotate.py <转写.json> <输出目录>
"""
import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from _atomicio import atomic_write  # noqa: E402

_LATIN = re.compile(r"[A-Za-z0-9]")
# 「不需要词间空格」的东方文字。按 Unicode 区段列，别只写汉字 ——
# 早先只有 汉字 + 中文标点 + 全角，于是日文假名和韩文谚文后面会被插进一个空格
# （`あ` + `hi` → `あ hi`）。这些文字本来就不用空格分词，插进去是把文本弄脏。
_CJK = re.compile(
    "["
    "\u1100-\u11FF"      # 谚文字母
    "\u3000-\u303F"      # CJK 标点（　、。〈〉…）
    "\u3040-\u309F"      # 平假名
    "\u3130-\u318F"      # 谚文兼容字母（ㄱㄴㄷ…）
    "\u30A0-\u30FF"      # 片假名
    "\u3400-\u4DBF"      # 汉字扩展 A（生僻字）
    "\u4E00-\u9FFF"      # 汉字基本区
    "\uA960-\uA97F"      # 谚文字母扩展 A
    "\uAC00-\uD7AF"      # 谚文音节
    "\uF900-\uFAFF"      # 汉字兼容表意
    "\uFF00-\uFFEF"      # 半角/全角形式
    "]")
# 刻意不含 U+2026「…」等通用标点:中英文共用,中文里不该补空格、
# 英文里 `wait… Word` 反而该补 —— 这是真歧义,不替调用方决定。


def needs_space(prev, nxt):
    """两段词级文本之间要不要补一个空格。"""
    if not prev or not nxt:
        return False
    if not isinstance(prev, str) or not isinstance(nxt, str):
        return False                 # 上游给了非字符串就当"不补",别把整条流程带崩
    a, b = prev[-1], nxt[0]
    if a.isspace() or b.isspace():
        return False
    if not _LATIN.match(b):          # 下一段不是拉丁开头(中文、撇号、标点)→ 不补
        return False
    if _CJK.match(a):                # 中文或全角标点之后 → 不补
        return False
    return True


def build_turns(segs):
    """连续同一说话人的 segment 合并成一「轮」,返回 [(speaker, text), ...]。"""
    turns, cur, sp = [], "", None
    for s in segs:
        spk = s.get("speaker") or "SPEAKER_?"
        if spk != sp and cur:
            turns.append((sp, cur))
            cur = ""
        sp = spk
        text = s.get("text", "")
        cur += (" " if needs_space(cur, text) else "") + text
    if cur:
        turns.append((sp, cur))
    return turns


def speaker_names(turns, identified=None):
    """SPEAKER_00 → 显示名。

    Args:
        turns: build_turns 的结果。
        identified: 声纹认出来的 ``{SPEAKER_xx: 真名}``。认出来的用真名，
            认不出的仍是按出场顺序的匿名牌 `说话人A/B/C`。

    匿名牌的字母**只在未识别者之间**顺序分配，所以一场会里可能是
    「李四 / 说话人A / 张三 / 说话人B」——这是刻意的：字母只是占位，
    不该因为有人被认出来就跳号。
    """
    identified = identified or {}
    names, anon = {}, 0
    for spk, _ in turns:
        if spk in names:
            continue
        if spk in identified:
            names[spk] = identified[spk]
        else:
            names[spk] = (f"说话人{string.ascii_uppercase[anon]}" if anon < 26
                          else f"说话人{anon + 1}")
            anon += 1
    return names


def write_annotated(json_path, out_dir, identified=None):
    """读转写 json,写 <名>_annotated.txt。没分人或只有一个说话人时返回 None 不写。

    单人时不写是刻意的:segment 文本没标点,不如带标点的纯 txt 好读,
    写了反而会被 minutes 优先选走。
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments") or []
    if not any(s.get("speaker") for s in segs):
        return None

    turns = build_turns(segs)
    names = speaker_names(turns, identified)
    if len(names) < 2:
        return None

    base = os.path.splitext(os.path.basename(json_path))[0]
    path = os.path.join(out_dir, base + "_annotated.txt")
    atomic_write(path, "\n".join(f"{names[spk]}：{txt}" for spk, txt in turns) + "\n")
    return path


def main(argv):
    if len(argv) < 2:
        print("用法: _annotate.py <转写.json> <输出目录>", file=sys.stderr)
        return 1
    json_path, out_dir = argv[0], argv[1]
    identified = {}
    if "--names" in argv:              # SPEAKER_00=张三:0.94,SPEAKER_03=李四:0.61
        for pair in argv[argv.index("--names") + 1].split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            name, sep, score = v.strip().rpartition(":")
            if sep and name:
                # 勉强够线的标 [?] —— 与逐字记录的存疑标注同一套约定。
                # 否则 0.56 和 0.94 在稿子里长得一模一样，没人知道该不该信。
                try:
                    name = name if float(score) >= 0.75 else f"{name}[?]"
                except ValueError:
                    name = v.strip()
            else:
                name = v.strip()
            identified[k.strip()] = name
    path = write_annotated(json_path, out_dir, identified)
    if path:
        n = len(speaker_names(build_turns(json.load(open(json_path, encoding="utf-8"))
                                          .get("segments") or [])))
        print(f"@@STAGE diarized {n}", file=sys.stderr)
        print(f"🗣  已按说话人合成: {os.path.basename(path)}（{n} 人）")
    else:
        print("🗣  仅检出 1 个说话人，纪要用带标点原文")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
