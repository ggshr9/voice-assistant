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

_LATIN = re.compile(r"[A-Za-z0-9]")
# CJK 汉字 + 中文标点 + 全角字符
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def needs_space(prev, nxt):
    """两段词级文本之间要不要补一个空格。"""
    if not prev or not nxt:
        return False
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


def speaker_names(turns):
    """SPEAKER_00 → 说话人A(按出场顺序;字母匿名牌,会后认出谁是谁再替换)。"""
    names = {}
    for spk, _ in turns:
        if spk not in names:
            i = len(names)
            names[spk] = f"说话人{string.ascii_uppercase[i]}" if i < 26 else f"说话人{i + 1}"
    return names


def write_annotated(json_path, out_dir):
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
    names = speaker_names(turns)
    if len(names) < 2:
        return None

    base = os.path.splitext(os.path.basename(json_path))[0]
    path = os.path.join(out_dir, base + "_annotated.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{names[spk]}：{txt}" for spk, txt in turns) + "\n")
    return path


def main(argv):
    if len(argv) < 2:
        print("用法: _annotate.py <转写.json> <输出目录>", file=sys.stderr)
        return 1
    json_path, out_dir = argv[0], argv[1]
    path = write_annotated(json_path, out_dir)
    if path:
        n = len(speaker_names(build_turns(json.load(open(json_path, encoding="utf-8"))
                                          .get("segments") or [])))
        print(f"🗣  已按说话人合成: {os.path.basename(path)}（{n} 人）")
    else:
        print("🗣  仅检出 1 个说话人，纪要用带标点原文")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
