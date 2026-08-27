# -*- coding: utf-8 -*-
"""ASR 幻觉/噪声过滤 —— 全项目唯一的一份。

**为什么必须只有一份**:这套判定曾在 caption_core.py(本机)和 web/stt.py(服务器)
各写一遍,内容随即漂移:服务器版有「整句复读」检测(trombone 循环),本机版没有;
本机版有「单字符刷屏」检测(1h48m 的会结尾被填了 330 个「。嗯」),服务器版没有。
两边用户遇到的是同一批模型幻觉,防线却各缺一半。
现在两端 import 这一份,由 sync-web 推到服务器(与 prompts.py 同机制)。

判定分四层,任一命中即丢弃:
  1. 整句就是已知幻觉短语(精确匹配,含单词语气填充)
  2. 含字幕水印片段(whisper 训练集里混进的盗版字幕组文案)
  3. 单字符刷屏(「。嗯。嗯。嗯」这类,Counter 占比判定)
  4. 整句复读(同一 ≥10 字片段连续 ≥4 次 —— 生成循环)

**误删真话的代价高于显示一行垃圾** —— 垃圾用户可以无视,删掉的话找不回来。
所以第 4 层刻意保守:人会把一个词连说几遍(「哈喽哈喽哈喽」「对对对」),
但不会把十几字的句子一字不差连说四遍。曾两次因门槛过紧误杀用户真话
("hello" 连说 3 次、又连说 5 次),教训是:区分「人在重复」和「模型跑飞」的
不是次数,是**重复单元的长度**。
"""
import re
from collections import Counter

# 整句就是这些 → 幻觉。刻意用【明确列表】而不是长度规则:
# 中文的「好的」「对的」也是两个字,那是真话;英文两个字母才多半是语气填充。
HALLU_EXACT = {
    "you", "thank you", "thanks", "by", ".", "", "请订阅", "谢谢观看", "字幕",
    # whisper 在安静段落上最常吐的单词填充,单独成句时没有任何信息量
    "so", "uh", "um", "mm", "hmm", "hm", "oh", "ah", "eh", "er",
    "bye", "okay.", "the", "and",
}

# 出现即判定为幻觉的片段(训练集字幕/水印污染,真会议里不会说)
HALLU_SUBSTR = (
    "优优独播剧场", "yoyo television", "请订阅", "谢谢观看", "谢谢大家观看",
    "字幕由", "字幕组", "amara", "本字幕", "请不吝", "订阅转发", "点赞",
    "打赏支持明镜", "明镜与点点栏目", "感谢观看", "下期再见", "关注我",
)

MIN_REPS = 4          # 连续 4 段完全相同
MIN_UNIT_LEN = 10     # 且单元长到是个「句子」而不是「词」


def is_repetitive(text, min_reps=MIN_REPS, min_unit=MIN_UNIT_LEN):
    """同一片段连续重复 ≥min_reps 次,且片段本身不短 → 模型在打转。"""
    t = text.strip()
    if len(t) < min_unit * min_reps:
        return False
    for size in range(min_unit, len(t) // min_reps + 1):
        unit = t[:size]
        if unit.strip() and t.startswith(unit * min_reps):
            return True
    parts = [p.strip() for p in re.split(r"[。.!！?？,，;；]+", t) if p.strip()]
    if len(parts) >= min_reps:
        for i in range(len(parts) - min_reps + 1):
            win = parts[i:i + min_reps]
            if len(set(win)) == 1 and len(win[0]) >= min_unit:
                return True
    return False


def is_noise(text):
    """空串 / 已知幻觉 / 刷屏 / 生成循环 → True(丢弃)。"""
    if not isinstance(text, str):
        return True
    c = text.strip().strip("。.,，!！?？、 ").lower()
    if (not c) or len(c) < 2 or c in HALLU_EXACT:
        return True
    if any(k in c for k in HALLU_SUBSTR):
        return True
    # 单字符刷屏:去掉标点后,同一个字符占了一半以上(且不是超短句)
    t = re.sub(r"[，。！？、\s,.!?]", "", text)
    if len(t) >= 4 and Counter(t).most_common(1)[0][1] >= max(5, len(t) * 0.5):
        return True
    return is_repetitive(c)
