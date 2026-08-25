#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""声纹注册表与匹配 —— 让会议转写自动认出每个说话人是谁。

在此之前所有人都是 `说话人A/B/C`，而「哪个是我」要每次手动传 `--me`。

**注册不靠念稿子**：分人本来就为每个说话人产出一条 256 维向量，给某场会的
`说话人C` 打上「李四」，等于一次性把李四注册了，以后所有会自动认。素材是真实
会议里的自然对话，比让每个人对着麦克风念一段更贴近实际发音状态。

实测依据（pyannote/speaker-diarization-community-1）：

    同一个人的两段独立录音        0.876
    同一个人跨独立分人的两段会议   0.882 / 0.943（与次高者差 0.686 / 0.754）
    陌生人                       −0.049 ~ 0.068

同人与陌生人差一个数量级，所以门槛卡在 0.55 很安全。

数据放 `~/.config/voiceprints.json`，**不进 git** —— 代码要可追溯、生物特征要可删除，
两者生命周期相反。
"""
import json
import os
from datetime import datetime

import numpy as np

REGISTRY = os.path.expanduser("~/.config/voiceprints.json")

THRESHOLD = 0.55      # 低于此值一律不认
MARGIN = 0.15         # 还要比次高的人高出这么多，否则宁可匿名


def cosine(a, b):
    """余弦相似度；任一为零向量时返回 0（而不是 NaN）。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def best_score(embedding, person):
    """一个人存多条向量时取**最大**相似度，不是平均。

    平均会被一条状态很差的旧录音拖垮；取最大意味着「只要有一次像就算」，
    这也是多存几次录音能越用越准的原因。
    """
    return max((cosine(embedding, e) for e in person.get("embeddings") or []), default=0.0)


def match_speakers(speakers, people, threshold=THRESHOLD, margin=MARGIN):
    """把本场会的说话人映射到已注册的人。

    Args:
        speakers: ``{标签: 向量}``，例如 ``{"SPEAKER_00": np.array([...])}``。
        people: 注册表条目列表。
        threshold: 相似度下限。
        margin: 与次高者的最小差距。

    Returns:
        ``{标签: 姓名}``，只含认得出的。**认不出的标签不会出现在结果里** ——
        宁可保留匿名牌，也不瞎猜。

    一对一：一个人在一场会里只占一个说话人位（分人器把同一个人切成两簇时，
    只有更像的那个拿到名字），反之一个说话人也只得到一个名字。
    """
    if not speakers or not people:
        return {}

    # 所有 (说话人, 人) 组合的得分，顺便记下每个说话人的次高分用于差距判定
    scored = []
    for label, emb in speakers.items():
        ranked = sorted(((best_score(emb, p), p["name"]) for p in people), reverse=True)
        if not ranked:
            continue
        top, runner_up = ranked[0], (ranked[1][0] if len(ranked) > 1 else 0.0)
        if top[0] < threshold:
            continue
        if top[0] - runner_up < margin:
            continue            # 两个注册者都沾边 → 拒绝，别挑一个
        scored.append((top[0], label, top[1]))

    mapping = {}
    used_names, used_labels = set(), set()
    for score, label, name in sorted(scored, reverse=True):
        if label in used_labels or name in used_names:
            continue            # 一对一
        mapping[label] = name
        used_labels.add(label)
        used_names.add(name)
    return mapping


def me_name(people):
    """注册表里标了 `is_me` 的那个人的名字；没有则 None。"""
    for p in people:
        if p.get("is_me"):
            return p.get("name")
    return None


def my_speaker(mapping, people):
    """从匹配结果里反查「我」是本场会的哪个说话人标签；我不在场则 None。"""
    mine = me_name(people)
    if not mine:
        return None
    for label, name in mapping.items():
        if name == mine:
            return label
    return None


# ---------- 注册表读写 ----------
def load_registry(path=REGISTRY):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("people", [])


def save_registry(people, path=REGISTRY):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "people": people}, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)      # 生物特征，别让同机其他用户随便读
    except OSError:
        pass
    return path


def add_embedding(people, name, embedding, source, is_me=False):
    """把一条向量记到某人名下；此人不存在就新建。

    重打同一个 source 不会重复攒（纠正打标时常发生）。
    `is_me=True` 会清掉其他人的 `is_me` —— 只能有一个「我」。
    """
    people = [dict(p) for p in people]
    vecs = np.asarray(embedding, dtype=np.float64).reshape(-1).tolist()

    person = next((p for p in people if p.get("name") == name), None)
    if person is None:
        person = {"name": name, "is_me": False, "embeddings": [], "sources": []}
        people.append(person)

    if source not in (person.get("sources") or []):
        person.setdefault("embeddings", []).append(vecs)
        person.setdefault("sources", []).append(source)
    person["updated"] = datetime.now().strftime("%Y-%m-%d")

    if is_me:
        for p in people:
            p["is_me"] = (p is person)
    return people


def forget(people, name):
    """整条删除某人。"""
    return [p for p in people if p.get("name") != name]
