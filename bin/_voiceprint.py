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
import math
import json
import os
from datetime import datetime

import numpy as np

REGISTRY = os.path.expanduser("~/.config/voiceprints.json")

THRESHOLD = 0.55      # 低于此值一律不认
MARGIN = 0.15         # 还要比次高的人高出这么多，否则宁可匿名
SURE = 0.75           # 达到此值才算「几乎确定」；之间的算「勉强够线」，输出里会标出来
MERGE_SIM = 0.80      # 两簇之间相似到这个程度，才认为是同一个人被拆开了


def is_sure(score):
    """这个匹配是「几乎确定」还是「勉强够线」。

    实测同人在 0.88~0.94、陌生人 ≤0.19，中间那段空得很 —— 落在 0.55~0.75 的
    多半是素材太短或状态差异大，值得让人知道，而不是和 0.94 显示成一样。
    """
    return float(score) >= SURE


def cosine(a, b):
    """余弦相似度。任何算不出真实相似度的情况一律返回 0，**绝不返回 NaN**。

    为什么这条特别重要：NaN 参与比较时结果永远是 False，于是
    ``nan < threshold`` 和 ``nan - x < margin`` **两道安全闸门会同时失效** ——
    含 NaN 的向量反而畅通无阻地拿到名字。实测过：注册表里放一条含 NaN 的向量，
    本该匹配到「李四」的说话人被判成了那个 NaN 条目，相似度显示 nan。
    对声纹来说这是代价最高的错误：把一个人的话安到另一个人头上。

    零向量、维度不一致、含 NaN/inf —— 全部按「不认识」处理（返回 0）。
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.size != b.size:      # 维度对不上说明来源不同,不是"不像",是没法比
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    r = float(a @ b / (na * nb))
    # 只在结果上判一次就够:输入里的 NaN/inf 必然经 norm 传导到 r。
    # 曾经在入口再加一遍 np.isfinite(a).all(),但那是每次比较都全数组扫一遍的
    # 热路径开销(说话人数 × 注册人数 × 每人向量数),而变异测试证明单靠这行就能拦住。
    return r if math.isfinite(r) else 0.0


def best_score(embedding, person):
    """一个人存多条向量时取**最大**相似度，不是平均。

    平均会被一条状态很差的旧录音拖垮；取最大意味着「只要有一次像就算」，
    这也是多存几次录音能越用越准的原因。
    """
    return max((cosine(embedding, e) for e in person.get("embeddings") or []), default=0.0)


def match_speakers(speakers, people, threshold=THRESHOLD, margin=MARGIN, scores=False,
                   merge_clusters=False, merge_sim=MERGE_SIM):
    """把本场会的说话人映射到已注册的人。

    Args:
        speakers: ``{标签: 向量}``，例如 ``{"SPEAKER_00": np.array([...])}``。
        people: 注册表条目列表。
        threshold: 相似度下限。
        margin: 与次高者的最小差距。

    Args (续):
        scores: 为 True 时返回 ``{标签: (姓名, 相似度)}``，便于上游标注可信度。
        merge_clusters: 允许一个人认领**多个**簇。开启后，额外的簇必须**与主簇
            本身相似** ≥ merge_sim 才会被并入，只是各自跟注册者沾边不算 ——
            否则两个真人都沾边时会被错并成一个。

            **默认关闭，而且多数情况下你不需要它。** 这个选项是为「分人器真的把
            一个人拆开了」准备的，但实测发现最常见的成因是**使用不当**：同一段
            音频，不指定人数 → 2 人、最大簇间相似度 0.212；强制 8 人 → 8 人、
            最大 0.927（0.9+ 是同一个人两段录音的水平）。**先别指定说话人数**，
            那比开这个开关有效得多。
        merge_sim: 上面那个簇间相似度门槛。

    Returns:
        ``{标签: 姓名}``（或带分数的二元组），只含认得出的。
        **认不出的标签不会出现在结果里** —— 宁可保留匿名牌，也不瞎猜。

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
    primary = {}                # 姓名 -> 主簇标签（分数最高的那个）
    used_names, used_labels = set(), set()
    for score, label, name in sorted(scored, reverse=True):
        if label in used_labels:
            continue
        if name in used_names:
            if not merge_clusters:
                continue        # 一对一
            # 过分割修复：只有跟主簇本身足够像，才认为是同一个人被拆开
            if cosine(speakers[label], speakers[primary[name]]) < merge_sim:
                continue
        else:
            primary[name] = label
            used_names.add(name)
        mapping[label] = (name, round(score, 4)) if scores else name
        used_labels.add(label)
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
