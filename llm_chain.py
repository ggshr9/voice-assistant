# -*- coding: utf-8 -*-
"""LLM 网关的候选链解析与重试分类 —— 全项目唯一的一份。

**为什么必须只有一份**:这套语义曾在 minutes_lib、caption_core、web/stt.py
各写一遍。第三份是补出来的:配置改成候选链后,前两处都会退避,唯独实时字幕的
translate 还是单点,把整串当模型名发出去 → 外语字幕的翻译死了一周,
而配置看起来完全正确。改一处漏两处比不改更糟。
现在三处 import 这一份,由 sync-web 推到服务器(与 prompts.py 同机制)。

各调用方保留自己的**重试节奏**(实时字幕等不起,批处理纪要可以多等),
这里只收敛真正共享的知识:链怎么解析、什么错误值得换下一个模型。
"""


def parse_chain(spec):
    """"A,B,C" → ["A","B","C"]。去重保序;空配置给 [""] —— 空串是合法的
    model(有些端点忽略该字段),给空列表会导致一次都不尝试。"""
    out = []
    for m in (spec or "").split(","):
        m = m.strip()
        if m and m not in out:
            out.append(m)
    return out or [""]


def try_next_model(e):
    """这个错误值不值得换下一个候选。

    分界线是**错在模型上还是错在网关上**:
      404 模型下线、429 限流、408/409、5xx 后端故障 —— 是【这个模型】的问题,换一个很可能就好
      连不上/DNS/超时(无 HTTP 状态码)          —— 是【网关本身】的问题,再试几个只是白等
    教训:第一版只认 404,实测撞上 429 就整个放弃,而换个模型立刻能用。
    """
    code = getattr(e, "code", None)
    if code is not None:
        return code in (404, 408, 409, 429) or code >= 500
    return False
