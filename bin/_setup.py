#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安装自检的纯逻辑：各项检查的判定、配置文件读写。

**为什么是自检而不是填表**：这套东西的坑几乎都是「填了也不知道对不对」那一类——
pyannote 条款没同意会静默降级成纯转写、音频设备名差一个字就找不到、
配置指向的隧道其实没开、模型目录名不对。这些必须实际连一次、查一次才知道。

CLI 在 `bin/setup`，这里只放不碰 IO 的部分，好测。
"""
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from _atomicio import atomic_write  # noqa: E402

CONFIG = os.path.expanduser("~/.config/voice-assistant.env")
MODELS = os.path.expanduser("~/models")

OK = "ok"              # 通过
MISSING = "missing"    # 必需，缺
OPTIONAL = "optional"  # 可选，缺了也能用

AUDIO_DEVICES = {
    "会议录制": "聚合设备（BlackHole 2ch + 麦克风）— `rec online` 从这里录",
    "会议外放": "多输出设备（BlackHole 2ch + 扬声器）— 录制时系统输出切到这里",
}

REQUIRED_MODELS = ["Qwen3.6-35B-A3B-8bit"]
OPTIONAL_MODELS = ["Kokoro-82M-bf16", "IndexTTS-1.5"]

_HF_REPO = {
    "Qwen3.6-35B-A3B-8bit": "mlx-community/Qwen3.6-35B-A3B-8bit",
    "Kokoro-82M-bf16": "mlx-community/Kokoro-82M-bf16",
    "IndexTTS-1.5": "IndexTeam/IndexTTS-1.5",
}


class Check:
    """一项检查的结果：状态 + 人话 + 怎么修。

    没有 `fix` 的失败结果是没用的 —— 用户看到「❌ 模型缺失」也不知道下一步做什么。
    """

    def __init__(self, name, status, detail, fix=""):
        self.name = name
        self.status = status
        self.detail = detail
        self.fix = fix if status != OK else ""

    @property
    def ok(self):
        return self.status == OK

    @property
    def optional(self):
        return self.status == OPTIONAL

    def __repr__(self):
        return f"<Check {self.name} {self.status}>"


def usable(checks):
    """必需项全过就算可用；可选项缺失不影响。"""
    return all(c.ok or c.optional for c in checks)


# ---------- 配置文件 ----------
def read_env(path=CONFIG):
    """读 shell 风格的 env 文件；不存在返回空 dict。"""
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            v = v[1:-1].replace('\\"', '"')
        out[k.strip()] = v
    return out


def write_env(values, path=CONFIG):
    """写成可 `source` 的格式，权限 600（里面有 token）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# voice-assistant 配置（由 `setup` 生成，可手改）",
             "# 各脚本会 source 它；单独的 caption.env 等仍可覆盖这里的值。", ""]
    for k, v in values.items():
        escaped = str(v).replace('"', '\\"')
        lines.append(f'export {k}="{escaped}"')
    # 显式传 0o600:里面有 HF_TOKEN。从前是"写完再 chmod",那之间有一小段时间
    # 文件按 umask(通常 644)敞着 —— token 就那样躺在那里。
    atomic_write(path, "\n".join(lines) + "\n", mode=0o600)
    return path


# ---------- 各项检查（IO 都靠注入，便于测试）----------
def check_audio_devices(list_inputs, list_outputs=None):
    """两个音频设备必须手工建好，**名字一字不差**。

    **输入和输出要分开查**：`ffmpeg -list_devices` 只列【输入】设备，
    多输出设备「会议外放」永远不在里面 —— 第一版就是这么误报的，
    在一台设备齐全的机器上说缺「会议外放」。输出侧要问 SwitchAudioSource。

    Args:
        list_inputs: 返回输入设备文本的函数（ffmpeg）。
        list_outputs: 返回输出设备文本的函数（SwitchAudioSource）；
            为 None 时跳过输出侧检查，只报输入侧。
    """
    try:
        listing = list_inputs() or ""
    except Exception as exc:                       # noqa: BLE001
        return Check("音频设备", OPTIONAL, f"列不出设备（{type(exc).__name__}）",
                     fix="确认装了 ffmpeg：brew install ffmpeg")
    if list_outputs is not None:
        try:
            listing += "\n" + (list_outputs() or "")
        except Exception:                          # noqa: BLE001
            pass                                   # 查不到输出侧就只按输入侧判

    missing = [d for d in AUDIO_DEVICES if d not in listing]
    if not missing:
        return Check("音频设备", OK, "「会议录制」「会议外放」都在")
    fix = "打开「音频 MIDI 设置」手工建（名字必须一字不差）：\n" + "\n".join(
        f"       {d} — {AUDIO_DEVICES[d]}" for d in missing)
    return Check("音频设备", OPTIONAL,
                 "缺 " + "、".join(f"「{d}」" for d in missing) + "（只影响录线上会议）",
                 fix=fix)


def check_models(root=MODELS, required=None, optional=None):
    """`~/models/` 下该有的模型在不在。目录名写错等于没有——脚本里是写死的。"""
    required = REQUIRED_MODELS if required is None else required
    optional = OPTIONAL_MODELS if optional is None else optional
    try:
        present = set(os.listdir(root))
    except OSError:
        present = set()

    miss_req = [m for m in required if m not in present]
    miss_opt = [m for m in optional if m not in present]

    def dl(names):
        return "\n".join(
            f"       hf download {_HF_REPO.get(n, n)} --local-dir ~/models/{n}"
            for n in names)

    if miss_req:
        return Check("模型", MISSING, "缺 " + "、".join(miss_req),
                     fix="国内先 export HF_ENDPOINT=https://hf-mirror.com，然后：\n" + dl(miss_req))
    if miss_opt:
        return Check("模型", OPTIONAL,
                     "主力大脑就位；缺 " + "、".join(miss_opt) + "（只影响 clone / va / chat）",
                     fix=dl(miss_opt))
    return Check("模型", OK, "都在")


def check_token(env):
    """HF_TOKEN 只为下载 pyannote 门控模型（分人）。没有也能纯转写。"""
    tok = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if tok.strip():
        return Check("HF_TOKEN", OK, "已设置")
    return Check("HF_TOKEN", OPTIONAL, "没设 —— 分人会自动降级为纯转写",
                 fix="到 https://hf.co/settings/tokens 建一个 read token，然后跑 `setup --token hf_xxx`")


def check_gate(probe):
    """pyannote 是门控模型，必须先在网页点同意——没同意时下载会 401，而且是静默降级。

    Args:
        probe: 返回 True/False 的函数，实际去查一次模型元数据。
    """
    try:
        if probe():
            return Check("pyannote 条款", OK, "已同意，分人可用")
    except Exception as exc:                       # noqa: BLE001
        return Check("pyannote 条款", OPTIONAL, f"查不了（{type(exc).__name__}）",
                     fix="确认网络，或先设好 HF_TOKEN")
    return Check("pyannote 条款", OPTIONAL,
                 "未同意或 token 无权 —— 分人会静默降级为纯转写",
                 fix="打开 https://hf.co/pyannote/speaker-diarization-community-1 登录后点同意")


def check_brain(probe):
    """本机大脑（8080）在不在。不在也不算错，用的时候会自动拉起。"""
    try:
        if probe():
            return Check("本机大脑", OK, "运行中")
    except Exception:                              # noqa: BLE001
        pass
    return Check("本机大脑", OPTIONAL, "未运行（`minutes` 会自动拉起，首次加载 30-60 秒）",
                 fix="要现在起就跑：llm start")


def check_command(name, present, hint):
    if present:
        return Check(name, OK, "已安装")
    return Check(name, MISSING, "没找到", fix=hint)
