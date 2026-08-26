# -*- coding: utf-8 -*-
"""实时字幕的 ASR 后端层 —— 换模型只动这一个文件。

**为什么值得抽一层**:2026-08-26 从 faster-whisper 换 Qwen3-ASR 时,改动散在
stt.py 的三处 —— 加载方式、调用怪癖(token 上限要设实例属性)、语言代码映射
(界面用 zh/en,Qwen 要 Chinese/English,whisper 又用 zh/en)。这些全是
**模型私有的知识**,却和能量门、幻觉过滤这些模型无关的逻辑搅在一起。
下次换 FireRedASR2 又得整个再理一遍。

契约(harness 与后端的唯一边界):
    backend.transcribe(audio_f32, lang_ui) -> (text, lang_ui)
      audio_f32: float32 单声道 16kHz,幅值 [-1,1]
      lang_ui:   界面语言码 "zh" / "en" / None(自动)
      返回:      (转写文本, 界面语言码) —— 语言映射由各后端自己消化,不外漏

选择后端: CAPTION_ASR_BACKEND=qwen3|whisper  (默认 qwen3)
新增后端: 写一个类,@register("名字"),实现 transcribe。别的都不用碰。
"""
import os

SR = 16000

BACKENDS = {}


def register(name):
    def deco(cls):
        BACKENDS[name] = cls
        return cls
    return deco


@register("qwen3")
class Qwen3Backend:
    """Qwen/Qwen3-ASR-1.7B。中文实测明显强于 whisper(见 stt.py 顶部的对照)。"""

    # 每秒音频允许生成多少 token。**延迟护栏,不是质量参数**:实测某个 10 秒片段
    # 模型进生成循环,跑满 256 token 用了 5.67 秒,正常片段只要 0.13 秒。
    TOKENS_PER_SEC = 8
    MIN_NEW, MAX_NEW = 24, 256
    _LANG = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ja": "Japanese"}
    _BACK = {v: k for k, v in _LANG.items()}

    def __init__(self):
        import torch
        from qwen_asr import Qwen3ASRModel
        model_id = os.environ.get("CAPTION_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
        device = os.environ.get("CAPTION_ASR_DEVICE", "cuda:1")  # 避开批处理占的 cuda:0
        print(f"加载 {model_id} 到 {device} ...", flush=True)
        self.m = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16,
                                               device_map=device,
                                               max_new_tokens=self.MAX_NEW)

    def transcribe(self, audio, lang_ui):
        secs = len(audio) / SR
        # transcribe() 签名里【没有】max_new_tokens 参数,它读实例属性
        # (qwen3_asr.py:379)。按参数传会静默走 TypeError 兜底,上限从不生效 —— 踩过。
        self.m.max_new_tokens = max(self.MIN_NEW,
                                    min(self.MAX_NEW, int(secs * self.TOKENS_PER_SEC)))
        r = self.m.transcribe(audio=(audio, SR), language=self._LANG.get(lang_ui or ""))
        if not r:
            return "", lang_ui or ""
        text = (getattr(r[0], "text", "") or "").strip()
        got = getattr(r[0], "language", None)
        return text, self._BACK.get(got, lang_ui or "")


@register("whisper")
class WhisperBackend:
    """faster-whisper。留作回退与英文场景 —— 中文上有臭名昭著的字幕水印幻觉
    (「优优独播剧场」),别在中文会议里选它。"""

    def __init__(self):
        from faster_whisper import WhisperModel
        model_id = os.environ.get("CAPTION_ASR_MODEL", "large-v3-turbo")
        device = os.environ.get("CAPTION_ASR_DEVICE", "cuda")
        print(f"加载 {model_id} 到 {device} (faster-whisper)...", flush=True)
        self.m = WhisperModel(model_id, device=device, compute_type="float16")

    def transcribe(self, audio, lang_ui):
        segs, info = self.m.transcribe(audio, language=lang_ui or None,
                                       condition_on_previous_text=False)
        return "".join(s.text for s in segs).strip(), (info.language or lang_ui or "")


def load_backend(name=None):
    name = name or os.environ.get("CAPTION_ASR_BACKEND", "qwen3")
    if name not in BACKENDS:
        raise ValueError(f"未知 ASR 后端: {name}（可用: {'/'.join(sorted(BACKENDS))}）")
    return BACKENDS[name]()
