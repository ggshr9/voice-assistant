#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录音时的实时反馈：电平表 + 抽样转写。从 stdin 读 16k 单声道 PCM。

**为什么需要**：`rec` 从前录完之前什么反馈都没有。最致命的场景是开两小时会，
结束才发现聚合设备没抓到对方声音 —— 不可恢复。电平表证明「有声音进来」，
转写证明「ASR 听得懂」，后者是电平表证明不了的（音量正常但严重回声/削波，
ASR 照样出垃圾）。

**为什么不切 10 秒文件**：固定边界会从句子中间劈开、有磁盘 churn、还得等满
10 秒才有反馈。这里复用 `caption_core` 的 VAD 按停顿切句，零文件、一句说完就出。

**成本**（实测，8bit 权重常驻）：模型加载 2.2s，之后每段 12-14 秒音频转写
0.26~0.39 秒 —— 占空比约 2.5%，所以默认就开着。

用法（由 `rec` 调用，一般不用手动跑）：
    ffmpeg ... -f s16le pipe:1 | _live_asr.py [--no-asr]

    --no-asr   只显示电平表，不加载模型（`rec --quiet` 用）
"""
import math
import os
import sys

SR = 16000
FRAME_MS = 30
FRAME_BYTES = SR * FRAME_MS // 1000 * 2

MAX_SPEECH = 12.0        # 一段最多攒这么久就断开(与 web/config.py 的 MAX_SEG_SEC 对齐)
BLANK_WARN_AFTER = 3     # 连续几段都空白就提示可能没录上

_BARS = "▁▂▃▄▅▆▇█"


def rms_dbfs(pcm):
    """一段 int16 PCM 的 dBFS；全静音返回 -99。"""
    import numpy as np
    a = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    if a.size == 0:
        return -99.0
    rms = float(np.sqrt((a * a).mean()))
    return 20 * math.log10(rms) if rms > 1e-9 else -99.0


def level_bar(dbfs, width=8):
    """把 dBFS 画成条。-60dB 以下全空，0dB 满格。

    录音场景常年在 -40~-10dB，所以刻度落在这一段，太线性的映射会看着一直没动静。
    """
    if dbfs <= -60:
        return _BARS[0] * width
    frac = min(1.0, max(0.0, (dbfs + 60) / 60))
    lit = max(1, round(frac * width))
    peak = _BARS[min(len(_BARS) - 1, int(frac * (len(_BARS) - 1)))]
    return (peak * lit).ljust(width, _BARS[0])


def should_warn(blank_streak, warn_after=BLANK_WARN_AFTER):
    """连续多段转写都空白 → 大概率没抓到声音，该提示了。"""
    return blank_streak >= warn_after


def fmt_elapsed(seconds):
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 \
        else f"{s // 60:02d}:{s % 60:02d}"


def _status(elapsed, dbfs, text=""):
    line = f"\r🔴 {fmt_elapsed(elapsed)}  {level_bar(dbfs)} {dbfs:5.0f}dB"
    if text:
        line += f"  {text}"
    return line[:150]


def main(argv):
    import time
    no_asr = "--no-asr" in argv
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    stdin = sys.stdin.buffer
    started = time.time()
    last_text = ""

    sess = vad = cc = None
    if not no_asr:
        try:
            import webrtcvad
            import caption_core as _cc
            from mlx_qwen3_asr import Session
            cc = _cc
            vad = webrtcvad.Vad(2)
            sess = Session(os.environ.get("MEETING_MODEL",
                                          "mlx-community/Qwen3-ASR-1.7B-8bit"))
        except Exception as e:                       # noqa: BLE001
            print(f"\r（实时转写不可用：{type(e).__name__}，只显示电平）", file=sys.stderr)
            sess = None

    def frames():
        """边读边更新电平；同时把帧交给 VAD。"""
        while True:
            b = stdin.read(FRAME_BYTES)
            if len(b) < FRAME_BYTES:
                return
            sys.stderr.write(_status(time.time() - started, rms_dbfs(b), last_text))
            sys.stderr.flush()
            yield b

    if sess is None:
        for _ in frames():
            pass
        return 0

    import numpy as np
    blank = 0
    for pcm in cc.segment_frames(frames(), vad, max_speech=MAX_SPEECH):
        audio = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        try:
            text = (sess.transcribe(audio, language=os.environ.get("MEETING_LANG", "Chinese")
                                    ).text or "").strip()
        except Exception:                            # noqa: BLE001
            continue
        if not text or cc.is_noise(text):
            blank += 1
            if should_warn(blank):
                print(f"\r⚠️  连续 {blank} 段没转出内容 —— 检查是否真的录到了声音"
                      f"（线上会议要确认系统输出切到了「会议外放」）", file=sys.stderr)
                blank = 0
            continue
        blank = 0
        last_text = "…" + text[-46:] if len(text) > 46 else text
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
