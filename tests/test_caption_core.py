# /// script
# requires-python = ">=3.10"
# dependencies = ["webrtcvad", "numpy", "setuptools<81"]
# ///
import sys, os, math
sys.path.insert(0, os.path.expanduser("~/voice-assistant"))
import numpy as np, webrtcvad
import caption_core as cc


def _tone_frame(ms=30, sr=16000, freq=300):
    n = sr * ms // 1000
    t = np.arange(n)
    return (0.3 * 32767 * np.sin(2 * math.pi * freq * t / sr)).astype("<i2").tobytes()


def _silence_frame(ms=30, sr=16000):
    return b"\x00\x00" * (sr * ms // 1000)


def test_segment_splits_on_silence():
    frames = [_tone_frame()] * 33 + [_silence_frame()] * 33 + [_tone_frame()] * 20
    segs = list(cc.segment_frames(iter(frames), webrtcvad.Vad(2)))
    assert len(segs) == 2, f"got {len(segs)} segs"
    assert all(isinstance(s, bytes) and len(s) > 0 for s in segs)


def test_segment_drops_too_short():
    frames = [_tone_frame()] * 3 + [_silence_frame()] * 33
    segs = list(cc.segment_frames(iter(frames), webrtcvad.Vad(2)))
    assert segs == [], segs


def test_is_noise_filters_empty_and_hallu():
    assert cc.is_noise("") is True
    assert cc.is_noise("谢谢观看") is True
    assert cc.is_noise("Let's review the roadmap") is False


def test_translate_passthrough_zh():
    assert cc.translate("你好", "zh") == "你好"


def _capture_translate_body(fake_reply):
    """截下发给大脑的请求体，并拿回 translate 的返回值。"""
    import json as _json, io, urllib.request
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["body"] = _json.loads(req.data.decode())
        return _Resp(_json.dumps({"choices": [{"message": {"content": fake_reply}}]}).encode())

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        out = cc.translate("hello world", "en", url="http://127.0.0.1:9999/v1/chat/completions")
    finally:
        urllib.request.urlopen = orig
    return captured["body"], out


def test_translate_disables_thinking():
    """必须显式关思考，否则本机大脑把推理过程当译文吐出来。

    设计文档(2026-06-20)明确写了 chat_template_kwargs:{enable_thinking:false}，
    实现里一直没加、只剥 <think> 标签 —— 而模型是直接吐明文推理、根本没有标签。
    实测拿到的"译文"是「Here's a thinking process: 1. **Analyze User Input:**…」，
    对字幕浮窗是致命的。
    """
    body, _ = _capture_translate_body("你好世界")
    assert "chat_template_kwargs" in body, "请求体里没有 chat_template_kwargs"
    assert body["chat_template_kwargs"].get("enable_thinking") is False


def test_translate_still_strips_think_tags():
    _, out = _capture_translate_body("<think>琢磨一下</think>你好世界")
    assert out == "你好世界", out


if __name__ == "__main__":
    test_segment_splits_on_silence()
    test_segment_drops_too_short()
    test_is_noise_filters_empty_and_hallu()
    test_translate_passthrough_zh()
    test_translate_disables_thinking()
    test_translate_still_strips_think_tags()
    print("OK")
