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


def test_translate_falls_back_when_vendor_field_rejected():
    """严格端点(OpenAI/DeepSeek)不认 chat_template_kwargs，会回 400。

    CAPTION_LLM_URL 是让用户随便填的，不能假设对端是 vLLM ——
    被拒了要脱掉扩展字段重试，否则「填个 URL 和 key」根本用不了。
    """
    import io, json as _json, urllib.error, urllib.request
    seen = []

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        body = _json.loads(req.data.decode())
        seen.append(body)
        if "chat_template_kwargs" in body:
            raise urllib.error.HTTPError("u", 400, "Unrecognized argument", {}, None)
        return _Resp(_json.dumps({"choices": [{"message": {"content": "你好世界"}}]}).encode())

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        out = cc.translate("hello world", "en", url="http://127.0.0.1:9999/v1/chat/completions")
    finally:
        urllib.request.urlopen = orig

    assert out == "你好世界", out
    assert len(seen) == 2, f"应该重试一次，实际 {len(seen)} 次"
    assert "chat_template_kwargs" not in seen[-1]
    assert seen[-1]["model"] and seen[-1]["messages"], "降级把标准字段弄丢了"


def test_translate_does_not_swallow_other_http_errors():
    """只对 400/422 降级；401/500 该抛就抛，别把配置错误伪装成成功。"""
    import io, urllib.error, urllib.request

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        try:
            cc.translate("hello", "en", url="http://127.0.0.1:9999/v1/chat/completions")
        except urllib.error.HTTPError as e:
            assert e.code == 401
        else:
            raise AssertionError("401 不该被吞掉")
    finally:
        urllib.request.urlopen = orig


def test_segment_forces_cut_when_too_long():
    """连续说话不停顿时，必须有个时长上限强制断开。

    caption 的参数是为逐句字幕调的（停 0.8 秒才算一句）。会议里有人一口气说两分钟
    很常见 —— 实测拿真实会议音频跑，VAD 切出了 130 秒一整段。
    录音时要靠它给反馈，等 130 秒等于没有反馈。
    """
    frames = [_tone_frame() for _ in range(400)]      # 12 秒不间断语音
    segs = list(cc.segment_frames(iter(frames), _AlwaysSpeech(), max_speech=3.0))
    assert len(segs) >= 3, f"12 秒连续语音、上限 3 秒，应切出 ≥3 段，实际 {len(segs)}"
    for s in segs:
        secs = len(s) / 2 / 16000
        assert secs <= 4.0, f"有段落 {secs:.1f}s 超过上限太多"


def test_segment_unlimited_by_default():
    """不传 max_speech 时行为不变 —— caption 那条路依赖原有的按停顿切。"""
    frames = [_tone_frame() for _ in range(200)]      # 6 秒不间断
    segs = list(cc.segment_frames(iter(frames), _AlwaysSpeech()))
    assert len(segs) <= 1, "默认不该因为长度而切断"


class _AlwaysSpeech:
    """把每一帧都判成语音的假 VAD —— 用来构造「一直说不停」。"""

    def is_speech(self, frame, sample_rate):
        return True


if __name__ == "__main__":
    test_segment_splits_on_silence()
    test_segment_drops_too_short()
    test_is_noise_filters_empty_and_hallu()
    test_translate_passthrough_zh()
    test_translate_disables_thinking()
    test_translate_still_strips_think_tags()
    test_segment_forces_cut_when_too_long()
    test_segment_unlimited_by_default()
    test_translate_falls_back_when_vendor_field_rejected()
    test_translate_does_not_swallow_other_http_errors()
    print("OK")
