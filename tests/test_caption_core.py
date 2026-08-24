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


if __name__ == "__main__":
    test_segment_splits_on_silence()
    test_segment_drops_too_short()
    test_is_noise_filters_empty_and_hallu()
    test_translate_passthrough_zh()
    print("OK")
