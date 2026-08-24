# /// script
# requires-python = ">=3.10"
# dependencies = ["sounddevice", "webrtcvad", "numpy", "setuptools<81", "pyobjc-framework-Cocoa"]
# ///
import sys, os
sys.path.insert(0, os.path.expanduser("~/voice-assistant"))
import caption


def test_process_segment_translates():
    caption.cc.stt = lambda path, **k: ("Let's review the roadmap", "en")
    caption.cc.translate = lambda text, lang, **k: "我们来看下路线图"
    out = caption.process_segment(b"\x00\x00" * 8000)
    assert out == ("Let's review the roadmap", "我们来看下路线图"), out


def test_process_segment_drops_noise():
    caption.cc.stt = lambda path, **k: ("谢谢观看", "zh")
    assert caption.process_segment(b"\x00\x00" * 8000) is None


if __name__ == "__main__":
    test_process_segment_translates()
    test_process_segment_drops_noise()
    print("OK")
