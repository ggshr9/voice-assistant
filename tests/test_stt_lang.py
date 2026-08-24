# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""验证 stt_server 能自动检测语种并返回 language 字段。需 8082 在跑。"""
import subprocess, requests


def _say(text, path):
    # macOS say 直接生成 16k 单声道 wav
    subprocess.run(["say", "-o", path, "--data-format=LEF32@16000", text], check=True)


def test_autodetect_english():
    p = "/tmp/cap_en.wav"
    _say("Let us review the third quarter roadmap.", p)
    r = requests.post("http://127.0.0.1:8082/transcribe",
                      json={"path": p, "language": "auto"}, timeout=30).json()
    assert "text" in r and r["text"].strip(), r
    assert r.get("language") == "en", r
    assert "roadmap" in r["text"].lower() or "review" in r["text"].lower(), r


def test_backward_compat_zh():
    p = "/tmp/cap_zh.wav"
    _say("今天我们过一下路线图", p)
    r = requests.post("http://127.0.0.1:8082/transcribe",
                      json={"path": p, "language": "zh"}, timeout=30).json()
    assert "language" in r, r


if __name__ == "__main__":
    test_autodetect_english()
    test_backward_compat_zh()
    print("OK")
