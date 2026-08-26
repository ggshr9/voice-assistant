# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""stt_server 的语种自动检测（集成测试，需要 `uv run stt_server.py` 起在 8082）。

**为什么用 skip 而不是 fail**：这是全套测试里唯一依赖外部常驻服务的一条。
服务没起时报失败，会让整批测试出现假红，久了就没人看结果了 —— 那比少一条覆盖更糟。
服务起着的时候它照常会挂，该抓的仍然抓得到。

跑: uv run tests/test_stt_lang.py
"""
import platform
import subprocess
import sys
import unittest

import requests

URL = "http://127.0.0.1:8082/transcribe"


def _server_up():
    try:
        requests.get("http://127.0.0.1:8082/", timeout=2)
        return True
    except requests.exceptions.ConnectionError:
        return False
    except requests.RequestException:
        return True          # 连上了、只是没这个路由，服务算活着


NEED = unittest.skipUnless(platform.system() == "Darwin" and _server_up(),
                           "需要 macOS 的 say + stt_server 起在 8082")


def _say(text, path):
    """macOS say 直接生成 16k 单声道 wav。"""
    subprocess.run(["say", "-o", path, "--data-format=LEF32@16000", text], check=True)


@NEED
class TestLanguageDetect(unittest.TestCase):
    def test_英文被自动判成en(self):
        p = "/tmp/cap_en.wav"
        _say("Let us review the third quarter roadmap.", p)
        r = requests.post(URL, json={"path": p, "language": "auto"}, timeout=30).json()
        self.assertTrue(r.get("text", "").strip(), r)
        self.assertEqual(r.get("language"), "en", r)
        self.assertTrue("roadmap" in r["text"].lower() or "review" in r["text"].lower(), r)

    def test_显式传语种仍然回language字段(self):
        """老调用方传 language=zh —— 加了自动检测之后不能把这条路走坏。"""
        p = "/tmp/cap_zh.wav"
        _say("今天我们过一下路线图", p)
        r = requests.post(URL, json={"path": p, "language": "zh"}, timeout=30).json()
        self.assertIn("language", r)


if __name__ == "__main__":
    if not _server_up():
        print("SKIP: stt_server 没起在 8082（先跑 `uv run stt_server.py`）")
        sys.exit(0)
    unittest.main(verbosity=2)
