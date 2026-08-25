# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/_setup.py 的纯逻辑测试：各项检查的判定与配置读写。

真正连网络/查设备的部分靠注入，所以这些测试不需要 HF token、不需要音频设备。
跑: uv run tests/test_setup.py
"""
import importlib.util
import pathlib
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "vsetup", SourceFileLoader("vsetup", str(REPO / "bin" / "_setup.py")))
vsetup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vsetup)


class TestEnvFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(pathlib.Path(self._tmp.name) / "voice-assistant.env")

    def tearDown(self):
        self._tmp.cleanup()

    def test_没有文件时返回空(self):
        self.assertEqual(vsetup.read_env(self.path), {})

    def test_写入再读出(self):
        vsetup.write_env({"HF_TOKEN": "hf_x", "CAPTION_LLM_URL": "http://a/b"}, self.path)
        got = vsetup.read_env(self.path)
        self.assertEqual(got["HF_TOKEN"], "hf_x")
        self.assertEqual(got["CAPTION_LLM_URL"], "http://a/b")

    def test_写出的是可source的shell格式(self):
        vsetup.write_env({"HF_TOKEN": "hf_x"}, self.path)
        text = pathlib.Path(self.path).read_text(encoding="utf-8")
        self.assertIn('export HF_TOKEN="hf_x"', text)

    def test_值里有引号也不破坏格式(self):
        vsetup.write_env({"K": 'a"b'}, self.path)
        self.assertEqual(vsetup.read_env(self.path)["K"], 'a"b')

    def test_更新保留其他键(self):
        vsetup.write_env({"A": "1", "B": "2"}, self.path)
        vsetup.write_env({**vsetup.read_env(self.path), "B": "3"}, self.path)
        got = vsetup.read_env(self.path)
        self.assertEqual((got["A"], got["B"]), ("1", "3"))

    def test_注释与空行被忽略(self):
        pathlib.Path(self.path).write_text(
            '# 注释\n\nexport A="1"\nB=2\n', encoding="utf-8")
        self.assertEqual(vsetup.read_env(self.path), {"A": "1", "B": "2"})

    def test_权限是600(self):
        import os
        vsetup.write_env({"HF_TOKEN": "hf_x"}, self.path)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600,
                         "配置里有 token，不能让同机其他用户读")


class TestCheckResults(unittest.TestCase):
    """检查项的结果结构：状态 + 人话 + 怎么修。"""

    def test_ok的检查不带修复建议(self):
        r = vsetup.Check("大脑", vsetup.OK, "运行中")
        self.assertTrue(r.ok)
        self.assertEqual(r.fix, "")

    def test_失败的检查必须给出怎么办(self):
        r = vsetup.Check("模型", vsetup.MISSING, "缺 X", fix="hf download …")
        self.assertFalse(r.ok)
        self.assertIn("hf download", r.fix)

    def test_可选项缺失不算失败(self):
        r = vsetup.Check("声音克隆", vsetup.OPTIONAL, "没装", fix="…")
        self.assertFalse(r.ok)
        self.assertTrue(r.optional, "可选项不该让整体自检判定为不可用")


class TestAudioDeviceCheck(unittest.TestCase):
    """录线上会议要两个手工建的设备，名字必须一字不差。"""

    def test_两个设备都在(self):
        listing = "[0] MacBook Pro Microphone\n[1] 会议录制\n[2] 会议外放"
        r = vsetup.check_audio_devices(lambda: listing)
        self.assertTrue(r.ok)

    def test_缺聚合设备时点名缺哪个(self):
        r = vsetup.check_audio_devices(lambda: "[0] MacBook Pro Microphone")
        self.assertFalse(r.ok)
        self.assertIn("会议录制", r.detail)

    def test_名字写错等于没有(self):
        """「会议录音」不是「会议录制」——差一个字 rec online 就找不到设备。"""
        r = vsetup.check_audio_devices(lambda: "[1] 会议录音\n[2] 会议外放")
        self.assertFalse(r.ok)
        self.assertIn("会议录制", r.detail)

    def test_输出设备要单独查(self):
        """ffmpeg -list_devices 只列【输入】，多输出设备「会议外放」永远不在里面。

        第一版只查输入，在一台设备齐全的真机上误报「缺 会议外放」。
        """
        inputs = "[0] 会议录制\n[1] MacBook Pro Microphone"
        outputs = "BlackHole 2ch\nMacBook Pro Speakers\n会议外放"
        self.assertTrue(vsetup.check_audio_devices(lambda: inputs, lambda: outputs).ok)

    def test_只查输入时不该误判输出设备(self):
        """不传输出查询函数就只按输入判——但那样必然报缺，所以调用方必须传。"""
        r = vsetup.check_audio_devices(lambda: "[0] 会议录制")
        self.assertIn("会议外放", r.detail)

    def test_查输出失败时退回只看输入(self):
        def boom():
            raise OSError("SwitchAudioSource 不在")
        r = vsetup.check_audio_devices(lambda: "[0] 会议录制\n[1] 会议外放", boom)
        self.assertTrue(r.ok, "输入侧已经都有时，查不到输出侧不该反而报错")

    def test_列设备失败时不崩(self):
        def boom():
            raise OSError("ffmpeg 不在")
        self.assertFalse(vsetup.check_audio_devices(boom).ok)


class TestModelCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_模型齐全(self):
        (self.root / "Qwen3.6-35B-A3B-8bit").mkdir()
        r = vsetup.check_models(str(self.root),
                                required=["Qwen3.6-35B-A3B-8bit"], optional=[])
        self.assertTrue(r.ok)

    def test_必需的在_可选的缺_算可选缺失不算失败(self):
        """缺 Kokoro/IndexTTS 只影响 clone/va，不该让人以为装坏了。"""
        (self.root / "Qwen3.6-35B-A3B-8bit").mkdir()
        r = vsetup.check_models(str(self.root),
                                required=["Qwen3.6-35B-A3B-8bit"], optional=["Kokoro-82M-bf16"])
        self.assertTrue(r.optional, "应标成可选缺失")
        self.assertIn("Kokoro", r.detail)
        self.assertIn("clone", r.detail)

    def test_缺模型时给出下载命令(self):
        r = vsetup.check_models(str(self.root),
                                required=["Qwen3.6-35B-A3B-8bit"], optional=[])
        self.assertFalse(r.ok)
        self.assertIn("hf download", r.fix)
        self.assertIn("Qwen3.6-35B-A3B-8bit", r.detail)

    def test_目录不存在时不崩(self):
        r = vsetup.check_models(str(self.root / "不存在"), required=["X"], optional=[])
        self.assertFalse(r.ok)


class TestOverallVerdict(unittest.TestCase):
    def test_必需项全过就是可用(self):
        checks = [vsetup.Check("a", vsetup.OK, ""),
                  vsetup.Check("b", vsetup.OPTIONAL, "", fix="x")]
        self.assertTrue(vsetup.usable(checks), "可选项缺失不该判定为不可用")

    def test_有必需项失败就是不可用(self):
        checks = [vsetup.Check("a", vsetup.OK, ""),
                  vsetup.Check("b", vsetup.MISSING, "", fix="x")]
        self.assertFalse(vsetup.usable(checks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
