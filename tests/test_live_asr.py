# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""bin/_live_asr.py 的纯逻辑测试：电平换算、条形渲染、空白告警、计时格式。

不加载模型、不读 stdin。跑: uv run tests/test_live_asr.py
"""
import importlib.util
import math
import pathlib
import unittest
from importlib.machinery import SourceFileLoader

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "live_asr", SourceFileLoader("live_asr", str(REPO / "bin" / "_live_asr.py")))
la = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la)


def pcm(amplitude, n=480):
    """造一段给定幅度的 int16 PCM（正弦）。"""
    t = np.arange(n)
    a = (amplitude * 32767 * np.sin(2 * math.pi * 300 * t / 16000)).astype("<i2")
    return a.tobytes()


class TestRmsDbfs(unittest.TestCase):
    def test_全静音是极小值(self):
        self.assertLessEqual(la.rms_dbfs(b"\x00\x00" * 480), -90)

    def test_空输入不崩(self):
        self.assertEqual(la.rms_dbfs(b""), -99.0)

    def test_响的比轻的高(self):
        self.assertGreater(la.rms_dbfs(pcm(0.5)), la.rms_dbfs(pcm(0.05)))

    def test_满幅接近0dB(self):
        """正弦满幅的 RMS 是 1/√2 ≈ -3dBFS。"""
        self.assertGreater(la.rms_dbfs(pcm(1.0)), -6)


class TestLevelBar(unittest.TestCase):
    def test_静音是空条(self):
        bar = la.level_bar(-99)
        self.assertEqual(len(bar), 8)
        self.assertEqual(set(bar), {"▁"})

    def test_大声时条更满(self):
        quiet, loud = la.level_bar(-45), la.level_bar(-6)
        self.assertLess(quiet.count("▁"), 8)
        self.assertLess(loud.count("▁"), quiet.count("▁"), "越响空格越少")

    def test_宽度恒定(self):
        """条宽会变的话，终端那行会不停抖。"""
        for db in (-99, -60, -40, -20, -6, 0):
            self.assertEqual(len(la.level_bar(db)), 8, f"{db}dB 宽度不对")

    def test_可自定义宽度(self):
        self.assertEqual(len(la.level_bar(-20, width=16)), 16)


class TestBlankWarning(unittest.TestCase):
    """连续转不出东西 = 大概率没录上，这是最致命且不可恢复的故障。"""

    def test_偶尔一段空白不告警(self):
        self.assertFalse(la.should_warn(1))
        self.assertFalse(la.should_warn(2))

    def test_连续三段就告警(self):
        self.assertTrue(la.should_warn(3))
        self.assertTrue(la.should_warn(9))

    def test_阈值可调(self):
        self.assertFalse(la.should_warn(3, warn_after=5))
        self.assertTrue(la.should_warn(5, warn_after=5))


class TestFmtElapsed(unittest.TestCase):
    def test_不足一小时是分秒(self):
        self.assertEqual(la.fmt_elapsed(0), "00:00")
        self.assertEqual(la.fmt_elapsed(125), "02:05")

    def test_超过一小时带小时位(self):
        self.assertEqual(la.fmt_elapsed(3725), "1:02:05")


class TestStatusLine(unittest.TestCase):
    def test_以回车开头才能原地刷新(self):
        self.assertTrue(la._status(10, -20).startswith("\r"))

    def test_超长被截断_免得换行刷屏(self):
        line = la._status(10, -20, "很长的转写内容" * 40)
        self.assertLessEqual(len(line), 150)

    def test_没有文字时只显示电平(self):
        self.assertNotIn("  …", la._status(10, -20))


if __name__ == "__main__":
    unittest.main(verbosity=2)
