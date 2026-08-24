# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/_asr_mps.py 的纯逻辑测试:选设备 / 挂钩上游私有函数 / 各种失败姿势。

不需要 torch、不需要 pyannote —— 用假模块喂进去,只验决策和挂钩行为。
跑: uv run tests/test_asr_mps.py
"""
import importlib.util
import pathlib
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "asr_mps", SourceFileLoader("asr_mps", str(REPO / "bin" / "_asr_mps.py")))
asr_mps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asr_mps)


class FakePipeline:
    def __init__(self, fail=False):
        self.moved_to = None
        self.fail = fail

    def to(self, device):
        if self.fail:
            raise RuntimeError("MPS 后端不支持某个算子")
        self.moved_to = device
        return self


class FakeTorch:
    def device(self, name):
        return f"device({name})"


class FakeDiarModule:
    """冒充 mlx_qwen3_asr.diarization。"""
    def __init__(self, pipeline=None, with_hook=True):
        self._pipeline = pipeline or FakePipeline()
        if with_hook:
            self._load_pyannote_pipeline = lambda: self._pipeline


class TestPickDevice(unittest.TestCase):
    def test_有MPS就用MPS(self):
        self.assertEqual(asr_mps.pick_device({}, mps_available=True), "mps")

    def test_没有MPS退回CPU(self):
        self.assertEqual(asr_mps.pick_device({}, mps_available=False), "cpu")

    def test_环境变量优先级最高(self):
        """有 MPS 也要能强制掰回 CPU —— 出问题时得有退路。"""
        env = {"MEETING_DIARIZE_DEVICE": "cpu"}
        self.assertEqual(asr_mps.pick_device(env, mps_available=True), "cpu")

    def test_环境变量大小写和空白不敏感(self):
        env = {"MEETING_DIARIZE_DEVICE": "  MPS \n"}
        self.assertEqual(asr_mps.pick_device(env, mps_available=False), "mps")

    def test_环境变量为空串视同没设(self):
        env = {"MEETING_DIARIZE_DEVICE": "   "}
        self.assertEqual(asr_mps.pick_device(env, mps_available=True), "mps")


class TestPatchDiarization(unittest.TestCase):
    def setUp(self):
        self.warnings = []

    def _warn(self, msg):
        self.warnings.append(msg)

    def test_挂钩成功后pipeline被搬到指定设备(self):
        pipe = FakePipeline()
        mod = FakeDiarModule(pipe)
        ok = asr_mps.patch_diarization(mod, "mps", FakeTorch(), self._warn)
        self.assertTrue(ok)
        self.assertIsNone(pipe.moved_to, "不该在挂钩时就加载,应该等真正调用")

        got = mod._load_pyannote_pipeline()
        self.assertIs(got, pipe)
        self.assertEqual(pipe.moved_to, "device(mps)")
        self.assertEqual(self.warnings, [])

    def test_上游没有这个私有函数时大声告警但不崩(self):
        """上游重构掉 _load_pyannote_pipeline 时,转写不能因此挂掉 ——
        只是退回 CPU 慢速,但必须吵到能被看见。"""
        mod = FakeDiarModule(with_hook=False)
        ok = asr_mps.patch_diarization(mod, "mps", FakeTorch(), self._warn)
        self.assertFalse(ok)
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("_load_pyannote_pipeline", self.warnings[0])

    def test_搬设备失败时回落CPU并告警(self):
        """MPS 某个算子不支持 → 拿到的仍是可用的 pipeline,不是异常。"""
        pipe = FakePipeline(fail=True)
        mod = FakeDiarModule(pipe)
        ok = asr_mps.patch_diarization(mod, "mps", FakeTorch(), self._warn)
        self.assertTrue(ok)

        got = mod._load_pyannote_pipeline()
        self.assertIs(got, pipe, "搬设备失败也得把 pipeline 交出去")
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("mps", self.warnings[0])

    def test_挂钩是惰性的_只在真正加载时才搬(self):
        """上游自带缓存,包一层不能把加载提前触发。"""
        calls = []
        mod = FakeDiarModule(with_hook=False)
        pipe = FakePipeline()
        mod._load_pyannote_pipeline = lambda: (calls.append(1), pipe)[1]

        asr_mps.patch_diarization(mod, "mps", FakeTorch(), self._warn)
        self.assertEqual(calls, [], "挂钩本身不该触发加载")
        mod._load_pyannote_pipeline()
        self.assertEqual(calls, [1])


class TestNeedsPatch(unittest.TestCase):
    """不分人时不该白白 import torch(慢 2 秒)。"""

    def test_有diarize才需要挂钩(self):
        self.assertTrue(asr_mps.needs_patch(["a.m4a", "--diarize", "-f", "json"]))

    def test_没有diarize就不挂钩(self):
        self.assertFalse(asr_mps.needs_patch(["a.m4a", "-f", "json"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
