# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""菜单栏进度条与 CLI 之间的契约：两边认识的 stage key 必须一致。

从前 `meeting_app._stage_label` 是拿正则匹配 CLI 的【中文措辞】——
「已按说话人」「转写完成」「块 N/M」。CLI 改一句话，菜单栏进度就**静默哑掉**，
不报错、没人知道。今天改 `meeting` 的输出时我特意核对过没改坏，但那是运气不是设计。

现在 CLI 发 `@@STAGE <key> [detail]`，措辞随便改；这个测试守住 key 两边对齐。

跑: uv run tests/test_stage_contract.py
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
APP = REPO / "meeting_app.py"
EMITTERS = ["bin/meeting", "bin/minutes", "bin/_annotate.py", "bin/_identify.py"]


def handled_keys():
    """app 的 _STAGE_LABELS 里认识哪些 key。"""
    src = APP.read_text(encoding="utf-8")
    block = re.search(r"_STAGE_LABELS = \{(.*?)\n\}", src, re.S)
    assert block, "meeting_app.py 里找不到 _STAGE_LABELS —— 契约测试需要更新"
    return set(re.findall(r'"([a-z_]+)":\s*lambda', block.group(1)))


def emitted_keys():
    """bin/ 里真的发出了哪些 key。"""
    keys = set()
    for rel in EMITTERS:
        src = (REPO / rel).read_text(encoding="utf-8")
        keys |= set(re.findall(r"@@STAGE ([a-z_]+)", src))
    return keys


class TestStageContract(unittest.TestCase):
    def test_app认识的key都有人发(self):
        missing = handled_keys() - emitted_keys()
        self.assertEqual(missing, set(),
                         f"菜单栏在等这些 stage，但没有 CLI 会发出来：{sorted(missing)}")

    def test_发出来的key_app都认识(self):
        unknown = emitted_keys() - handled_keys()
        self.assertEqual(unknown, set(),
                         f"CLI 发了这些 stage，但菜单栏不认识、会被丢掉：{sorted(unknown)}")

    def test_关键阶段一个都不能少(self):
        """真删掉某个阶段时要显式改这里，而不是悄悄少一格进度。"""
        self.assertTrue({"transcribed", "diarized", "chunk", "done"} <= emitted_keys())


class TestStageParsing(unittest.TestCase):
    def setUp(self):
        import importlib.util
        from importlib.machinery import SourceFileLoader
        spec = importlib.util.spec_from_loader(
            "meeting_app", SourceFileLoader("meeting_app", str(APP)))
        self.app = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.app)
        except ImportError as e:              # rumps/AppKit 在 CI 上可能没有
            self.skipTest(f"菜单栏依赖不可用：{e}")

    def test_解析带细节的标记(self):
        self.assertEqual(self.app._stage_label("@@STAGE chunk 2/4"), "✍️ 2/4")
        self.assertIn("3", self.app._stage_label("@@STAGE diarized 3"))

    def test_解析不带细节的标记(self):
        self.assertEqual(self.app._stage_label("@@STAGE done"), "✅ 完成")

    def test_上游的百分比仍然认(self):
        """ASR 进度是上游 mlx-qwen3-asr 的格式，我们控制不了，只能按原样匹配。"""
        self.assertEqual(self.app._stage_label("Progress: chunk 3/406 (8.1%) ETA 05:44"),
                         "📝 8%")

    def test_普通输出不当成进度(self):
        self.assertIsNone(self.app._stage_label("✅ 纪要已生成:"))
        self.assertIsNone(self.app._stage_label("🗣  已按说话人合成: x.txt（8 人）"))

    def test_不认识的key安静忽略(self):
        self.assertIsNone(self.app._stage_label("@@STAGE 未来新增的阶段"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
