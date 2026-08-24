# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/_annotate.py:把词级 segment 合成带说话人标注的稿子。

核心是拼接规则 —— 分人会强制开词级时间戳,segment 是一个词一个词的。
中文直接拼没问题,英文直接拼就糊成一坨(生产数据里真的发生了)。
跑: uv run tests/test_annotate.py
"""
import importlib.util
import json
import pathlib
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "annotate", SourceFileLoader("annotate", str(REPO / "bin" / "_annotate.py")))
annotate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(annotate)


class TestNeedsSpace(unittest.TestCase):
    def test_两个英文词之间要空格(self):
        """这就是那个 bug:So+let's 拼成 Solet's。"""
        self.assertTrue(annotate.needs_space("So", "let's"))

    def test_标点后接英文也要空格(self):
        self.assertTrue(annotate.needs_space("loans,", "or"))

    def test_中文之间不加空格(self):
        self.assertFalse(annotate.needs_space("你好", "世界"))

    def test_中文标点后接英文不加空格(self):
        """跟 ASR 自己产出的纯 txt 风格一致:「呃，OK」不是「呃， OK」。"""
        self.assertFalse(annotate.needs_space("呃，", "OK"))

    def test_中文后接英文不加空格(self):
        self.assertFalse(annotate.needs_space("嗯", "OK"))

    def test_英文后接中文不加空格(self):
        self.assertFalse(annotate.needs_space("Yeah", "对"))

    def test_已有空格不重复加(self):
        self.assertFalse(annotate.needs_space("So ", "let's"))
        self.assertFalse(annotate.needs_space("So", " let's"))

    def test_撇号不该被拆开(self):
        """let + 's 必须是 let's,不能是 let 's。"""
        self.assertFalse(annotate.needs_space("let", "'s"))

    def test_空串不加空格(self):
        self.assertFalse(annotate.needs_space("", "word"))
        self.assertFalse(annotate.needs_space("word", ""))


class TestBuildTurns(unittest.TestCase):
    def test_连续同一说话人合并成一轮(self):
        segs = [{"speaker": "SPEAKER_00", "text": "So"},
                {"speaker": "SPEAKER_00", "text": "let's"},
                {"speaker": "SPEAKER_00", "text": "say"}]
        self.assertEqual(annotate.build_turns(segs), [("SPEAKER_00", "So let's say")])

    def test_换人就切一轮(self):
        segs = [{"speaker": "SPEAKER_00", "text": "你好"},
                {"speaker": "SPEAKER_01", "text": "世界"},
                {"speaker": "SPEAKER_00", "text": "再见"}]
        self.assertEqual(annotate.build_turns(segs),
                         [("SPEAKER_00", "你好"), ("SPEAKER_01", "世界"), ("SPEAKER_00", "再见")])

    def test_中英混说各按各的规则拼(self):
        segs = [{"speaker": "A", "text": "呃，"}, {"speaker": "A", "text": "OK"},
                {"speaker": "A", "text": "I"}, {"speaker": "A", "text": "think"},
                {"speaker": "A", "text": "对"}]
        self.assertEqual(annotate.build_turns(segs), [("A", "呃，OK I think对")])

    def test_没有speaker字段时兜底(self):
        segs = [{"text": "孤儿段"}]
        turns = annotate.build_turns(segs)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0][1], "孤儿段")


class TestSpeakerNames(unittest.TestCase):
    def test_按出场顺序映射成字母(self):
        turns = [("SPEAKER_03", "a"), ("SPEAKER_00", "b"), ("SPEAKER_03", "c")]
        self.assertEqual(annotate.speaker_names(turns),
                         {"SPEAKER_03": "说话人A", "SPEAKER_00": "说话人B"})

    def test_超过26人用数字(self):
        turns = [(f"S{i}", "x") for i in range(28)]
        names = annotate.speaker_names(turns)
        self.assertEqual(names["S0"], "说话人A")
        self.assertEqual(names["S25"], "说话人Z")
        self.assertEqual(names["S26"], "说话人27")


class TestWriteAnnotated(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _json(self, segs):
        p = self.d / "会议.json"
        p.write_text(json.dumps({"segments": segs}, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def test_多人时写出标注稿(self):
        fp = self._json([{"speaker": "SPEAKER_00", "text": "So"},
                         {"speaker": "SPEAKER_00", "text": "let's"},
                         {"speaker": "SPEAKER_01", "text": "好的"}])
        out = annotate.write_annotated(fp, str(self.d))
        self.assertIsNotNone(out)
        self.assertEqual(pathlib.Path(out).read_text(encoding="utf-8"),
                         "说话人A：So let's\n说话人B：好的\n")

    def test_只有一个说话人时不写(self):
        """单人时纯 txt 带标点更好读,不该被 annotated 抢走优先级。"""
        fp = self._json([{"speaker": "SPEAKER_00", "text": "一个人自言自语"}])
        self.assertIsNone(annotate.write_annotated(fp, str(self.d)))
        self.assertEqual(list(self.d.glob("*_annotated.txt")), [])

    def test_没有speaker标签时不写(self):
        fp = self._json([{"text": "没分人"}])
        self.assertIsNone(annotate.write_annotated(fp, str(self.d)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
