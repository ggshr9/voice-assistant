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

class TestIdentifiedNames(unittest.TestCase):
    """声纹认出来的人用真名，认不出的仍是匿名牌。"""

    def test_认出来的用真名(self):
        turns = [("SPEAKER_00", "a"), ("SPEAKER_01", "b")]
        names = annotate.speaker_names(turns, {"SPEAKER_01": "李四"})
        self.assertEqual(names, {"SPEAKER_00": "说话人A", "SPEAKER_01": "李四"})

    def test_匿名牌字母只在未识别者之间顺序分配(self):
        """认出来的人不占字母 —— 否则一场会里会出现「李四 / 说话人B」这种跳号。"""
        turns = [("S0", "a"), ("S1", "b"), ("S2", "c")]
        names = annotate.speaker_names(turns, {"S1": "李四"})
        self.assertEqual(names["S0"], "说话人A")
        self.assertEqual(names["S1"], "李四")
        self.assertEqual(names["S2"], "说话人B")

    def test_全部认出来时没有匿名牌(self):
        turns = [("S0", "a"), ("S1", "b")]
        names = annotate.speaker_names(turns, {"S0": "甲", "S1": "乙"})
        self.assertEqual(set(names.values()), {"甲", "乙"})

    def test_不传映射时行为不变(self):
        turns = [("S0", "a"), ("S1", "b")]
        self.assertEqual(annotate.speaker_names(turns),
                         {"S0": "说话人A", "S1": "说话人B"})

    def test_写出的稿子里是真名(self):
        import json as _json, tempfile, pathlib as _p
        with tempfile.TemporaryDirectory() as d:
            fp = _p.Path(d) / "会.json"
            fp.write_text(_json.dumps({"segments": [
                {"speaker": "SPEAKER_00", "text": "开始吧"},
                {"speaker": "SPEAKER_01", "text": "好的"}]}, ensure_ascii=False),
                encoding="utf-8")
            out = annotate.write_annotated(str(fp), d, {"SPEAKER_00": "李四"})
            text = _p.Path(out).read_text(encoding="utf-8")
            self.assertIn("李四：开始吧", text)
            self.assertIn("说话人A：好的", text)

class TestNamesArgParsing(unittest.TestCase):
    """`--names` 现在带置信度：`SPEAKER_00=张三:0.94`。

    勉强够线的标 `[?]`，与逐字记录的存疑标注同一套约定 —— 否则 0.56 和 0.94
    在稿子里长得一模一样，读的人没法判断该不该信这个名字。
    """

    def _run(self, names):
        import json as _json
        import subprocess
        import tempfile
        repo = pathlib.Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as d:
            js = pathlib.Path(d) / "会.json"
            js.write_text(_json.dumps({"segments": [
                {"speaker": "SPEAKER_00", "text": "开始吧"},
                {"speaker": "SPEAKER_01", "text": "好的"},
                {"speaker": "SPEAKER_02", "text": "收到"}]}, ensure_ascii=False),
                encoding="utf-8")
            cmd = ["python3", str(repo / "bin/_annotate.py"), str(js), d]
            if names:
                cmd += ["--names", names]
            subprocess.run(cmd, capture_output=True)
            return list(pathlib.Path(d).glob("*_annotated.txt"))[0].read_text(encoding="utf-8")

    def test_高置信用干净真名(self):
        self.assertIn("张三：开始吧", self._run("SPEAKER_00=张三:0.94"))

    def test_勉强够线标问号(self):
        self.assertIn("李四[?]：好的", self._run("SPEAKER_01=李四:0.61"))

    def test_认不出的仍是匿名牌(self):
        self.assertIn("说话人A：收到", self._run("SPEAKER_00=张三:0.94,SPEAKER_01=李四:0.61"))

    def test_不带分数时向后兼容(self):
        """老格式（只有名字）不能因为加了分数就失效。"""
        self.assertIn("张三：开始吧", self._run("SPEAKER_00=张三"))

    def test_分数是坏值时原样当名字用_不崩(self):
        self.assertIn("张三:abc：开始吧", self._run("SPEAKER_00=张三:abc"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
