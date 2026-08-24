# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/recall 的纯逻辑测试:目录渲染 / 抠编号 / 关键词兜底 / 拼上下文。不联网。"""
import importlib.util
import pathlib
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader("recall", SourceFileLoader("recall", str(REPO / "bin" / "recall")))
recall = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recall)

ENTRIES = [
    {"id": "a", "date": "2026-06-19 23:01", "title": "目标用户业务数据",
     "summary": "讨论某地区目标用户的业务数据获取难题。"},
    {"id": "b", "date": "2026-06-12 22:34", "title": "声音克隆测试",
     "summary": "测试声音克隆技术的音色复刻。"},
]


class TestCatalog(unittest.TestCase):
    def test_每场会一行且带编号(self):
        out = recall.format_catalog(ENTRIES)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertTrue(out.startswith("1. [2026-06-19 23:01] 目标用户业务数据"))

    def test_没摘要也不崩(self):
        out = recall.format_catalog([{"id": "x", "title": "t"}])
        self.assertIn("(无摘要)", out)


class TestParsePicks(unittest.TestCase):
    def test_标准逗号格式(self):
        self.assertEqual(recall.parse_picks("1,3", 3), [0, 2])

    def test_大脑话多时也能抠出来(self):
        self.assertEqual(recall.parse_picks("相关的是会议 1 和 3。", 3), [0, 2])

    def test_中文顿号(self):
        self.assertEqual(recall.parse_picks("1、2", 3), [0, 1])

    def test_none表示都不相关(self):
        self.assertEqual(recall.parse_picks("none", 3), [])

    def test_越界编号被丢掉(self):
        self.assertEqual(recall.parse_picks("1,9", 2), [0])

    def test_重复编号只留一个(self):
        self.assertEqual(recall.parse_picks("2,2,1", 3), [1, 0])

    def test_空回答(self):
        self.assertEqual(recall.parse_picks("", 3), [])


class TestKeywordFallback(unittest.TestCase):
    def test_按字面重合挑出正确的会(self):
        picks = recall.keyword_fallback("业务数据怎么搞", ENTRIES, limit=1)
        self.assertEqual(picks, [0])

    def test_完全不相关时返回空(self):
        self.assertEqual(recall.keyword_fallback("烤箱温度", ENTRIES), [])


class TestBuildContext(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _entry(self, text, name="x_annotated.txt"):
        (self.d / name).write_text(text, encoding="utf-8")
        return {"id": "a", "title": "会甲", "date": "2026-06-19 23:01",
                "dir": str(self.d), "files": {"transcript": name}}

    def test_拼进标题与正文(self):
        ctx, notes = recall.build_context([self._entry("说话人A：内容")], [0])
        self.assertIn("### 会议：会甲", ctx)
        self.assertIn("说话人A：内容", ctx)
        self.assertEqual(notes, [])

    def test_超长时截断并留下说明(self):
        ctx, notes = recall.build_context([self._entry("啊" * 100)], [0], max_chars=50)
        self.assertEqual(ctx.count("啊"), 50)
        self.assertEqual(len(notes), 1)
        self.assertIn("只取了前 50 字", notes[0])

    def test_没有转写就退到纪要(self):
        (self.d / "纪要_x.md").write_text("# 纪要正文", encoding="utf-8")
        e = {"id": "a", "title": "会甲", "dir": str(self.d),
             "files": {"transcript": "", "minutes": "纪要_x.md"}}
        ctx, _ = recall.build_context([e], [0])
        self.assertIn("# 纪要正文", ctx)

    def test_文件不存在时跳过而不是崩(self):
        e = {"id": "a", "title": "会甲", "dir": str(self.d),
             "files": {"transcript": "不存在.txt"}}
        ctx, _ = recall.build_context([e], [0])
        self.assertEqual(ctx, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
