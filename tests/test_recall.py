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
    {"id": "a", "date": "2026-06-19 23:01", "title": "接口联调排期",
     "summary": "讨论接口联调的排期与测试环境交付。"},
    {"id": "b", "date": "2026-06-12 22:34", "title": "声音克隆测试",
     "summary": "测试声音克隆技术的音色复刻。"},
]


class TestCatalog(unittest.TestCase):
    def test_每场会一行且带编号(self):
        out = recall.format_catalog(ENTRIES)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertTrue(out.startswith("1. [2026-06-19 23:01] 接口联调排期"))

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
        picks = recall.keyword_fallback("联调排期怎么定", ENTRIES, limit=1)
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
        """契约变了:从前是「只取前 N 字」,现在是「掐中间、保两头、总量不超 N」。

        改动原因见 TestTrimMiddle —— 会议的结论在结尾,只取开头恰好把
        最该被问到的部分丢掉。这条测试原先断言 count("啊")==50,
        那是在固定旧行为;现在改成断言新契约本身。
        """
        ctx, notes = recall.build_context([self._entry("头" + "啊" * 98 + "尾")], [0],
                                          max_chars=50)
        body = ctx.split("\n", 1)[1]
        self.assertLessEqual(len(body), 50, "总量超了")
        self.assertTrue(body.startswith("头"), "开头没保住")
        self.assertTrue(body.endswith("尾"), "结尾没保住 —— 结论就在那里")
        self.assertEqual(len(notes), 1)
        self.assertIn("略去", notes[0])

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


class TestTrimMiddle(unittest.TestCase):
    """超长时掐中间、保两头。

    从前是 text[:max_chars] 只留开头 —— 但会议的结论、待办、期限几乎总在最后
    (「那就定 Redis」「九月九号之前上灰度」「散会」)。只取前半截恰好把最该被
    问到的部分丢掉,而 recall 最常见的问题正是「最后定的是什么」。
    """

    def test_不超长时原样返回(self):
        t, cut = recall.trim_middle("短文本", 1000)
        self.assertEqual(t, "短文本")
        self.assertFalse(cut)

    def test_超长时结尾必须保住(self):
        text = "头" * 100 + "中" * 50000 + "结论是采用Redis"
        t, cut = recall.trim_middle(text, 1000)
        self.assertTrue(cut)
        self.assertIn("结论是采用Redis", t, "结尾被丢掉了 —— 结论就在那里")

    def test_超长时开头也保住(self):
        """开头交代议题和背景,同样不能全丢。"""
        text = "今天过三件事" + "中" * 50000 + "散会"
        t, _ = recall.trim_middle(text, 1000)
        self.assertIn("今天过三件事", t)

    def test_不超过给定上限(self):
        for n in (200, 1000, 24000):
            t, _ = recall.trim_middle("字" * 100000, n)
            self.assertLessEqual(len(t), n, f"上限 {n} 被突破：{len(t)}")

    def test_中间有明确的略去标记(self):
        t, _ = recall.trim_middle("字" * 100000, 2000)
        self.assertIn("略去", t, "得让模型知道中间断开了,否则会把两段当连续的")

    def test_结尾分到的份额不少于开头的一半(self):
        """结论在尾部,尾巴给太少等于没修。"""
        text = "H" * 50000 + "T" * 50000
        t, _ = recall.trim_middle(text, 3000)
        head, tail = t.count("H"), t.count("T")
        self.assertGreaterEqual(tail, head // 2, f"尾部只分到 {tail},头部 {head}")

    def test_上限小到放不下标记时不崩(self):
        for n in (0, 1, 5, 20):
            t, cut = recall.trim_middle("字" * 1000, n)
            self.assertLessEqual(len(t), max(n, 0))


class TestSplitBudget(unittest.TestCase):
    """max_chars 是【所有会加起来】的上限。

    从前它按【每场】施加 —— 名字叫 MAX_CONTEXT_CHARS 却能吐出它的 N 倍
    (实测 3 场 → 72055 字),而 top 是用户可传的(`recall 10 "…"` 就是 24 万字),
    足以撑爆上下文窗口。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _entry(self, i, size):
        name = f"t{i}.txt"
        pathlib.Path(self.dir, name).write_text("字" * size, encoding="utf-8")
        return {"id": f"会{i}", "title": f"第{i}场", "dir": self.dir,
                "files": {"transcript": name}}

    def _body_len(self, ctx):
        """去掉 "### 会议：…" 标题行,只数正文。"""
        total = 0
        for block in ctx.split("\n\n### "):
            if "\n" in block:
                total += len(block.split("\n", 1)[1])
        return total

    def test_多场会合计不超总预算(self):
        for sizes in ([30000] * 3, [30000] * 10, [50000, 50000]):
            entries = [self._entry(i, n) for i, n in enumerate(sizes)]
            ctx, _ = recall.build_context(entries, list(range(len(sizes))))
            self.assertLessEqual(self._body_len(ctx), recall.MAX_CONTEXT_CHARS,
                                 f"{sizes} 突破了总预算")

    def test_短会用不完的额度让给长会(self):
        """两场一短一长时,长会不该被腰斩到均分份额。"""
        entries = [self._entry(0, 500), self._entry(1, 60000)]
        ctx, _ = recall.build_context(entries, [0, 1])
        self.assertGreater(self._body_len(ctx), recall.MAX_CONTEXT_CHARS * 0.9,
                           "预算没被用满,短会的额度浪费了")

    def test_都不超长时原样全给(self):
        entries = [self._entry(i, 100) for i in range(3)]
        ctx, notes = recall.build_context(entries, [0, 1, 2])
        self.assertEqual(notes, [], "没超长却报了截断")
        self.assertEqual(self._body_len(ctx), 300)

    def test_越界的pick被忽略而不是崩(self):
        entries = [self._entry(0, 100)]
        ctx, _ = recall.build_context(entries, [0, 5, -3])
        self.assertIn("第0场", ctx)

    def test_空picks返回空(self):
        self.assertEqual(recall.build_context([], []), ("", []))

    def test_截断说明里有原始字数(self):
        """让用户知道丢了多少,而不是只说「截断了」。"""
        entries = [self._entry(0, 60000)]
        _, notes = recall.build_context(entries, [0])
        self.assertTrue(notes)
        self.assertIn("60000", notes[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
