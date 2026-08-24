# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/_index.py 的纯逻辑测试:解析纪要 / 取标题 / 认日期 / 去重合并 / 渲染。

不碰文件系统、不联网。跑: uv run tests/test_index.py
"""
import importlib.util
import pathlib
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "meetindex", SourceFileLoader("meetindex", str(REPO / "bin" / "_index.py")))
meetindex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(meetindex)


MINUTES_MD = """# 会议纪要 · 线上会议_20260619_2301

> 来源转写：`x_annotated.txt`

## 一句话摘要
会议演示了基于某通讯平台的注册流程，并讨论了某地区目标用户的业务数据难题。

## 关键决议
- 甲
- 乙

## 待办事项
| 事项 | 负责人 | 期限 |
| :--- | :--- | :--- |
| 确认支持的语种列表 | — | — |
| 调研第三方数据商 | **我** | — |
| 确认账单导出格式 | 说话人B | — |

## 讨论要点
略
"""


class TestParseMinutes(unittest.TestCase):
    def test_取出一句话摘要(self):
        got = meetindex.parse_minutes(MINUTES_MD)
        self.assertTrue(got["summary"].startswith("会议演示了基于某通讯平台"))
        self.assertNotIn("##", got["summary"])

    def test_数待办条数_不把表头算进去(self):
        self.assertEqual(meetindex.parse_minutes(MINUTES_MD)["todos"], 3)

    def test_数我的待办(self):
        self.assertEqual(meetindex.parse_minutes(MINUTES_MD)["mine"], 1)

    def test_没有摘要段时不炸(self):
        got = meetindex.parse_minutes("# 标题\n\n## 讨论要点\n略\n")
        self.assertEqual(got["summary"], "")
        self.assertEqual(got["todos"], 0)


class TestDeriveTitle(unittest.TestCase):
    def test_从摘要切出短标题(self):
        t = meetindex.derive_title("会议演示了基于某通讯平台的注册流程，并讨论了联调难题。", limit=20)
        self.assertLessEqual(len(t), 20)
        self.assertTrue(t.startswith("会议演示了"))
        self.assertNotIn("。", t)

    def test_摘要为空时回退到会议标识(self):
        self.assertEqual(meetindex.derive_title("", fallback="线上会议_20260619_2301"),
                         "线上会议_20260619_2301")

    def test_短摘要原样返回(self):
        self.assertEqual(meetindex.derive_title("周会", limit=20), "周会")


class TestLLMTitle(unittest.TestCase):
    """标题交给 LLM 起,derive_title 只是没大脑时的兜底。

    截断摘要当标题的效果很差 —— 回填时真实产出是
    「会议演示了基于某通讯平台的新用户注册及小组」这种半截话。
    """

    def test_用llm起标题并清掉多余包装(self):
        got = meetindex.llm_title("摘要", ask=lambda sys_p, user: '「接口联调排期评审」\n')
        self.assertEqual(got, "接口联调排期评审")

    def test_llm返回空时回退(self):
        got = meetindex.llm_title("会议讨论了联调。", ask=lambda s, u: "  ",
                                  fallback="会议_x")
        self.assertEqual(got, "会议讨论了联调")

    def test_llm抛错时不影响索引(self):
        def boom(s, u):
            raise RuntimeError("网关挂了")
        got = meetindex.llm_title("会议讨论了联调。", ask=boom, fallback="会议_x")
        self.assertEqual(got, "会议讨论了联调")

    def test_llm话太多时截断(self):
        got = meetindex.llm_title("摘要", ask=lambda s, u: "这是一个非常冗长的标题" * 5)
        self.assertLessEqual(len(got), 24)


class TestParseDate(unittest.TestCase):
    def test_从名字里认出日期时间(self):
        self.assertEqual(meetindex.parse_date("线上会议_20260619_2301"), "2026-06-19 23:01")

    def test_另一种前缀(self):
        self.assertEqual(meetindex.parse_date("会议_20260612_2234"), "2026-06-12 22:34")

    def test_认不出时返回空(self):
        self.assertEqual(meetindex.parse_date("随手录的东西"), "")


class TestUpsert(unittest.TestCase):
    def _e(self, mid, date, title="t"):
        return {"id": mid, "date": date, "title": title}

    def test_新条目按日期倒序插入(self):
        entries = [self._e("a", "2026-06-12 22:34")]
        out = meetindex.upsert(entries, self._e("b", "2026-06-19 23:01"))
        self.assertEqual([e["id"] for e in out], ["b", "a"])

    def test_同一个会重跑时是覆盖不是重复(self):
        entries = [self._e("a", "2026-06-12 22:34", title="旧")]
        out = meetindex.upsert(entries, self._e("a", "2026-06-12 22:34", title="新"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "新")

    def test_没日期的排在最后(self):
        entries = [self._e("a", "2026-06-12 22:34")]
        out = meetindex.upsert(entries, self._e("z", ""))
        self.assertEqual([e["id"] for e in out], ["a", "z"])


class TestRender(unittest.TestCase):
    def test_渲染成表格且含关键字段(self):
        entries = [{
            "id": "线上会议_20260619_2301", "title": "接口联调排期",
            "date": "2026-06-19 23:01", "duration_sec": 6483, "speakers": 8,
            "todos": 5, "mine": 2, "summary": "讨论了业务数据难题。",
            "dir": "/x/转写_线上会议_20260619_2301",
        }]
        md = meetindex.render_markdown(entries)
        self.assertIn("接口联调排期", md)
        self.assertIn("2026-06-19 23:01", md)
        self.assertIn("1:48:03", md)       # 6483 秒
        self.assertIn("8", md)
        self.assertIn("讨论了业务数据难题。", md)

    def test_空索引也能渲染(self):
        self.assertIn("暂无", meetindex.render_markdown([]))


class TestFormatDuration(unittest.TestCase):
    """按媒体时长惯例:超过 1 小时才带小时位,短会议要能看见秒。

    一场 14 秒的测试录音显示成「0:00」是错的 —— 回填现有会议时真的这样了。
    """

    def test_超过一小时带小时位(self):
        self.assertEqual(meetindex.format_duration(6483), "1:48:03")

    def test_不足一小时是分秒(self):
        self.assertEqual(meetindex.format_duration(125), "2:05")

    def test_十几秒也看得见(self):
        self.assertEqual(meetindex.format_duration(14), "0:14")

    def test_未知时长(self):
        self.assertEqual(meetindex.format_duration(0), "—")


if __name__ == "__main__":
    unittest.main(verbosity=2)
