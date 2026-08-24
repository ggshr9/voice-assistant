# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""minutes 的纯逻辑测试:分块 / 去噪 / 转写文件定位 / 「我的-他人」待办归属。

全部不联网 —— 需要 LLM 的地方 stub 掉 minutes.ask,只断言编排(调了几次、
system prompt 里放了什么)。跑: uv run tests/test_minutes.py
"""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
BIN = REPO / "bin" / "minutes"

# bin/minutes 没有 .py 后缀,按源码文件显式加载(有 __main__ 守卫,导入不会跑 main)
_spec = importlib.util.spec_from_loader("minutes", SourceFileLoader("minutes", str(BIN)))
minutes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(minutes)


class TestSplitChunks(unittest.TestCase):
    def test_短文本不分块(self):
        self.assertEqual(minutes.split_chunks("很短的一段会议", size=100), ["很短的一段会议"])

    def test_超长按行切且每块不超上限(self):
        lines = [f"说话人{i%3}：" + "话" * 50 for i in range(40)]
        text = "\n".join(lines)
        chunks = minutes.split_chunks(text, size=500)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)

    def test_不切断说话人轮次(self):
        """每一行必须完整落在【某一个块】里,不能跨块被劈开。

        注意断言的写法:不能用 "".join(chunks) 去找 —— 那样拼回原文,
        硬切版也能通过,测试就成了摆设。
        """
        lines = [f"说话人{i}：这是第{i}轮完整发言内容" for i in range(30)]
        text = "\n".join(lines)
        chunks = minutes.split_chunks(text, size=200)
        for ln in lines:
            self.assertTrue(any(ln in c for c in chunks), f"轮次被切断跨块: {ln}")

    def test_单行超长时硬切(self):
        """极端情况:一行本身就超过块大小,只能硬切,但内容不能丢。"""
        text = "啊" * 500
        chunks = minutes.split_chunks(text, size=100)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).replace("\n", ""), text)

    def test_内容无损(self):
        text = "\n".join(f"说话人A：第{i}句" for i in range(50))
        chunks = minutes.split_chunks(text, size=120)
        self.assertEqual("".join(chunks).strip(), text.strip())


class TestDenoise(unittest.TestCase):
    def test_折叠重复三次以上的字(self):
        self.assertEqual(minutes._denoise("嘶嘶嘶嘶嘶嘶嘶"), "嘶")
        self.assertEqual(minutes._denoise("嗯嗯嗯"), "嗯")

    def test_重复两次保留(self):
        """「谢谢」「好好」这类正常叠词不能被吃掉。"""
        self.assertEqual(minutes._denoise("谢谢"), "谢谢")

    def test_保留说话人前缀(self):
        self.assertEqual(minutes._denoise("说话人A：嗯嗯嗯嗯对"), "说话人A：嗯对")

    def test_清完变空的行被丢掉(self):
        text = "说话人A：正常发言\n说话人B：。。。。。\n说话人C：又一句"
        out = minutes._denoise(text).split("\n")
        self.assertEqual(out, ["说话人A：正常发言", "说话人C：又一句"])

    def test_纯空白行被丢掉(self):
        self.assertEqual(minutes._denoise("有内容\n   \n还有内容"), "有内容\n还有内容")


class TestFindTranscript(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, content):
        p = self.d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_优先带说话人标注的_annotated(self):
        self._write("会议.txt", "无标注版")
        self._write("会议_annotated.txt", "说话人A：有标注版")
        text, name = minutes.find_transcript(str(self.d))
        self.assertEqual(text, "说话人A：有标注版")
        self.assertEqual(name, "会议_annotated.txt")

    def test_没有annotated时退到_zh(self):
        self._write("会议.txt", "任意版")
        self._write("会议_zh.txt", "中文版")
        text, name = minutes.find_transcript(str(self.d))
        self.assertEqual(name, "会议_zh.txt")

    def test_递归找嵌套子目录(self):
        """whisply 会把结果放进 转写_xxx/xxx/ 这样的嵌套目录。"""
        self._write("会议/深一层/会议_annotated.txt", "说话人A：藏得深")
        text, name = minutes.find_transcript(str(self.d))
        self.assertEqual(text, "说话人A：藏得深")

    def test_没有txt时退到json(self):
        self._write("会议.json", json.dumps(
            {"transcription": {"zh": {"text": "来自 json 的纯文本"}}}, ensure_ascii=False))
        text, name = minutes.find_transcript(str(self.d))
        self.assertEqual(text, "来自 json 的纯文本")
        self.assertEqual(name, "会议.json")

    def test_直接给txt文件(self):
        p = self._write("单独.txt", "  直接给的文件  ")
        text, name = minutes.find_transcript(str(p))
        self.assertEqual(text, "直接给的文件")   # 首尾空白被 strip
        self.assertEqual(name, "单独.txt")


class TestReadJson(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _json(self, obj):
        p = self.d / "t.json"
        p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def test_优先取带说话人标注的字段(self):
        fp = self._json({"transcription": {"zh": {
            "text": "没标注的",
            "text_with_speaker_annotation": "说话人A：有标注的"}}})
        self.assertEqual(minutes.read_json(fp), "说话人A：有标注的")

    def test_没标注则取纯文本(self):
        fp = self._json({"transcription": {"en": {"text": "plain text"}}})
        self.assertEqual(minutes.read_json(fp), "plain text")


class TestMakeMinutesAttribution(unittest.TestCase):
    """--me 的「我的/他人」待办归属 —— 最容易悄悄坏的地方,stub 掉 LLM 只验编排。"""

    def setUp(self):
        self.calls = []          # [(system, user, kwargs), ...]
        self._real_ask = minutes.ask
        self._real_log = minutes.log
        minutes.ask = lambda system, user, **kw: (
            self.calls.append((system, user, kw)) or "## 一句话摘要\n假的纪要")
        minutes.log = lambda *a: None

    def tearDown(self):
        minutes.ask = self._real_ask
        minutes.log = self._real_log

    def test_短会议单次生成(self):
        minutes.make_minutes("一段很短的会议转写")
        self.assertEqual(len(self.calls), 1)

    def test_me_把该说话人指认为我(self):
        minutes.make_minutes("说话人1：我来跟进这件事", me="说话人1")
        system = self.calls[0][0]
        self.assertIn("说话人1", system)
        self.assertIn("我", system)
        self.assertIn("负责人一律写 **我**", system)

    def test_不传me时不注入归属指令(self):
        minutes.make_minutes("说话人1：我来跟进这件事")
        self.assertEqual(self.calls[0][0], minutes.FINAL_SYS)

    def test_长会议走map_reduce(self):
        text = "\n".join(f"说话人{i%2}：" + "内容" * 200 for i in range(60))
        n_chunks = len(minutes.split_chunks(text))
        self.assertGreater(n_chunks, 1, "构造的样本没触发分块,测试本身失效")

        minutes.make_minutes(text, me="说话人0")
        # 每块一次记笔记 + 最后一次汇总
        self.assertEqual(len(self.calls), n_chunks + 1)

        note_calls, final_call = self.calls[:-1], self.calls[-1]
        for sys_prompt, _, _ in note_calls:
            self.assertEqual(sys_prompt, minutes.NOTE_SYS)
            self.assertNotIn("负责人一律写", sys_prompt,
                             "归属指令不该混进逐块笔记,只应出现在最终汇总")
        self.assertIn("负责人一律写 **我**", final_call[0])
        self.assertIn("【片段1】", final_call[1], "汇总的输入应该是各块笔记")


if __name__ == "__main__":
    unittest.main(verbosity=2)
