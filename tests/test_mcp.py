# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/meetings_mcp.py 的协议与工具分发测试。不联网、不碰真索引。"""
import importlib.util
import json
import pathlib
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "meetings_mcp", SourceFileLoader("meetings_mcp", str(REPO / "bin" / "meetings_mcp.py")))
mcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcp)

ENTRIES = [
    {"id": "线上会议_20260619_2301", "title": "接口联调排期", "date": "2026-06-19 23:01",
     "duration_sec": 6483, "speakers": 8, "todos": 5, "mine": 2,
     "summary": "讨论联调排期。", "dir": "/x", "files": {"minutes": "纪要_x.md"}},
    {"id": "会议_20260612_2234", "title": "声音克隆测试", "date": "2026-06-12 22:34",
     "duration_sec": 14, "speakers": 0, "todos": 1, "mine": 1,
     "summary": "测试音色复刻。", "dir": "/y", "files": {"minutes": "纪要_y.md"}},
]


class TestProtocol(unittest.TestCase):
    def test_initialize_回协议版本与服务名(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(r["result"]["serverInfo"]["name"], "meetings")
        self.assertIn("tools", r["result"]["capabilities"])

    def test_initialize_回显客户端协议版本(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")

    def test_通知类消息不回响应(self):
        self.assertIsNone(mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_ping(self):
        self.assertEqual(mcp.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})["result"], {})

    def test_tools_list_四个工具都有schema(self):
        tools = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual({t["name"] for t in tools},
                         {"list_meetings", "get_meeting", "search_meetings", "ask_meetings"})
        for t in tools:
            self.assertIn("inputSchema", t)
            self.assertTrue(t["description"].strip())

    def test_未知方法回_32601(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 3, "method": "no/such"})
        self.assertEqual(r["error"]["code"], -32601)

    def test_非对象请求回_32600(self):
        self.assertEqual(mcp.handle([1, 2, 3])["error"]["code"], -32600)

    def test_未知方法的通知不回响应(self):
        """没有 id 就是通知，按 JSON-RPC 不能回错误，否则客户端会收到孤儿响应。"""
        self.assertIsNone(mcp.handle({"jsonrpc": "2.0", "method": "no/such"}))


class TestToolErrorsAreInBand(unittest.TestCase):
    """工具报错要走 isError 的正常响应，不是 JSON-RPC error —— 否则 agent 看不到原因。"""

    def test_工具抛错时isError为真且带可读原因(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "search_meetings", "arguments": {"query": ""}}})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("不能为空", r["result"]["content"][0]["text"])

    def test_未知工具名(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                        "params": {"name": "nope", "arguments": {}}})
        self.assertTrue(r["result"]["isError"])
        txt = r["result"]["content"][0]["text"]
        self.assertIn("nope", txt, "报错要点名是哪个工具")
        self.assertIn("list_meetings", txt, "并列出可用的,让 agent 能自我纠正")


class TestFind(unittest.TestCase):
    def test_按id精确命中(self):
        self.assertEqual(mcp._find(ENTRIES, "会议_20260612_2234")["title"], "声音克隆测试")

    def test_按标题片段模糊命中(self):
        self.assertEqual(mcp._find(ENTRIES, "联调")["id"], "线上会议_20260619_2301")

    def test_按日期片段命中(self):
        self.assertEqual(mcp._find(ENTRIES, "2026-06-12")["id"], "会议_20260612_2234")

    def test_歧义时报错并列出候选(self):
        with self.assertRaises(ValueError) as cm:
            mcp._find(ENTRIES, "2026-06")
        self.assertIn("匹配到多场会", str(cm.exception))

    def test_找不到时提示怎么办(self):
        with self.assertRaises(ValueError) as cm:
            mcp._find(ENTRIES, "不存在")
        self.assertIn("list_meetings", str(cm.exception))


class TestBrief(unittest.TestCase):
    def test_时长渲染成人读格式且不漏内部字段(self):
        b = mcp._brief(ENTRIES[0])
        self.assertEqual(b["duration"], "1:48:03")
        self.assertEqual(b["my_todos"], 2)
        self.assertNotIn("dir", b)
        self.assertNotIn("files", b)

    def test_零说话人显示为空而不是0(self):
        self.assertIsNone(mcp._brief(ENTRIES[1])["speakers"])


class TestIntCoercion(unittest.TestCase):
    def test_null回落默认值(self):
        """客户端常发 "limit": null。"""
        self.assertEqual(mcp._int(None, 50), 50)
        self.assertEqual(mcp._int("", 20), 20)

    def test_正常取值(self):
        self.assertEqual(mcp._int(3, 50), 3)
        self.assertEqual(mcp._int("7", 50), 7)


class TestArgValidation(unittest.TestCase):
    """调这个接口的是 **agent**,不是人 —— 报错必须指名道姓说清哪个参数要什么。

    从前缺 id 会原样抛 KeyError,agent 只看到「错误: 'id'」,无从自我纠正;
    传数字当 query 则漏出「'int' object has no attribute 'strip'」。
    """

    def _call(self, name, args):
        r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
        return r["result"]["isError"], r["result"]["content"][0]["text"]

    def test_缺必填参数时报错里有参数名(self):
        for tool, field in (("get_meeting", "id"), ("search_meetings", "query"),
                            ("ask_meetings", "question")):
            err, msg = self._call(tool, {})
            self.assertTrue(err)
            self.assertIn(field, msg, f"{tool} 的报错没点名 {field}：{msg}")

    def test_报错里不含python内部异常措辞(self):
        LEAKS = ("object has no attribute", "invalid literal", "Traceback",
                 "KeyError", "requires string as left operand")
        cases = [("get_meeting", {}), ("search_meetings", {"query": 123}),
                 ("get_meeting", {"id": None}), ("list_meetings", {"limit": "abc"})]
        for tool, args in cases:
            _, msg = self._call(tool, args)
            for leak in LEAKS:
                self.assertNotIn(leak, msg, f"{tool}{args} 漏出内部异常：{msg}")

    def test_类型错了要明确报而不是猜(self):
        for args in ({"query": 123}, {"query": ["a"]}, {"query": {"a": 1}}):
            err, msg = self._call("search_meetings", args)
            self.assertTrue(err, f"{args} 应报错")
            self.assertIn("字符串", msg)

    def test_bool不能当整数(self):
        """limit=true 显然是笔误,但 bool 是 int 的子类,不特判就会被当成 1。"""
        err, msg = self._call("list_meetings", {"limit": True})
        self.assertTrue(err, "limit=true 应被拒")

    def test_arguments不是对象时明确报错(self):
        for bad in ([1, 2], "hello", 42):
            err, msg = self._call("list_meetings", bad)
            self.assertTrue(err, f"arguments={bad!r} 应报错")
            self.assertIn("对象", msg)

    def test_未知工具列出可用的(self):
        err, msg = self._call("no_such_tool", {})
        self.assertTrue(err)
        self.assertIn("list_meetings", msg, "应把可用工具列出来")

    def test_params不是对象时回协议级错误(self):
        r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]})
        self.assertEqual(r["error"]["code"], -32602)


class TestIntClamp(unittest.TestCase):
    """越界夹紧而不是报错 —— 对 agent 来说,limit=99999 夹到上限比整个调用失败有用。"""

    def test_超上限被夹紧(self):
        self.assertEqual(mcp._int(10 ** 9, 50), mcp.MAX_LIMIT)

    def test_负数被夹到下界(self):
        """从前 entries[:-1] 会静默丢掉最后一场会。"""
        self.assertEqual(mcp._int(-1, 50), 0)

    def test_max_chars下界是1不是0(self):
        """max_chars=0 切出空串却报 truncated,等于什么都没返回。"""
        self.assertEqual(mcp._int(0, 800, "max_chars", lo=1, hi=10 ** 6), 1)
        self.assertEqual(mcp._int(-5, 800, "max_chars", lo=1, hi=10 ** 6), 1)

    def test_浮点向下取整(self):
        self.assertEqual(mcp._int(2.9, 50), 2)


class TestListMeetingsCount(unittest.TestCase):
    """count 从前给的是匹配总数,而 meetings 只有 limit 条 —— limit=0 时
    返回 count:4 / meetings:[] 自相矛盾。"""

    def setUp(self):
        self._orig = mcp._load_entries
        mcp._load_entries = lambda: list(ENTRIES)

    def tearDown(self):
        mcp._load_entries = self._orig

    def test_count等于实际返回条数(self):
        for limit in (0, 1, 2, 99):
            r = mcp.list_meetings("", limit)
            self.assertEqual(r["count"], len(r["meetings"]),
                             f"limit={limit} 时 count 与实际条数不符")

    def test_total是匹配总数(self):
        self.assertEqual(mcp.list_meetings("", 1)["total"], 2)

    def test_截断时明确标出(self):
        self.assertTrue(mcp.list_meetings("", 1)["truncated"])
        self.assertFalse(mcp.list_meetings("", 99)["truncated"])

    def test_过滤后total反映过滤结果(self):
        r = mcp.list_meetings("声音克隆", 99)
        self.assertEqual(r["total"], 1)
        self.assertEqual(r["count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
