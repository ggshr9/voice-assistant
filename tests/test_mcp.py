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
    {"id": "线上会议_20260619_2301", "title": "目标用户联调", "date": "2026-06-19 23:01",
     "duration_sec": 6483, "speakers": 8, "todos": 5, "mine": 2,
     "summary": "讨论业务数据难题。", "dir": "/x", "files": {"minutes": "纪要_x.md"}},
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
        self.assertIn("unknown tool", r["result"]["content"][0]["text"])


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
