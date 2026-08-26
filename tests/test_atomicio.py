# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""bin/_atomicio.py：全项目共用的原子写。

这个项目已经在同一个错误上栽过三次 —— 录音 m4a、会议索引、声纹库，
模式完全一样：**截断在先、内容在后**。这份测试是那三次的收口。

跑: uv run tests/test_atomicio.py
"""
import importlib.util
import os
import pathlib
import stat
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "atomicio", SourceFileLoader("atomicio", str(REPO / "bin" / "_atomicio.py")))
aio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aio)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.d = self._t.name
        self.p = os.path.join(self.d, "target.txt")

    def tearDown(self):
        self._t.cleanup()

    def _leftovers(self):
        return [n for n in os.listdir(self.d) if n.startswith(".tmp-") or n.endswith(".part")]


class TestBasics(_Tmp):
    def test_写入并读回(self):
        aio.atomic_write(self.p, "内容")
        self.assertEqual(pathlib.Path(self.p).read_text(encoding="utf-8"), "内容")

    def test_覆写(self):
        aio.atomic_write(self.p, "旧")
        aio.atomic_write(self.p, "新")
        self.assertEqual(pathlib.Path(self.p).read_text(encoding="utf-8"), "新")

    def test_目录不存在时自动建(self):
        deep = os.path.join(self.d, "a", "b", "c.txt")
        aio.atomic_write(deep, "x")
        self.assertTrue(os.path.exists(deep))

    def test_空内容也能写(self):
        aio.atomic_write(self.p, "")
        self.assertEqual(pathlib.Path(self.p).read_text(encoding="utf-8"), "")

    def test_返回路径(self):
        self.assertEqual(aio.atomic_write(self.p, "x"), self.p)


class TestAtomicity(_Tmp):
    """崩溃时要么旧内容、要么新内容，没有中间态。"""

    def _boom_at_fsync(self):
        orig = aio.os.fsync
        aio.os.fsync = lambda *a: (_ for _ in ()).throw(OSError("模拟断电"))
        return orig

    def test_写入中途崩溃时原文件分毫不动(self):
        aio.atomic_write(self.p, "原内容" * 500)
        before = pathlib.Path(self.p).read_bytes()
        orig = self._boom_at_fsync()
        try:
            with self.assertRaises(OSError):
                aio.atomic_write(self.p, "新内容")
        finally:
            aio.os.fsync = orig
        self.assertEqual(pathlib.Path(self.p).read_bytes(), before, "原文件被破坏了")

    def test_崩溃后不留临时文件(self):
        orig = self._boom_at_fsync()
        try:
            with self.assertRaises(OSError):
                aio.atomic_write(self.p, "x")
        finally:
            aio.os.fsync = orig
        self.assertEqual(self._leftovers(), [], "崩溃后有残留")

    def test_目标不存在时崩溃也不留半截文件(self):
        """新建场景:半截文件看起来跟完整的一样,比没有更糟。"""
        orig = self._boom_at_fsync()
        try:
            with self.assertRaises(OSError):
                aio.atomic_write(self.p, "x")
        finally:
            aio.os.fsync = orig
        self.assertFalse(os.path.exists(self.p), "留下了半截的新文件")

    def test_正常写入不留临时文件(self):
        aio.atomic_write(self.p, "x")
        self.assertEqual(self._leftovers(), [])


class TestPermissions(_Tmp):
    def test_沿用目标文件现有权限(self):
        """别把原文件权限悄悄改掉 —— mkstemp 建的是 0600,直接 replace 会把
        644 的文件变成 600(索引就这样被悄悄改过)。

        这里刻意用 0o640 而不是 0o644:644 恰好等于 DEFAULT_MODE,
        断言会因为巧合而通过 —— 第一版就是这么写的,把"不沿用现有权限"的
        变异整个放了过去。挑一个和默认值不同的模式,断言才有区分度。
        """
        aio.atomic_write(self.p, "x")
        os.chmod(self.p, 0o640)
        self.assertNotEqual(0o640, aio.DEFAULT_MODE, "测试用的模式必须区别于默认值")
        aio.atomic_write(self.p, "y")
        self.assertEqual(stat.S_IMODE(os.stat(self.p).st_mode), 0o640)

    def test_沿用现有权限时不受默认值影响(self):
        """再取一个方向相反的:比默认更宽松。"""
        aio.atomic_write(self.p, "x")
        os.chmod(self.p, 0o664)
        aio.atomic_write(self.p, "y")
        self.assertEqual(stat.S_IMODE(os.stat(self.p).st_mode), 0o664)

    def test_新文件默认644(self):
        aio.atomic_write(self.p, "x")
        self.assertEqual(stat.S_IMODE(os.stat(self.p).st_mode), aio.DEFAULT_MODE)

    def test_显式模式优先于现有权限(self):
        aio.atomic_write(self.p, "x")
        os.chmod(self.p, 0o640)
        aio.atomic_write(self.p, "y", mode=0o600)
        self.assertEqual(stat.S_IMODE(os.stat(self.p).st_mode), 0o600)

    def test_密钥文件从来没有敞开的窗口(self):
        """这是"写完再 chmod"最要命的地方:那之间文件按 umask(通常 644)躺着。

        这里用一个在 fsync 时窥视的钩子 —— 此刻内容已落到临时文件上,
        正是旧写法里"内容在、权限还没收紧"的那一刻。
        """
        seen = []
        orig = aio.os.fsync

        def peek(fd):
            for n in os.listdir(self.d):
                if n.startswith(".tmp-"):
                    seen.append(stat.S_IMODE(os.stat(os.path.join(self.d, n)).st_mode))
            return orig(fd)

        aio.os.fsync = peek
        try:
            aio.atomic_write(self.p, "HF_TOKEN=hf_secret", mode=0o600)
        finally:
            aio.os.fsync = orig
        self.assertTrue(seen, "没观察到临时文件")
        for m in seen:
            self.assertEqual(m & 0o077, 0, f"临时文件一度是 {oct(m)}，别人能读到 token")
        self.assertEqual(stat.S_IMODE(os.stat(self.p).st_mode), 0o600)


class TestContent(_Tmp):
    def test_中文与emoji(self):
        text = "会议纪要 · 你好 🎙 café ある"
        aio.atomic_write(self.p, text)
        self.assertEqual(pathlib.Path(self.p).read_text(encoding="utf-8"), text)

    def test_大文件(self):
        text = "行\n" * 200000
        aio.atomic_write(self.p, text)
        self.assertEqual(len(pathlib.Path(self.p).read_text(encoding="utf-8")), len(text))

    def test_路径含空格与中文(self):
        p = os.path.join(self.d, "会议 录音", "纪要 a.md")
        aio.atomic_write(p, "x")
        self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main(verbosity=2)
