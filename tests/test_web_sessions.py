# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""服务器侧 web/sessions.py 与 web/jobs.py 的纯逻辑测试。

**为什么值得单独写**：这半边一直零覆盖，而它是**线上跑着的**（对外服务、带访问口令）。
GPU/CUDA 相关的部分测不了，但会话目录、口令校验、留存策略、任务表清理都是纯逻辑，
在 macOS 上就能测 —— 尤其是 `sess_dir` 的路径穿越防护，那是公开服务上最不能回归的一条。

跑: uv run tests/test_web_sessions.py
"""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
WEB = REPO / "web"


def load_sessions(sessions_dir, access_pw=""):
    """装载 sessions.py，把它依赖的 config 换成假的（真 config 会去建目录、读 env）。"""
    fake = types.ModuleType("config")
    fake.SESSIONS = str(sessions_dir)
    fake.ACCESS_PW = access_pw
    fake.RETENTION_DAYS = 0
    fake.SESSIONS_WARN_GB = 50.0
    fake.SR = 16000
    fake.HERE = str(WEB)
    fake.MODEL = "fake"
    fake.LLM_URL = fake.LLM_KEY = fake.LLM_MODEL = ""
    sys.modules["config"] = fake

    spec = importlib.util.spec_from_file_location("wsessions", WEB / "sessions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # 必须 realpath：macOS 的 tempfile 给的是 /var/...，而 /var 本身是
        # 指向 /private/var 的软链。不化解的话，「去掉 realpath」这种变异会先
        # 让【正常目录】那条测试挂掉，从而掩盖真正该抓的「软链越权」那条 ——
        # 一个断言替另一个断言背了锅，覆盖就成了假的。
        self.root = pathlib.Path(os.path.realpath(self._tmp.name)) / "sessions"
        self.root.mkdir()
        self.s = load_sessions(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _session(self, name, started=None):
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        meta = {"id": name, "started": started if started is not None else time.time()}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return d


class TestPathTraversal(_Base):
    """sid 来自 HTTP 请求 —— 这是唯一一处外部输入直接参与拼路径的地方。"""

    def test_正常会话目录能取到(self):
        self._session("20260825_1030_会议_ab12")
        self.assertIsNotNone(self.s.sess_dir("20260825_1030_会议_ab12"))

    def test_上跳被拦住(self):
        for evil in ("../", "../../etc", "..", "a/../../..", "../" * 8 + "etc/passwd"):
            self.assertIsNone(self.s.sess_dir(evil), f"没拦住：{evil}")

    def test_绝对路径被拦住(self):
        self.assertIsNone(self.s.sess_dir("/etc"))
        self.assertIsNone(self.s.sess_dir("/tmp"))

    def test_软链指向外部被拦住(self):
        """realpath 之后才判断前缀 —— 否则软链能绕过去。"""
        outside = pathlib.Path(os.path.realpath(self._tmp.name)) / "outside"
        outside.mkdir()
        try:
            os.symlink(outside, self.root / "escape")
        except OSError:
            self.skipTest("这个文件系统建不了软链")
        self.assertIsNone(self.s.sess_dir("escape"))

    def test_空值与不存在(self):
        self.assertIsNone(self.s.sess_dir(""))
        self.assertIsNone(self.s.sess_dir(None))
        self.assertIsNone(self.s.sess_dir("查无此会"))

    def test_前缀相同的兄弟目录不算在内(self):
        """SESSIONS 是 /x/sessions 时，/x/sessions-evil 不该被当成合法。"""
        sibling = self.root.parent / (self.root.name + "-evil")
        sibling.mkdir()
        self.assertIsNone(self.s.sess_dir("../" + sibling.name))


class TestCheckPw(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_没设口令时一律放行(self):
        s = load_sessions(self.root, access_pw="")
        self.assertTrue(s.check_pw(""))
        self.assertTrue(s.check_pw("随便"))

    def test_设了口令时只认对的(self):
        s = load_sessions(self.root, access_pw="secret")
        self.assertTrue(s.check_pw("secret"))
        self.assertFalse(s.check_pw("Secret"), "口令应区分大小写")
        self.assertFalse(s.check_pw(""))
        self.assertFalse(s.check_pw(None))


class TestSlug(_Base):
    def test_中文标题保留(self):
        self.assertEqual(self.s._slug("接口联调排期"), "接口联调排期")

    def test_剔掉路径分隔符与特殊字符(self):
        """slug 会进目录名 —— 斜杠和点必须清掉。"""
        out = self.s._slug("a/b\\c:d*e?f")
        for ch in "/\\:*?":
            self.assertNotIn(ch, out)

    def test_空标题有兜底(self):
        self.assertEqual(self.s._slug(""), "会议")
        self.assertEqual(self.s._slug("   "), "会议")
        self.assertEqual(self.s._slug(None), "会议")

    def test_超长被截断(self):
        self.assertLessEqual(len(self.s._slug("很长的标题" * 20)), 24)


class TestPruneOldAudio(_Base):
    def _with_audio(self, name, days_ago):
        d = self._session(name, started=time.time() - days_ago * 86400)
        (d / "recording.opus").write_bytes(b"x")
        (d / "audio.wav").write_bytes(b"x")
        (d / "纪要.md").write_text("纪要正文", encoding="utf-8")
        return d

    def test_关闭时不删任何东西(self):
        d = self._with_audio("old", 99)
        self.assertEqual(self.s.prune_old_audio(0), 0)
        self.assertTrue((d / "recording.opus").exists())

    def test_超期的录音被删(self):
        d = self._with_audio("old", 30)
        n = self.s.prune_old_audio(7)
        self.assertEqual(n, 2)
        self.assertFalse((d / "recording.opus").exists())
        self.assertFalse((d / "audio.wav").exists())

    def test_只删录音_文字与纪要保留(self):
        """留存策略是省磁盘，不是销毁记录 —— 删错了就永久丢内容。"""
        d = self._with_audio("old", 30)
        self.s.prune_old_audio(7)
        self.assertTrue((d / "纪要.md").exists(), "纪要不该被删")
        self.assertTrue((d / "meta.json").exists(), "元数据不该被删")

    def test_未超期的不动(self):
        d = self._with_audio("fresh", 1)
        self.assertEqual(self.s.prune_old_audio(7), 0)
        self.assertTrue((d / "recording.opus").exists())

    def test_没有meta的目录被跳过而不是崩(self):
        (self.root / "垃圾目录").mkdir()
        self.assertEqual(self.s.prune_old_audio(7), 0)


def load_jobs(tmp):
    """jobs.py 依赖 config 和 sessions，都得先就位。"""
    load_sessions(tmp)                       # 顺带把假 config 装进 sys.modules
    spec = importlib.util.spec_from_file_location("wsessions_for_jobs", WEB / "sessions.py")
    sess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sess)
    sys.modules["sessions"] = sess

    spec = importlib.util.spec_from_file_location("wjobs", WEB / "jobs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPruneJobs(unittest.TestCase):
    """JOBS 是进程内字典，不清就会无限涨 —— 这是个长跑服务。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.j = load_jobs(pathlib.Path(self._tmp.name))
        self.j.JOBS.clear()

    def tearDown(self):
        self._tmp.cleanup()

    def test_没超上限时不动(self):
        for i in range(5):
            self.j.JOBS[f"j{i}"] = {"done": True}
        self.j.prune_jobs(cap=60)
        self.assertEqual(len(self.j.JOBS), 5)

    def test_超上限时清掉已完成的旧任务(self):
        for i in range(70):
            self.j.JOBS[f"j{i}"] = {"done": True}
        self.j.prune_jobs(cap=60)
        self.assertEqual(len(self.j.JOBS), 60)

    def test_未完成的任务不能被清掉(self):
        """正在跑的任务被清掉，前端就永远轮询不到结果了。"""
        for i in range(70):
            self.j.JOBS[f"j{i}"] = {"done": False}
        self.j.prune_jobs(cap=60)
        self.assertEqual(len(self.j.JOBS), 70, "未完成的一个都不该删")

    def test_混合时只清已完成的(self):
        for i in range(65):
            self.j.JOBS[f"done{i}"] = {"done": True}
        for i in range(5):
            self.j.JOBS[f"live{i}"] = {"done": False}
        self.j.prune_jobs(cap=60)
        for i in range(5):
            self.assertIn(f"live{i}", self.j.JOBS)


class TestUploadAuthOrder(unittest.TestCase):
    """上传必须先验口令再落盘 —— 这是源码层面的不变量,单测 check_pw 抓不到。

    从前 `session_upload` 把整个文件流式写完才 `check_pw`,实测口令错误的请求
    照样在磁盘上留下 789KB。任何能连到服务的人都能不带口令写满盘。
    """

    def setUp(self):
        self.src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        i = self.src.index("async def session_upload")
        self.body = self.src[i:self.src.index("\nasync def", i + 10)]

    def test_写文件之前先判过口令(self):
        auth = self.body.index("authed")
        write = self.body.index('open(audio, "wb")')
        self.assertLess(auth, write, "口令校验必须排在打开文件之前")

    def test_收到音频而未鉴权时直接拒绝(self):
        seg = self.body[self.body.index('field.name == "audio"'):]
        gate = seg[:seg.index('open(audio, "wb")')]
        self.assertIn("not authed", gate, "音频分支开头必须拦一道")
        self.assertIn("403", gate)

    def test_有大小上限(self):
        """没有上限的话,一个请求就能把盘写满。"""
        self.assertIn("MAX_UPLOAD_BYTES", self.body)
        self.assertIn("413", self.body, "超限应回 413 而不是静默截断")

    def test_超限时清掉半截文件(self):
        self.assertIn("rmtree", self.body, "超限要把已写的半截目录删掉")

    def test_前端把pw排在audio之前(self):
        """服务端按到达顺序鉴权,前端顺序反了就永远传不上。"""
        html = (WEB / "index.html").read_text(encoding="utf-8")
        i = html.index("'/session/upload'")
        form = html[max(0, i - 400):i]
        self.assertLess(form.index("append('pw'"), form.index("append('audio'"),
                        "前端必须先 append('pw') 再 append('audio')")


if __name__ == "__main__":
    unittest.main(verbosity=2)
