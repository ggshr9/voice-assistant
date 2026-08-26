# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""服务器侧 meeting_pipeline / jobs 的纯逻辑测试。

**为什么补**:`meeting_pipeline.py` 此前零覆盖,里面藏着一个把 ffmpeg 参数插错位的
bug —— `ff[5:5] = ["-ac","1"]` 插进了 `-ar` 和 `16000` 中间,生成 `-ar -ac 1 16000`。
它只在【非线上】场景触发,所以线上会议一路正常,而**每一次上传都必然失败**,
活了两个月。同期 `jobs.py` 把子进程输出按关键词过滤,traceback 全丢,
失败只报一句"处理未产出结果" —— 两个缺陷叠加,等于故障不可诊断。

跑: uv run tests/test_web_pipeline.py
"""
import os
import importlib.util
import pathlib
import sys
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
WEB = REPO / "web"


def _load(name, path, fake_deps=()):
    for dep in fake_deps:
        sys.modules.setdefault(dep, types.ModuleType(dep))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_pipeline():
    """meeting_pipeline 顶层会 import minutes_lib,给个空壳顶掉。"""
    sys.modules["minutes_lib"] = types.ModuleType("minutes_lib")
    return _load("wpipeline", WEB / "meeting" / "meeting_pipeline.py")


class TestToWavCmd(unittest.TestCase):
    """ffmpeg 参数顺序 —— 这正是坏了两个月的地方。"""

    def setUp(self):
        self.p = load_pipeline()

    def _pairs(self, cmd):
        """把 ['-ar','16000','-ac','1'] 这类选项收成 {选项: 值}。"""
        return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("-")}

    def test_非线上_采样率和声道都取到正确的值(self):
        cmd = self.p.to_wav_cmd("in.m4a", "out.wav", "0")
        opts = self._pairs(cmd)
        self.assertEqual(opts.get("-ar"), "16000", f"-ar 拿错了值：{cmd}")
        self.assertEqual(opts.get("-ac"), "1", f"-ac 拿错了值：{cmd}")

    def test_没有任何选项把另一个选项当成值(self):
        """`-ar -ac` 就是原来的症状:一个选项的值位上是另一个选项。"""
        for roles in ("0", "1"):
            cmd = self.p.to_wav_cmd("in.m4a", "out.wav", roles)
            for i, tok in enumerate(cmd[:-1]):
                if tok in ("-ar", "-ac", "-i"):
                    self.assertFalse(cmd[i + 1].startswith("-"),
                                     f"{tok} 的值位上是 {cmd[i+1]}：{cmd}")

    def test_线上场景不下混声道(self):
        """线上是双声道 L=对方/R=我,下混就没法按声道分人了。"""
        cmd = self.p.to_wav_cmd("in.m4a", "out.wav", "1")
        self.assertNotIn("-ac", cmd)

    def test_输入输出路径都在且顺序正确(self):
        cmd = self.p.to_wav_cmd("in.m4a", "out.wav", "0")
        self.assertEqual(cmd[-1], "out.wav", "输出必须是最后一个参数")
        self.assertEqual(cmd[cmd.index("-i") + 1], "in.m4a")

    def test_含空格和中文的路径原样传递(self):
        """走的是 list 形式的 subprocess,不该有任何转义处理。"""
        cmd = self.p.to_wav_cmd("/a b/会议 录音.m4a", "/c d/音频.wav", "0")
        self.assertIn("/a b/会议 录音.m4a", cmd)
        self.assertEqual(cmd[-1], "/c d/音频.wav")


class TestFailReason(unittest.TestCase):
    """失败时必须带出子进程的真实死因。"""

    def setUp(self):
        sys.modules["config"] = types.ModuleType("config")
        sys.modules["config"].HERE = str(WEB)
        fake = types.ModuleType("sessions")
        fake.sess_dir = fake.read_meta = fake.write_meta = lambda *a, **k: None
        sys.modules["sessions"] = fake
        self.j = _load("wjobs2", WEB / "jobs.py")

    def test_挑出traceback那行(self):
        tail = ["准备：转 16k wav", "Traceback (most recent call last):",
                "subprocess.CalledProcessError: returned non-zero exit status 234"]
        r = self.j.fail_reason(1, tail)
        self.assertIn("234", r, f"真实死因没带出来：{r}")

    def test_没有错误关键词时也给最后一行(self):
        r = self.j.fail_reason(3, ["准备：转 16k wav"])
        self.assertIn("转 16k wav", r)
        self.assertIn("3", r, "退出码要带上")

    def test_子进程完全没输出也不能是空话(self):
        r = self.j.fail_reason(9, [])
        self.assertIn("9", r)
        self.assertTrue(len(r) > 8)

    def test_超长行被截断(self):
        r = self.j.fail_reason(1, ["Error: " + "x" * 5000])
        self.assertLess(len(r), 400, "错误信息不能撑爆前端")

    def test_退出码0但没产出也要报(self):
        """ffmpeg 有时返回 0 却没写出文件 —— 这种最容易被当成成功。"""
        r = self.j.fail_reason(0, ["Error opening output files: Invalid argument"])
        self.assertIn("Invalid argument", r)


def load_minutes_lib():
    """minutes_lib 顶层要 import 仓库根的 prompts,路径它自己会插。"""
    sys.path.insert(0, str(REPO))
    return _load("wminutes", WEB / "meeting" / "minutes_lib.py")


class TestModelChain(unittest.TestCase):
    """网关上的模型会被下线 —— 实测配置里的 Qwen3.6 已 404,纪要功能因此静默失效。"""

    def setUp(self):
        self.m = load_minutes_lib()

    def test_单个模型(self):
        self.assertEqual(self.m.model_chain("Qwen3.6"), ["Qwen3.6"])

    def test_逗号分隔按序展开(self):
        self.assertEqual(self.m.model_chain("A,B,C"), ["A", "B", "C"])

    def test_空白被剔除(self):
        self.assertEqual(self.m.model_chain(" A , B "), ["A", "B"])

    def test_重复的只留第一次(self):
        """重复候选会让失败时白等两轮重试。"""
        self.assertEqual(self.m.model_chain("A,B,A"), ["A", "B"])

    def test_空配置不产生空候选(self):
        for spec in ("", "   ", ",,,"):
            self.assertEqual(self.m.model_chain(spec), [], f"{spec!r} 应得空列表")


class TestAskErrors(unittest.TestCase):
    """全挂时必须说清楚是哪个模型、为什么 —— 从前只有一句"网关无返回"。"""

    def setUp(self):
        self.m = load_minutes_lib()
        self._orig = self.m._call

    def tearDown(self):
        self.m._call = self._orig

    def test_首选挂了自动退到备选(self):
        seen = []

        def fake(model, *a):
            seen.append(model)
            if model == "dead":
                raise RuntimeError("Error code: 404")
            return "纪要正文"
        self.m._call = fake
        self.m.LLM_MODEL = "dead,alive"
        self.assertEqual(self.m.ask("s", "u"), "纪要正文")
        self.assertEqual(seen, ["dead", "alive"], "404 应立刻换模型,不该重试")

    def test_全挂时错误里带上模型名和原因(self):
        self.m._call = lambda *a: (_ for _ in ()).throw(RuntimeError("Error code: 404"))
        self.m.LLM_MODEL = "m1,m2"
        with self.assertRaises(RuntimeError) as cm:
            self.m.ask("s", "u")
        msg = str(cm.exception)
        self.assertIn("m2", msg, f"没带上模型名：{msg}")
        self.assertIn("404", msg, f"没带上真实死因：{msg}")

    def test_返回空内容也算失败(self):
        """网关返回 200 但 content 为空,从前会被当成成功。"""
        self.m._call = lambda *a: ""
        self.m.LLM_MODEL = "m1"
        with self.assertRaises(RuntimeError):
            self.m.ask("s", "u")


def load_stt():
    """stt.py 顶层会加载模型,SKIP_MODEL 让它跳过。"""
    os.environ["SKIP_MODEL"] = "1"
    fake = types.ModuleType("config")
    fake.MODEL = "x"; fake.LLM_URL = "http://127.0.0.1:1/v1/chat/completions"
    fake.LLM_KEY = ""; fake.LLM_MODEL = "m1,m2"
    sys.modules["config"] = fake
    return _load("wstt", WEB / "stt.py")


class TestSttModelChain(unittest.TestCase):
    """minutes_lib 和 caption_core 都加过 fallback,唯独实时字幕的 translate 是单点。

    于是配置改成 "Qwen3.6,qwen3.5-...,DeepSeek" 之后,它把整串当成一个模型名
    发出去,网关回 404 —— **外语字幕的中文翻译全线不工作**,而配置看起来是对的。
    改一处漏两处比不改更糟。
    """

    def setUp(self):
        self.m = load_stt()

    def test_逗号分隔按序展开(self):
        self.assertEqual(self.m.model_chain("A,B,C"), ["A", "B", "C"])

    def test_去重且保序(self):
        self.assertEqual(self.m.model_chain("A, B ,A"), ["A", "B"])

    def test_空配置也有一个候选(self):
        self.assertEqual(self.m.model_chain(""), [""])

    def test_单个模型不受影响(self):
        self.assertEqual(self.m.model_chain("Qwen3.6"), ["Qwen3.6"])


class TestRetryClassification(unittest.TestCase):
    """换不换下一个候选,分界线是**错在模型上还是错在网关上**。

    第一版只认 404,实测撞上 429 就整个放弃了,而换个模型立刻能用。
    """

    def setUp(self):
        self.m = load_stt()

    def _http(self, code):
        import urllib.error
        return urllib.error.HTTPError("u", code, "err", {}, None)

    def test_模型侧错误要换下一个(self):
        for code in (404, 429, 500, 502, 503):
            self.assertTrue(self.m._try_next(self._http(code)),
                            f"HTTP {code} 应换下一个候选")

    def test_网关连不上就别耗时间(self):
        """三个候选 × 60 秒超时 = 白等三分钟,实时字幕等不起。"""
        import urllib.error
        self.assertFalse(self.m._try_next(urllib.error.URLError("connection refused")))
        self.assertFalse(self.m._try_next(OSError("timed out")))

    def test_客户端自己写错了不该重试(self):
        self.assertFalse(self.m._try_next(self._http(400)))
        self.assertFalse(self.m._try_next(self._http(401)))


class TestTranslateDegrades(unittest.TestCase):
    def setUp(self):
        self.m = load_stt()

    def test_中文原样返回不发请求(self):
        self.assertEqual(self.m.translate("已经是中文", "zh"), "已经是中文")

    def test_空文本直接返回(self):
        self.assertEqual(self.m.translate("", "en"), "")

    def test_全挂时返回空而不是抛异常(self):
        """字幕宁可只显示原文,也别因为翻译失败整条消失。"""
        self.assertEqual(self.m.translate("hello", "en"), "")

    def test_translate真的把候选挨个试过(self):
        """光测 model_chain() 这个纯函数不够 —— 那正是这次 bug 的形状:
        函数写好了、配置也对,但调用方压根没用它。必须断言【实际发出的请求】。
        """
        seen = []
        import json as _json, urllib.error, urllib.request
        orig = urllib.request.urlopen

        def fake(req, timeout=None):
            seen.append(_json.loads(req.data.decode())["model"])
            raise urllib.error.HTTPError("u", 404, "Model not found", {}, None)

        urllib.request.urlopen = fake
        try:
            self.m.LLM_MODEL = "m1,m2,m3"
            self.m.translate("hello", "en")
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(seen, ["m1", "m2", "m3"], f"没有逐个试候选：{seen}")

    def test_流式也把候选挨个试过(self):
        seen = []
        import json as _json, urllib.error, urllib.request
        orig = urllib.request.urlopen

        def fake(req, timeout=None):
            seen.append(_json.loads(req.data.decode())["model"])
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)

        urllib.request.urlopen = fake
        try:
            self.m.LLM_MODEL = "m1,m2"
            list(self.m.translate_stream("hello", "en"))
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(seen, ["m1", "m2"], f"流式没走候选链：{seen}")

    def test_首个候选可用时不再试后面的(self):
        """退避不能变成"每次都把所有模型跑一遍"。"""
        seen = []
        import io as _io, json as _json, urllib.request

        class _R(_io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake(req, timeout=None):
            seen.append(_json.loads(req.data.decode())["model"])
            return _R(_json.dumps({"choices": [{"message": {"content": "译文"}}]}).encode())

        orig = urllib.request.urlopen
        urllib.request.urlopen = fake
        try:
            self.m.LLM_MODEL = "m1,m2,m3"
            self.assertEqual(self.m.translate("hello", "en"), "译文")
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(seen, ["m1"], f"第一个就成功了却还试了别的：{seen}")

    def test_流式全挂时不抛异常(self):
        self.assertEqual("".join(self.m.translate_stream("hello", "en")), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
