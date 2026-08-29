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
import tempfile
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

    def test_空配置给一个空串候选(self):
        """契约已统一到 llm_chain.parse_chain:空配置 → [""]。
        空串是合法的 model(有些端点忽略该字段),给空列表会导致一次都不尝试 ——
        这条测试原先锁的是 minutes_lib 独立实现时的旧语义([]),三处合一后
        以共享契约为准(stt 与 caption_core 的测试一直就是这么断言的)。"""
        for spec in ("", "   ", ",,,"):
            self.assertEqual(self.m.model_chain(spec), [""], f"{spec!r}")


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


def load_asr_backends():
    return _load("wasrb", WEB / "asr_backends.py")


class TestBackendRegistry(unittest.TestCase):
    """换模型只动 asr_backends.py —— 这层的价值就在注册表和统一契约。"""

    def setUp(self):
        self.b = load_asr_backends()

    def test_内置两个后端都注册了(self):
        self.assertIn("qwen3", self.b.BACKENDS)
        self.assertIn("whisper", self.b.BACKENDS)

    def test_未知后端报错并列出可用的(self):
        with self.assertRaises(ValueError) as cm:
            self.b.load_backend("不存在的模型")
        msg = str(cm.exception)
        self.assertIn("qwen3", msg, "要列出可用后端,让人知道能选什么")

    def test_注册新后端不用改任何现有代码(self):
        """这是这层存在的理由:新模型 = 一个类 + 一行装饰器。"""
        @self.b.register("fake")
        class Fake:
            def transcribe(self, audio, lang_ui):
                return "假转写", lang_ui or ""
        try:
            inst = self.b.load_backend("fake")
            self.assertEqual(inst.transcribe(None, "zh"), ("假转写", "zh"))
        finally:
            self.b.BACKENDS.pop("fake", None)

    def test_环境变量选择后端(self):
        @self.b.register("fromenv")
        class F:
            pass
        try:
            os.environ["CAPTION_ASR_BACKEND"] = "fromenv"
            self.assertIsInstance(self.b.load_backend(), F)
        finally:
            os.environ.pop("CAPTION_ASR_BACKEND", None)
            self.b.BACKENDS.pop("fromenv", None)

    def test_qwen的token上限按时长走(self):
        """延迟护栏是 qwen 私有知识,搬进后端后不能丢:10 秒段曾因生成循环耗 5.67 秒。"""
        import numpy as np
        cls = self.b.BACKENDS["qwen3"]
        inst = cls.__new__(cls)                    # 不加载真模型
        calls = {}
        class FakeM:
            def transcribe(self, audio, language):
                calls["cap"] = self.max_new_tokens
                return []
        inst.m = FakeM()
        inst.transcribe(np.zeros(10 * 16000, dtype="float32"), "zh")
        self.assertEqual(calls["cap"], 80, "10 秒 × 8 token/秒")
        inst.transcribe(np.zeros(1 * 16000, dtype="float32"), "zh")
        self.assertEqual(calls["cap"], cls.MIN_NEW, "短段也要有下限")

    def test_qwen语言映射不外漏(self):
        """契约:进出都是界面码 zh/en。Chinese/English 是 qwen 的私事。"""
        import numpy as np
        cls = self.b.BACKENDS["qwen3"]
        inst = cls.__new__(cls)
        seen = {}
        class FakeM:
            max_new_tokens = 0
            def transcribe(self, audio, language):
                seen["lang"] = language
                class R: text = "好"; language = "Chinese"
                return [R()]
        inst.m = FakeM()
        text, lang = inst.transcribe(np.zeros(16000, dtype="float32"), "zh")
        self.assertEqual(seen["lang"], "Chinese", "送进 qwen 的要是全称")
        self.assertEqual(lang, "zh", "吐出来的要是界面码")


class TestSttDelegates(unittest.TestCase):
    """transcribe_pcm 必须真的把活交给后端 —— 上次的教训:纯函数都对,调用方没用它。"""

    def test_能量门之后交给后端(self):
        m = load_stt()
        calls = []
        class FakeBackend:
            def transcribe(self, audio, lang):
                calls.append((len(audio), lang))
                return "转写结果", "zh"
        m.model = FakeBackend()
        import numpy as np
        loud = (np.sin(np.linspace(0, 800, 16000)) * 20000).astype("<i2").tobytes()
        self.assertEqual(m.transcribe_pcm(loud, "zh"), ("转写结果", "zh"))
        self.assertEqual(calls, [(16000, "zh")])

    def test_静音被能量门拦住不进后端(self):
        m = load_stt()
        class Boom:
            def transcribe(self, audio, lang):
                raise AssertionError("静音不该进模型")
        m.model = Boom()
        self.assertEqual(m.transcribe_pcm(b"\x00" * 32000, "zh"), ("", "zh"))


class TestTemplateWiring(unittest.TestCase):
    """模板要真的传到 LLM —— 「函数对了≠被调用了」,断言实际发出的 sysmsg。"""

    def test_make_minutes按模板换sys(self):
        m = load_minutes_lib()
        seen = []
        orig = m.ask
        m.ask = lambda sysmsg, user, *a, **k: (seen.append(sysmsg) or "纪要")
        try:
            m.make_minutes("转写内容", template="weekly")
            self.assertIn("各人进展", seen[-1], "weekly 模板没传到 LLM")
            m.make_minutes("转写内容", template="不存在的")
            self.assertIn("关键决议", seen[-1], "未知模板该落回默认结构")
        finally:
            m.ask = orig

    def test_pipeline从meta读template(self):
        src = (WEB / "meeting" / "meeting_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('get("template"', src)
        self.assertIn("template=template", src)

    def test_start与upload都存template(self):
        src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(src.count('"template"') + src.count("'template'"), 2)


class TestMakeEnhanced(unittest.TestCase):
    """增强笔记:用户手记为骨架,转写为血肉(Granola 式)。

    与 make_minutes 是两种产物 —— 那个是 AI 从零写的,这个是用户自己的笔记
    被补全后的样子。契约的关键:笔记一个字都不能丢在压缩里。
    """

    def setUp(self):
        self.m = load_minutes_lib()
        self._orig = self.m.ask
        self.calls = []
        self.m.ask = lambda sysmsg, user, *a, **k: (self.calls.append((sysmsg, user)) or "增强结果")

    def tearDown(self):
        self.m.ask = self._orig

    def test_没有手记直接返回空_不烧LLM(self):
        for empty in ("", "   ", None, "\n\n"):
            self.assertEqual(self.m.make_enhanced(empty, "转写", log=lambda *a: None), "")
        self.assertEqual(self.calls, [], "空手记不该调 LLM")

    def test_手记和转写都进了prompt(self):
        self.m.make_enhanced("- 定了用Redis", "说话人A：那就定 Redis", log=lambda *a: None)
        sysmsg, user = self.calls[-1]
        self.assertIn("骨架", sysmsg, "要用 ENHANCE_SYS 而不是别的 prompt")
        self.assertIn("定了用Redis", user)
        self.assertIn("那就定 Redis", user)

    def test_超长转写被压缩但手记原样保留(self):
        """笔记是主角:压缩只能压转写,笔记一个字不能少。"""
        notes = "- 我记的独特要点ABC"
        self.m.make_enhanced(notes, "长" * 30000, log=lambda *a: None)
        final_user = self.calls[-1][1]
        self.assertIn(notes, final_user, "手记在压缩中丢了")
        self.assertLess(len(final_user), 25000, "转写没被压缩,会撑爆上下文")

    def test_正常长度不压缩_一次调用(self):
        self.m.make_enhanced("- 要点", "短转写", log=lambda *a: None)
        self.assertEqual(len(self.calls), 1, "不超长不该多跑压缩步")


class TestNotesEndpointSource(unittest.TestCase):
    """notes 接口的两条源码级不变量(与 session_minutes 的教训同款)。"""

    def setUp(self):
        src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        i = src.index("async def session_notes")
        self.body = src[i:src.index("\nasync def", i + 10)]

    def test_口令校验在查会话之前(self):
        """404/403 的差别会向未鉴权者泄漏会话存不存在 —— session_minutes 犯过。"""
        self.assertLess(self.body.index("check_pw"), self.body.index("sess_dir"))

    def test_写入是原子的(self):
        """手记是用户唯一手打的东西,比转写更不可再生;自动保存每几秒打一次,
        截断式写法碰上刷新/断电就把笔记清空了。"""
        self.assertIn("os.replace", self.body)
        self.assertLess(self.body.index('".notes.md.part"'), self.body.index("os.replace"))

    def test_有大小上限(self):
        self.assertIn("413", self.body)


class TestPipelineWritesEnhanced(unittest.TestCase):
    """有 notes.md 才产出增强笔记;没有则完全不碰这条路。"""

    def setUp(self):
        self.p = load_pipeline()
        self.src = (WEB / "meeting" / "meeting_pipeline.py").read_text(encoding="utf-8")

    def test_产物写入是原子的(self):
        i = self.src.index('".增强笔记.md.part"')
        self.assertLess(i, self.src.index('os.replace(tmp, os.path.join(outdir, "增强笔记.md"))'))

    def test_返回值带enhanced字段(self):
        self.assertIn('"enhanced": enhanced', self.src)


def load_recall_lib():
    sys.path.insert(0, str(REPO))                    # recall_core / prompts 在仓库根
    fake = types.ModuleType("minutes_lib")
    fake.ask = lambda *a, **k: "1"
    sys.modules["minutes_lib"] = fake
    return _load("wrecall", WEB / "meeting" / "recall_lib.py"), fake


class TestRecallLib(unittest.TestCase):
    """服务器侧跨会议检索:目录扫描的健壮性 + 两段式的接线。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.r, self.fake = load_recall_lib()

    def tearDown(self):
        self._tmp.cleanup()

    def _sess(self, name, title="会", minutes=None, started=1787700000):
        d = os.path.join(self.root, name); os.makedirs(d)
        import json as _j
        open(os.path.join(d, "meta.json"), "w", encoding="utf-8").write(
            _j.dumps({"id": name, "title": title, "started": started}))
        if minutes is not None:
            open(os.path.join(d, "会议纪要.md"), "w", encoding="utf-8").write(minutes)
        return d

    def test_没有文字内容的会不进目录(self):
        """只有录音没转写的会,检索了也答不出 —— 别把它塞给选会模型当噪声。"""
        self._sess("a", minutes="## 一句话摘要\n定了用Redis\n")
        self._sess("b", minutes=None)                # 只有 meta,没有任何文字
        entries = self.r.load_entries(self.root)
        self.assertEqual([e["id"] for e in entries], ["a"])

    def test_坏meta跳过而不是拖垮整个检索(self):
        d = os.path.join(self.root, "bad"); os.makedirs(d)
        open(os.path.join(d, "meta.json"), "w").write("不是json{{{")
        self._sess("good", minutes="纪要")
        self.assertEqual(len(self.r.load_entries(self.root)), 1)

    def test_摘要取一句话摘要段(self):
        self._sess("a", minutes="# 头\n\n## 一句话摘要\n确定采用 Redis 方案\n\n## 决议\n…")
        self.assertIn("Redis", self.r.load_entries(self.root)[0]["summary"])

    def test_两段式真的接上了(self):
        """第一段收到目录、第二段收到正文与问题 —— 断言实际传给 LLM 的内容,
        不只测各函数自身(「函数对了≠被调用了」的教训)。"""
        self._sess("a", title="缓存评审", minutes="## 一句话摘要\n定Redis\n\n正文:那就定 Redis")
        calls = []
        self.fake.ask = lambda sysmsg, user, **k: (calls.append((sysmsg, user)) or ("1" if len(calls) == 1 else "答案:Redis"))
        r = self.r.ask_meetings("缓存定了什么", self.root)
        self.assertEqual(len(calls), 2, "该恰好两段:选会 + 作答")
        self.assertIn("缓存评审", calls[0][1], "第一段要看到目录")
        self.assertIn("那就定 Redis", calls[1][1], "第二段要看到正文")
        self.assertEqual(r["sources"][0]["title"], "缓存评审")

    def test_模型没挑出来走关键词兜底(self):
        self._sess("a", title="Redis缓存评审", minutes="## 一句话摘要\n定Redis\n")
        self.fake.ask = lambda sysmsg, user, **k: ("none" if "挑出" in sysmsg or "编号" in sysmsg else "答")
        r = self.r.ask_meetings("Redis缓存", self.root)
        self.assertTrue(r["sources"], "兜底没接上")

    def test_空库给友好提示不调LLM(self):
        calls = []
        self.fake.ask = lambda *a, **k: calls.append(1)
        r = self.r.ask_meetings("问题", self.root)
        self.assertIn("空", r["answer"])
        self.assertEqual(calls, [])

    def test_空问题明确报错(self):
        with self.assertRaises(ValueError):
            self.r.ask_meetings("  ", self.root)


class TestAskEndpointSource(unittest.TestCase):
    def setUp(self):
        src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        i = src.index("async def ask_library")
        self.body = src[i:src.index("\nasync def", i + 10)]

    def test_口令先于一切(self):
        self.assertLess(self.body.index("check_pw"), self.body.index("recall_lib"))

    def test_LLM走executor不卡事件循环(self):
        """检索要 5~20 秒,阻塞事件循环会卡住正在录音的 WebSocket。"""
        self.assertIn("run_in_executor", self.body)

    def test_有长度上限(self):
        self.assertIn("413", self.body)


class TestAutoLangMapping(unittest.TestCase):
    """qwen_asr 没有自动检测:None 静默返回空、"auto" 抛异常被吞 ——
    用户选「语言·自动」曾导致实时字幕与批处理【双双整场空转写】,且查无此错。
    auto 必须落到默认语种,这条约定两条路径各锁一测。"""

    def test_实时后端auto落到默认语种(self):
        import numpy as np
        b = load_asr_backends()
        cls = b.BACKENDS["qwen3"]; inst = cls.__new__(cls)
        seen = {}
        class M:
            max_new_tokens = 0
            def transcribe(self, audio, language):
                seen["lang"] = language; return []
        inst.m = M()
        for ui in (None, "", "auto"):
            inst.transcribe(np.zeros(16000, dtype="float32"), ui)
            self.assertEqual(seen["lang"], "Chinese",
                             f"lang_ui={ui!r} 必须映射成 Chinese,传 None 会静默得到空串")

    def test_批处理语言表没有auto陷阱(self):
        src = (WEB / "meeting" / "meeting_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn('"auto": "auto"', src,
                         '"auto"→"auto" 会让 validate 抛异常并被 per-clip except 吞成空转写')


class TestMaterialWarning(unittest.TestCase):
    """素材质检:警告必须写进【产物本身】。

    事故:用户拿一场被硬转的背景音乐评估了半天系统,整条链路零提示 ——
    pyannote 明明知道(语音帧 0.0%),但没人把它的话传出去。"""

    def setUp(self):
        self.p = load_pipeline()

    def test_音乐素材触发警告且写明占比(self):
        w = self.p.material_warning({"speech_ratio": 0.0, "diarization_fallback": True})
        self.assertIsNotNone(w)
        self.assertIn("0%", w)
        self.assertIn("不可尽信", w)

    def test_正常会议不打扰(self):
        """误报会让警告失去公信力 —— 正常占比(实测 0.5+)绝不能弹。"""
        self.assertIsNone(self.p.material_warning({"speech_ratio": 0.52,
                                                   "diarization_fallback": False}))

    def test_占比正常但降级过_提示分人而非素材(self):
        w = self.p.material_warning({"speech_ratio": 0.5, "diarization_fallback": True})
        self.assertIn("说话人", w)
        self.assertNotIn("不可尽信", w, "素材没问题就别吓唬人")

    def test_没有边车时静默(self):
        """老会话没有 diar_stats.json —— 质检缺失不该拦主流程也不该报错。"""
        self.assertIsNone(self.p.material_warning({}))
        self.assertIsNone(self.p.material_warning(None))

    def test_警告进纪要头部_源码级(self):
        src = (WEB / "meeting" / "meeting_pipeline.py").read_text(encoding="utf-8")
        i = src.index("make_minutes(text, me, log")
        blk = src[i:src.index("会议纪要.md", i)]
        self.assertIn("warn", blk, "警告没有拼进纪要正文")


class TestExportEndpointSource(unittest.TestCase):
    """导出接口不变量:doc 参数走白名单(拼路径的地方绝不放行任意文件名)。"""

    def setUp(self):
        src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        i = src.index("async def session_export")
        self.body = src[i:src.index("\nasync def", i + 10)]

    def test_doc走白名单不拼任意文件名(self):
        self.assertIn("_EXPORT_DOCS.get", self.body)
        self.assertNotIn('request.query.get("doc")]', self.body.replace(" ", ""))

    def test_口令先于查会话(self):
        self.assertLess(self.body.index("check_pw"), self.body.index("sess_dir"))

    def test_pandoc缺失给可操作提示(self):
        self.assertIn("501", self.body)
        self.assertIn("DEPLOY", self.body)


class TestClaimEndpointSource(unittest.TestCase):
    """声纹认领(网页化)的四条源码级不变量。"""

    def setUp(self):
        src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        i = src.index("async def session_claim")
        self.body = src[i:src.index("\nasync def", i + 10)]
        self.full = src

    def test_注册表路径与批处理识别一致(self):
        """首个 E2E 就翻在这:claim 写了默认 ~/.config/,识别读 ~/voice-svc/,
        注册进了一个没人读的本子。路径必须同源。"""
        self.assertIn("voice-svc/voiceprints.json", self.full)
        step = (WEB / "meeting" / "asr_diarize_step.py").read_text(encoding="utf-8")
        self.assertIn("voice-svc/voiceprints.json", step)

    def test_两种标签排版都要替(self):
        """annotated 是「说话人B：」,会议记录被 LLM 排版成「说话人 B：」——
        第一版只查一种,第二种整块被跳过。"""
        self.assertIn("variants", self.body)

    def test_口令先于查会话(self):
        self.assertLess(self.body.index("check_pw"), self.body.index("sess_dir"))

    def test_写注册表在锁内(self):
        i = self.body.index("registry_lock")
        blk = self.body[i:i + 300]
        self.assertIn("add_embedding", blk)
        self.assertIn("save_registry", blk)

    def test_没有向量时给出可操作的指引(self):
        self.assertIn("生成会议纪要", self.body, "409 要告诉用户先跑一遍分人,不是干报错")


class TestContainerKnobs(unittest.TestCase):
    """容器化(issue #2)依赖的三个开关 —— 都是源码级不变量,防被"顺手清理"掉。"""

    def test_whisper_compute_不写死float16(self):
        src = (WEB / "asr_backends.py").read_text(encoding="utf-8")
        self.assertIn("CAPTION_ASR_COMPUTE", src)
        self.assertIn("int8", src, "CPU 默认要落到 int8,否则容器 CPU profile 起不来")

    def test_批处理解释器可被环境顶掉(self):
        src = (WEB / "meeting" / "meeting_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("MEETING_PY_STT", src)
        self.assertIn("MEETING_PY_DIA", src)

    def test_自签证书必须显式开启(self):
        """裸机缺证书要大声报错的语义不能被容器便利冲掉。"""
        src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        self.assertIn('CAPTION_TLS_SELFSIGN") == "1"', src)
        self.assertIn("raise SystemExit", src[src.index("CAPTION_TLS_SELFSIGN"):],
                      "没开自签且没证书时仍要拒绝启动")


class TestPauseProtocol(unittest.TestCase):
    """暂停控制流的三条源码级不变量(issue #1 的彻底方案)。"""

    def setUp(self):
        self.src = (WEB / "web_caption.py").read_text(encoding="utf-8")
        i = self.src.index("async def ws_handler")
        self.body = self.src[i:self.src.index("\nasync def", i + 10)]

    def test_pause先定稿在途半句再清状态(self):
        """顺序错了半句就丢了:必须先 seg_q.put 再清 seg。"""
        i = self.body.index('ctrl.get("pause")')
        blk = self.body[i:self.body.index('ctrl.get("resume")')]
        self.assertIn("seg_q.put_nowait", blk, "暂停时在途半句没定稿")
        self.assertLess(blk.index("seg_q.put_nowait"), blk.index("seg, buf = bytearray()"),
                        "先清了 seg 再定稿 —— 半句已经没了")

    def test_暂停区间写进meta(self):
        self.assertIn("pause_spans", self.body, "暂停发生过这件事必须留痕:录音里是无缝的,"
                                                "不记 meta,音频/墙钟时间轴永远对不上")

    def test_暂停中直接停止时悬空区间闭合(self):
        """finally 里要把 [start, None] 补上终点,否则时长统计会算出负数/None。"""
        i = self.body.index("finally:")
        self.assertIn("pause_spans", self.body[i:], "收尾没闭合悬空的暂停区间")

    def test_未知文本消息仍走停止_向后兼容(self):
        """旧前端只发 {eof:1}:新服务端对认不出的文本必须还是 break(收尾),
        否则旧页面点停止会卡住。"""
        i = self.body.index('ctrl.get("resume")')
        tail = self.body[i:self.body.index("if msg.type != web.WSMsgType.BINARY")]
        self.assertIn("break", tail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
