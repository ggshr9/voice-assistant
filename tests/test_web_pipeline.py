# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""服务器侧 meeting_pipeline / jobs 的纯逻辑测试。

**为什么补**:`meeting_pipeline.py` 此前零覆盖,里面藏着一个把 ffmpeg 参数插错位的
bug —— `ff[5:5] = ["-ac","1"]` 插进了 `-ar` 和 `16000` 中间,生成 `-ar -ac 1 16000`。
它只在【非线上】场景触发,所以线上会议一路正常,而**每一次上传都必然失败**,
活了两个月。同期 `jobs.py` 把子进程输出按关键词过滤,traceback 全丢,
失败只报一句"处理未产出结果" —— 两个缺陷叠加,等于故障不可诊断。

跑: uv run tests/test_web_pipeline.py
"""
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
