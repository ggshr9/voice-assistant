# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""server/systemd/ 里的 unit 文件：占位符可渲染、结构完整。

真机上装错 unit 的代价很高（服务起不来、或者以 root 跑），而这些文件在 macOS 上
没法用 systemd-analyze 校验，所以这里做结构检查。

跑: uv run tests/test_systemd_units.py
"""
import configparser
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
UNITS = sorted((REPO / "server" / "systemd").glob("*.service")) + \
        sorted((REPO / "server" / "systemd").glob("*.timer"))

# 由 timer 触发的 oneshot 不该有 [Install] —— 它不该被 enable
TIMER_TRIGGERED = {"caption-renew.service"}


def render(text, user="testuser", home="/home/testuser"):
    return text.replace("__USER__", user).replace("__HOME__", home)


def parse(text):
    c = configparser.ConfigParser(strict=False)
    c.optionxform = str
    c.read_string(text)
    return c


class TestUnitsExist(unittest.TestCase):
    def test_三个服务加一个定时器都在(self):
        names = {p.name for p in UNITS}
        self.assertEqual(names, {"caption-web.service", "caption-stt.service",
                                 "caption-renew.service", "caption-renew.timer"},
                         "unit 少了或多了——sync-web 的 --delete 曾把新建的 unit 删掉过")


class TestPlaceholders(unittest.TestCase):
    """不用 systemd 的 %h/%i：系统级 unit 里 %h 解析成 /root、%i 只在模板单元有效。"""

    def test_没有写死的家目录或用户名(self):
        for p in UNITS:
            text = p.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"/home/(?!testuser)[a-z]+",
                                f"{p.name} 里写死了某人的家目录")
            self.assertNotRegex(text, r"^User=(?!__USER__)[a-z]+$",
                                f"{p.name} 里写死了用户名")

    def test_不用systemd的specifier(self):
        for p in UNITS:
            text = p.read_text(encoding="utf-8")
            body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
            self.assertNotIn("%h", body, f"{p.name} 用了 %h —— 系统级 unit 里它是 /root")
            self.assertNotIn("%i", body, f"{p.name} 用了 %i —— 只在模板单元有效")

    def test_渲染后不留占位符(self):
        for p in UNITS:
            out = render(p.read_text(encoding="utf-8"))
            self.assertNotIn("__USER__", out)
            self.assertNotIn("__HOME__", out)


class TestStructure(unittest.TestCase):
    def test_段落完整(self):
        for p in UNITS:
            c = parse(render(p.read_text(encoding="utf-8")))
            self.assertIn("Unit", c.sections(), p.name)
            want = "Timer" if p.suffix == ".timer" else "Service"
            self.assertIn(want, c.sections(), p.name)

    def test_该被enable的才有Install段(self):
        """caption-renew.service 由 timer 触发，有 [Install] 反而是错的。"""
        for p in UNITS:
            c = parse(render(p.read_text(encoding="utf-8")))
            has = "Install" in c.sections()
            if p.name in TIMER_TRIGGERED:
                self.assertFalse(has, f"{p.name} 由 timer 触发，不该有 [Install]")
            else:
                self.assertTrue(has, f"{p.name} 需要 [Install] 才能 enable")

    def test_长跑服务要能自愈(self):
        for p in UNITS:
            if p.suffix != ".service" or p.name in TIMER_TRIGGERED:
                continue
            c = parse(render(p.read_text(encoding="utf-8")))
            self.assertEqual(c["Service"].get("Restart"), "always",
                             f"{p.name} 挂了不会自己起来")

    def test_绑443的服务有对应capability(self):
        c = parse(render((REPO / "server/systemd/caption-web.service").read_text(encoding="utf-8")))
        self.assertIn("CAP_NET_BIND_SERVICE", c["Service"].get("AmbientCapabilities", ""),
                      "以普通用户绑 443 必须有这个 capability")


class TestLaunchAgents(unittest.TestCase):
    """macOS 的 plist 同理：不能写死某个人的家目录。"""

    PLISTS = sorted((REPO / "launchagents").glob("*.plist"))

    def test_没有写死的家目录(self):
        for p in self.PLISTS:
            self.assertNotRegex(p.read_text(encoding="utf-8"), r"/Users/[A-Za-z0-9_.-]+",
                                f"{p.name} 里写死了某人的家目录")

    def test_渲染后是合法plist(self):
        import plistlib
        for p in self.PLISTS:
            rendered = p.read_text(encoding="utf-8").replace("__HOME__", "__HOME__")
            self.assertNotIn("__HOME__", rendered)
            try:
                plistlib.loads(rendered.encode("utf-8"))
            except Exception as e:                      # noqa: BLE001
                self.fail(f"{p.name} 渲染后不是合法 plist：{e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
