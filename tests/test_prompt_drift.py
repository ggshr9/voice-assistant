# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""prompt 只能有一份 —— 本机 CLI 与服务器网页版共用仓库根的 prompts.py。

这个守卫存在的原因:两边曾各存一份,谁也没同步谁。实测漂移到 FINAL_SYS 只剩
66% 相似,而服务器那份是活的(sessions/ 里最近一场真实会议是 2026-07-02),
也就是说走网页版的会用的一直是旧 prompt。

跑: uv run tests/test_prompt_drift.py
"""
import importlib.util
import pathlib
import re
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import prompts  # noqa: E402

CONSUMERS = ["bin/minutes", "web/meeting/minutes_lib.py"]
NAMES = ["NOTE_SYS", "FINAL_SYS", "RECORD_SYS", "ENHANCE_SYS", "PICK_SYS", "ANSWER_SYS"]


class TestSingleSource(unittest.TestCase):
    def test_共享模块三个常量都在且非空(self):
        for name in NAMES:
            self.assertTrue(getattr(prompts, name, "").strip(), f"{name} 是空的")

    def test_没有消费者自己另存一份(self):
        """谁再把 prompt 定义写回自己文件里，这里就会失败。"""
        offenders = {}
        for rel in CONSUMERS:
            src = (REPO / rel).read_text(encoding="utf-8")
            local = [n for n in NAMES if re.search(rf"^{n}\s*=\s*\(", src, re.M)]
            if local:
                offenders[rel] = local
        self.assertEqual(offenders, {},
                         f"这些文件自己又存了一份 prompt，应从 prompts.py 导入：{offenders}")

    def test_每个消费者都确实导入了共享模块(self):
        for rel in CONSUMERS:
            src = (REPO / rel).read_text(encoding="utf-8")
            self.assertIn("from prompts import", src, f"{rel} 没有导入共享 prompt")

    def test_me指令由共享函数生成(self):
        """「哪个说话人是我」的指令也曾是两边各写各的。"""
        out = prompts.me_instruction("说话人C")
        self.assertIn("说话人C", out)
        self.assertIn("**我**", out)
        self.assertEqual(prompts.me_instruction(""), "")
        self.assertEqual(prompts.me_instruction(None), "")

    def test_服务器版找共享模块的路径算法正确(self):
        """web/meeting/minutes_lib.py 用 dirname x3 上溯。

        本机是 repo/web/meeting -> repo；服务器是 voice-svc/web/meeting -> voice-svc。
        同一份代码在两边都要落到放着 prompts.py 的那一层。
        """
        src = (REPO / "web/meeting/minutes_lib.py").read_text(encoding="utf-8")
        self.assertIn("os.path.realpath(__file__)", src,
                      "必须用 realpath —— abspath 不解析软链")
        self.assertEqual(src.count("os.path.dirname("), 3,
                         "上溯层数变了就会找不到 prompts.py")
        landing = (REPO / "web/meeting/minutes_lib.py").parent.parent.parent
        self.assertTrue((landing / "prompts.py").exists(),
                        f"上溯到 {landing}，那里没有 prompts.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
