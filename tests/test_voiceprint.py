# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""bin/_voiceprint.py 的纯逻辑测试：余弦 / 门槛 / 一对一分配 / 注册表增删。

全部注入假向量，不需要 GPU、不碰真声纹。跑: uv run tests/test_voiceprint.py
"""
import importlib.util
import pathlib
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_loader(
    "voiceprint", SourceFileLoader("voiceprint", str(REPO / "bin" / "_voiceprint.py")))
vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vp)


def vec(seed, dim=256):
    """造一个确定性的单位向量。"""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def near(base, noise=0.15, seed=99):
    """造一个"同一个人另一次录音"——与 base 高度相似但不相同。

    噪声要按维数缩放：256 维里直接加 0.15*randn，噪声总范数是 0.15*sqrt(256)=2.4，
    会把单位基向量整个淹没（第一版就这么错了，余弦只有 0.32）。
    """
    rng = np.random.default_rng(seed)
    v = base + noise * rng.normal(size=len(base)) / np.sqrt(len(base))
    return v / np.linalg.norm(v)


class TestCosine(unittest.TestCase):
    def test_自己和自己是1(self):
        v = vec(1)
        self.assertAlmostEqual(vp.cosine(v, v), 1.0, places=5)

    def test_零向量不炸(self):
        self.assertEqual(vp.cosine(np.zeros(256), vec(1)), 0.0)

    def test_同人相似度远高于陌生人(self):
        me, other = vec(1), vec(2)
        self.assertGreater(vp.cosine(me, near(me)), 0.9)
        self.assertLess(abs(vp.cosine(me, other)), 0.3)


class TestMatchSpeakers(unittest.TestCase):
    """核心：把本场会的说话人向量映射到已注册的人。"""

    def setUp(self):
        self.me, self.zhang = vec(1), vec(2)
        self.people = [
            {"name": "张三", "is_me": True, "embeddings": [self.me.tolist()]},
            {"name": "李四", "is_me": False, "embeddings": [self.zhang.tolist()]},
        ]

    def test_认出注册过的人(self):
        speakers = {"SPEAKER_00": near(self.me), "SPEAKER_01": near(self.zhang, seed=7)}
        m = vp.match_speakers(speakers, self.people)
        self.assertEqual(m["SPEAKER_00"], "张三")
        self.assertEqual(m["SPEAKER_01"], "李四")

    def test_没注册的人保持匿名(self):
        speakers = {"SPEAKER_00": near(self.me), "SPEAKER_09": vec(42)}
        m = vp.match_speakers(speakers, self.people)
        self.assertEqual(m["SPEAKER_00"], "张三")
        self.assertNotIn("SPEAKER_09", m, "认不出的说话人不该出现在映射里")

    def test_一个人只占一个说话人位(self):
        """同一个人被分人器切成两个簇时，只有更像的那个拿到名字。"""
        strong, weak = near(self.me, 0.05), near(self.me, 0.5, seed=5)
        m = vp.match_speakers({"SPEAKER_00": strong, "SPEAKER_01": weak}, self.people)
        self.assertEqual(m.get("SPEAKER_00"), "张三")
        self.assertIsNone(m.get("SPEAKER_01"), "同一个人不该同时占两个位")

    def test_低于门槛不认(self):
        m = vp.match_speakers({"SPEAKER_00": vec(77)}, self.people)
        self.assertEqual(m, {})

    def test_多条向量取最大而不是平均(self):
        """一个人存多次录音，只要有一次像就该认出来。"""
        old = vec(31)                       # 一条毫不相干的旧向量
        people = [{"name": "张三", "is_me": True,
                   "embeddings": [old.tolist(), self.me.tolist()]}]
        m = vp.match_speakers({"SPEAKER_00": near(self.me)}, people)
        self.assertEqual(m["SPEAKER_00"], "张三")

    def test_两人都沾边且差距不够时都不认(self):
        """宁可匿名也不瞎猜 —— 这是设计里写死的原则。"""
        blurry = (self.me + self.zhang)
        blurry = blurry / np.linalg.norm(blurry)
        m = vp.match_speakers({"SPEAKER_00": blurry}, self.people, threshold=0.3)
        self.assertEqual(m, {}, "两个注册者都沾边时应拒绝，而不是挑一个")

    def test_空注册表(self):
        self.assertEqual(vp.match_speakers({"SPEAKER_00": vec(1)}, []), {})

    def test_空说话人(self):
        self.assertEqual(vp.match_speakers({}, self.people), {})


class TestWhoAmI(unittest.TestCase):
    def test_找出标了is_me的人(self):
        people = [{"name": "李四", "is_me": False, "embeddings": []},
                  {"name": "张三", "is_me": True, "embeddings": []}]
        self.assertEqual(vp.me_name(people), "张三")

    def test_没人标is_me时返回空(self):
        self.assertIsNone(vp.me_name([{"name": "李四", "is_me": False, "embeddings": []}]))

    def test_从匹配结果反查我是哪个说话人(self):
        people = [{"name": "张三", "is_me": True, "embeddings": []}]
        mapping = {"SPEAKER_00": "李四", "SPEAKER_03": "张三"}
        self.assertEqual(vp.my_speaker(mapping, people), "SPEAKER_03")

    def test_我没出现在这场会里(self):
        people = [{"name": "张三", "is_me": True, "embeddings": []}]
        self.assertIsNone(vp.my_speaker({"SPEAKER_00": "李四"}, people))


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(pathlib.Path(self._tmp.name) / "voiceprints.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_没有文件时返回空表(self):
        self.assertEqual(vp.load_registry(self.path), [])

    def test_存进去再读出来(self):
        people = vp.add_embedding([], "李四", vec(2), source="会甲/说话人A")
        vp.save_registry(people, self.path)
        got = vp.load_registry(self.path)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["name"], "李四")
        self.assertEqual(len(got[0]["embeddings"][0]), 256)

    def test_同一个人第二次打标是追加不是覆盖(self):
        """重打标既是纠正也是学习 —— 向量要攒起来。"""
        people = vp.add_embedding([], "李四", vec(2), source="会甲/说话人A")
        people = vp.add_embedding(people, "李四", vec(3), source="会乙/说话人B")
        self.assertEqual(len(people), 1)
        self.assertEqual(len(people[0]["embeddings"]), 2)
        self.assertEqual(people[0]["sources"], ["会甲/说话人A", "会乙/说话人B"])

    def test_同一来源重复打标不重复攒(self):
        people = vp.add_embedding([], "李四", vec(2), source="会甲/说话人A")
        people = vp.add_embedding(people, "李四", vec(9), source="会甲/说话人A")
        self.assertEqual(len(people[0]["embeddings"]), 1, "同一段素材不该攒两遍")

    def test_标记我本人且只能有一个(self):
        people = vp.add_embedding([], "甲", vec(1), source="s1", is_me=True)
        people = vp.add_embedding(people, "乙", vec(2), source="s2", is_me=True)
        me = [p["name"] for p in people if p.get("is_me")]
        self.assertEqual(me, ["乙"], "改标别人为我时，旧的要被清掉")

    def test_删除一个人(self):
        people = vp.add_embedding([], "李四", vec(2), source="s")
        self.assertEqual(vp.forget(people, "李四"), [])

    def test_删除不存在的人不炸(self):
        self.assertEqual(vp.forget([], "查无此人"), [])

class TestConfidence(unittest.TestCase):
    """认出来了还不够 —— 得让人知道有多确定。

    门槛 0.55 之上就直接写真名，0.56 和 0.94 在输出里长得一模一样，
    用户没法判断该不该信。成熟项目（minutes）的说话人标签带 high/medium，
    我们至少要能把「勉强够线」和「几乎确定」区分开。
    """

    def setUp(self):
        self.me, self.other = vec(1), vec(2)
        self.people = [
            {"name": "张三", "is_me": True, "embeddings": [self.me.tolist()]},
            {"name": "李四", "is_me": False, "embeddings": [self.other.tolist()]},
        ]

    def test_默认仍返回名字映射_不破坏既有调用(self):
        m = vp.match_speakers({"SPEAKER_00": near(self.me)}, self.people)
        self.assertEqual(m["SPEAKER_00"], "张三")

    def test_要分数时返回名字和相似度(self):
        m = vp.match_speakers({"SPEAKER_00": near(self.me)}, self.people, scores=True)
        name, score = m["SPEAKER_00"]
        self.assertEqual(name, "张三")
        self.assertGreater(score, 0.9)
        self.assertLessEqual(score, 1.0)

    def test_勉强过线的分数也如实返回(self):
        """不能因为过了门槛就假装很确定。"""
        weak = near(self.me, noise=1.15, seed=3)
        m = vp.match_speakers({"S": weak}, self.people, scores=True, threshold=0.5, margin=0.05)
        if m:                                   # 造出的向量刚好在门槛附近
            _, score = m["S"]
            self.assertLess(score, 0.9, "勉强匹配不该报成高分")

    def test_is_sure_区分几乎确定与勉强够线(self):
        self.assertTrue(vp.is_sure(0.94))
        self.assertFalse(vp.is_sure(0.60))
        self.assertFalse(vp.is_sure(vp.THRESHOLD))

    def test_认不出的仍然不出现(self):
        m = vp.match_speakers({"S": vec(77)}, self.people, scores=True)
        self.assertEqual(m, {})

class TestOverSegmentationRepair(unittest.TestCase):
    """分人器把同一个人拆成多簇时，声纹应该把它们都认领回来。

    真实数据里这很常见：那场标称 8 人的会，两两相似度中位数 0.213（正常不同人的
    量级），但 SPEAKER_01↔05 是 0.927、01↔06 是 0.900 —— 那是「同一个人两段录音」
    的水平（实测同人 0.876/0.943）。也就是 8 簇里大概只有 4~5 个真人。

    原来的一对一约束是为了防瞎猜，但在过分割场景下恰恰做错了事：认领一簇、
    把另一簇留成匿名牌，稿子里同一个人一半有名字一半没有。
    """

    def setUp(self):
        self.me = vec(1)
        self.people = [{"name": "张三", "is_me": True, "embeddings": [self.me.tolist()]}]

    def test_同一个人的多簇都被认领(self):
        a, b = near(self.me, 0.05, seed=11), near(self.me, 0.08, seed=12)
        m = vp.match_speakers({"S0": a, "S1": b}, self.people, merge_clusters=True)
        self.assertEqual(m.get("S0"), "张三")
        self.assertEqual(m.get("S1"), "张三", "同一个人的第二簇也该拿到名字")

    def test_默认仍是一对一_不破坏既有行为(self):
        a, b = near(self.me, 0.05, seed=11), near(self.me, 0.08, seed=12)
        m = vp.match_speakers({"S0": a, "S1": b}, self.people)
        self.assertEqual(len([k for k, v in m.items() if v == "张三"]), 1)

    def test_只是碰巧过线的簇不会被顺带认领(self):
        """合并要靠【簇之间也相似】，不能只看各自跟注册者的分数 ——
        否则两个真人都沾边时会被错误合并成一个人。"""
        strong = near(self.me, 0.05, seed=11)
        weak = near(self.me, 0.9, seed=13)          # 勉强沾边，但跟 strong 不像
        m = vp.match_speakers({"S0": strong, "S1": weak}, self.people,
                              merge_clusters=True, threshold=0.4, margin=0.0)
        self.assertEqual(m.get("S0"), "张三")
        self.assertIsNone(m.get("S1"), "跟主簇不相似的不该被合并进来")

    def test_带分数时多簇各自带自己的分数(self):
        a, b = near(self.me, 0.05, seed=11), near(self.me, 0.08, seed=12)
        m = vp.match_speakers({"S0": a, "S1": b}, self.people,
                              merge_clusters=True, scores=True)
        self.assertEqual(m["S0"][0], "张三")
        self.assertEqual(m["S1"][0], "张三")
        self.assertNotEqual(m["S0"][1], m["S1"][1], "两簇的相似度不该被抹成一样")


if __name__ == "__main__":
    unittest.main(verbosity=2)
