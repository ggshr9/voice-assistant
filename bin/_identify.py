#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拿转写目录里截获的声纹向量去注册表认人。

stdout 输出 `SPEAKER_00=李四,SPEAKER_03=张三` 供 _annotate.py 消费；
认不出任何人就什么都不输出（调用方据此走匿名路径）。

认人失败一律静默退出 —— 声纹是附加能力，绝不该让转写本身失败。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))


def main(argv):
    if not argv:
        return 1
    try:
        import numpy as np
        import _voiceprint as vp

        path = os.path.join(argv[0], ".embeddings.npz")
        if not os.path.exists(path):
            return 0
        people = vp.load_registry()
        if not people:
            return 0
        d = np.load(path, allow_pickle=True)
        speakers = {str(l): d["embeddings"][i] for i, l in enumerate(d["labels"])}
        mapping = vp.match_speakers(speakers, people, scores=True, merge_clusters=True)
        if mapping:
            # 带上相似度,让 _annotate 决定要不要标「勉强够线」
            print(",".join(f"{k}={n}:{sc}" for k, (n, sc) in sorted(mapping.items())))
            shown = "、".join(
                f"{n}" if vp.is_sure(sc) else f"{n}(勉强,{sc:.2f})"
                for n, sc in (mapping[k] for k in sorted(mapping)))
            print(f"@@STAGE identified {len(mapping)}", file=sys.stderr)
            print(f"🔊 声纹认出 {len(mapping)} 人：{shown}", file=sys.stderr)
    except Exception as exc:                                  # noqa: BLE001
        print(f"（声纹识别跳过：{type(exc).__name__}: {exc}）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
