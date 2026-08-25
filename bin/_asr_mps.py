#!/usr/bin/env python3
"""mlx-qwen3-asr 的包装器:把声纹分人搬到 GPU(MPS)上跑。

**为什么需要这个**:上游 mlx-qwen3-asr(0.3.5,已是最新)创建 pyannote pipeline 后
从不 `.to(device)` —— `diarization.py:258` 就一句 `Pipeline.from_pretrained(...)`,
于是分人一直在 CPU 上跑,M5 Max 的 GPU 全程闲着。上游没有任何 device 选项。

**实测**(265 秒真实会议音频,pyannote/speaker-diarization-community-1):

    CPU   117.5s   2.21x 实时
    MPS     4.4s  59.32x 实时     ← 快 26.7 倍

质量零代价:87 段边界完全相同、8 个说话人标签一一对应,按 0.1 秒栅格逐帧比对
说话人归属**一致率 100.00%**。

**做法**:不改上游任何一行,只在运行时把模块级私有函数 `_load_pyannote_pipeline`
包一层(它无参数、自带缓存,是干净的挂钩点),加载完把 pipeline 搬到 MPS,再调
上游自己的 CLI。`uv tool upgrade` 后依然生效;上游若哪天重构掉这个函数,会在
stderr 大声告警并退回 CPU 慢速路径 —— 慢但不会让转写挂掉。

用法与 mlx-qwen3-asr 完全一致,参数原样透传:
    <mlx-qwen3-asr 所在 venv 的 python> _asr_mps.py <音频> --diarize ...

同一个钩子还负责**截获声纹向量**:上游从 DiarizeOutput 只取标注、把 256 维的
speaker_embeddings 扔了。而事后另跑一次 pyannote 拿不到能用的向量 —— 第二次的
聚类结果不保证与第一次一致,标签对不上就全错。所以必须在这一次调用里截下来。

环境变量:
    MEETING_DIARIZE_DEVICE=cpu    强制掰回 CPU(出问题时的退路)
    MEETING_EMBED_OUT=<路径>      把本次分人的说话人向量写到这里(.npz)
"""
import os
import sys

BANNER = "[_asr_mps]"


def warn(msg):
    print(f"{BANNER} ⚠️  {msg}", file=sys.stderr, flush=True)


def pick_device(env, mps_available):
    """选分人跑在哪:显式指定优先,否则有 MPS 就用 MPS。"""
    want = (env.get("MEETING_DIARIZE_DEVICE") or "").strip().lower()
    if want:
        return want
    return "mps" if mps_available else "cpu"


def needs_patch(argv):
    """只有真要分人时才值得 import torch(否则白等 2 秒)。"""
    return "--diarize" in argv


def patch_diarization(diar_module, device, torch_module, warn_fn):
    """把 _load_pyannote_pipeline 包一层,加载后搬到 device。

    返回是否挂钩成功。挂钩是惰性的 —— 不在这里触发加载,交给上游的缓存逻辑。
    """
    orig = getattr(diar_module, "_load_pyannote_pipeline", None)
    if not callable(orig):
        warn_fn(
            "上游 mlx_qwen3_asr.diarization 里找不到 _load_pyannote_pipeline，"
            "无法把分人搬到 GPU —— 大概是上游重构了。这次会退回 CPU 跑分人（慢约 27 倍）。"
        )
        return False

    def loader():
        pipeline = orig()
        try:
            pipeline.to(torch_module.device(device))
        except Exception as exc:
            warn_fn(f"pipeline 搬到 {device} 失败({type(exc).__name__}: {exc})，退回 CPU。")
        return _wrap_capture(pipeline, warn_fn)

    diar_module._load_pyannote_pipeline = loader
    return True


class _CapturingPipeline:
    """代理 pyannote pipeline,在它被调用时把 speaker_embeddings 落盘。

    **为什么要代理而不是给实例赋 __call__**:Python 查 dunder 方法是查【类型】不是
    实例,`pipeline.__call__ = wrapped` 不会改变 `pipeline(...)` 的行为 —— 而且
    不报错,只是静默没生效(第一版就这么写的,截获一直是空的)。
    """

    def __init__(self, inner, out_path, warn_fn):
        self._inner = inner
        self._out_path = out_path
        self._warn = warn_fn

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        try:
            import numpy as np
            emb = getattr(out, "speaker_embeddings", None)
            ann = getattr(out, "speaker_diarization", None)
            if emb is not None and ann is not None:
                np.savez(self._out_path,
                         embeddings=np.asarray(emb),
                         labels=np.array(list(ann.labels())))
        except Exception as exc:                      # noqa: BLE001
            self._warn(f"声纹向量没截到({type(exc).__name__}: {exc})，不影响转写。")
        return out

    def __getattr__(self, name):                      # 其余属性一律透传
        return getattr(self._inner, name)


def _wrap_capture(pipeline, warn_fn, out_path=None):
    """设了 MEETING_EMBED_OUT 才包;否则原样返回,零开销。"""
    out_path = out_path or os.environ.get("MEETING_EMBED_OUT")
    if not out_path:
        return pipeline
    return _CapturingPipeline(pipeline, out_path, warn_fn)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if needs_patch(argv):
        # 有算子不支持 MPS 时让 torch 自己退回 CPU，而不是整个崩掉
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch
            import mlx_qwen3_asr.diarization as diar
        except ImportError as exc:
            warn(f"import 失败({exc})，按上游默认行为继续。")
        else:
            device = pick_device(os.environ, torch.backends.mps.is_available())
            if device == "cpu":
                warn("分人跑在 CPU 上（慢约 27 倍）。不想这样就别设 MEETING_DIARIZE_DEVICE=cpu。")
            else:
                patch_diarization(diar, device, torch, warn)

    from mlx_qwen3_asr.cli import main as cli_main

    sys.argv = ["mlx-qwen3-asr"] + argv
    return cli_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
