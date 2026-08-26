# -*- coding: utf-8 -*-
"""原子写文件。

**为什么单独抽出来**：这个项目已经在同一个错误上栽过三次 ——
录音的 m4a（ffmpeg 被强杀，moov 没写，786KB 音频救不回来）、
会议索引（写到一半 Ctrl+C，11778 字节变 28 字节）、
声纹库（19970 字节变 26 字节，而声纹重建要每个人重新录一遍）。
模式完全一样：**截断在先、内容在后**。`open(path, "w")` 那一刻文件就空了，
后面任何异常都留下一个半截文件——而且往往看起来是完整的。

规矩很简单：写同目录的 .tmp → fsync → os.replace。
replace 在同一文件系统上是原子的，崩溃时要么是旧内容、要么是新内容，没有中间态。

`bin/_voiceprint.py` 里有一份近乎相同的实现，那是**刻意的**：
它会被 sync-web 推到服务器单独使用，跨文件 import 会在那边断掉。
十行重复胜过一个易碎的依赖。改这里时记得同步看一眼那边。
"""
import os
import tempfile

DEFAULT_MODE = 0o644


def atomic_write(path, text, mode=None, encoding="utf-8"):
    """把 text 原子地写到 path。

    Args:
        path: 目标路径。
        text: 全部内容（这个工具只做全量覆写，不做追加）。
        mode: 权限。``None`` 表示沿用目标文件现有权限；文件不存在时用
            ``DEFAULT_MODE``。**含密钥的文件要显式传 0o600** ——
            别指望"写完再 chmod"，那之间有一段时间文件是按 umask 敞着的。

    Returns:
        path
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    if mode is None:
        try:
            mode = os.stat(path).st_mode & 0o777   # 别把原文件权限悄悄改掉
        except OSError:
            mode = DEFAULT_MODE
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())                   # 断电也不留半截
        os.chmod(tmp, mode)                        # 在 replace 之前就位,没有敞开的窗口
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
