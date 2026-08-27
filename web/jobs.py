"""会议纪要生成任务:子进程跑 meeting/meeting_pipeline.py(分人 + ASR + LLM),轮询取进度。"""
import os, asyncio, collections
from config import HERE
from sessions import sess_dir, read_meta, write_meta

JOBS = {}
MEETING_PY = os.path.join(HERE, "meeting", "meeting_pipeline.py")
PY_MEETING = os.path.expanduser("~/voice-svc/.venv-meeting/bin/python")
_KEEP = ("准备", "转写", "分人", "合并", "生成", "纪要", "记录", "完成", "transcribed", "diarized")
TAIL_LINES = 40


def fail_reason(rc, tail):
    """把子进程真实的死因带出来。

    **为什么必须有**:原先失败只写死一句"处理未产出结果",而进度过滤器只留含关键词
    的行,traceback 全被丢掉 —— systemd 日志里也一行都没有。曾经因此有个 ffmpeg
    参数顺序的 bug 让上传功能整整坏了两个月都没人知道为什么。
    """
    for line in reversed(tail):          # 最后一条像错误的行最接近真因
        if any(k in line for k in ("Error", "error", "Traceback", "Exception",
                                   "Failed", "失败", "No such file")):
            return f"处理失败(退出码 {rc})：{line[:300]}"
    if tail:
        return f"处理未产出结果(退出码 {rc})：{tail[-1][:300]}"
    return f"处理未产出结果(退出码 {rc}，子进程无任何输出)"


def prune_jobs(cap=60):
    """只保留最近的任务,清掉已完成的旧任务,避免 JOBS 无限增长。"""
    if len(JOBS) <= cap:
        return
    done = [k for k, v in JOBS.items() if v.get("done")]
    for k in done[:len(JOBS) - cap]:
        JOBS.pop(k, None)


async def run_meeting_job(job, audio, outdir, lang, me):
    ok = False
    try:
        proc = await asyncio.create_subprocess_exec(
            PY_MEETING, MEETING_PY, audio, outdir, me or "-", lang or "zh",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PATH": "/usr/bin:" + os.environ.get("PATH", "")})
        tail = collections.deque(maxlen=TAIL_LINES)   # 见 fail_reason:失败时唯一的线索来源
        async for line in proc.stdout:
            s = line.decode("utf-8", "ignore").strip()
            if not s:
                continue
            tail.append(s)
            if any(k in s for k in _KEEP):
                JOBS[job]["progress"].append(s)
        rc = await proc.wait()
        m, r = os.path.join(outdir, "会议纪要.md"), os.path.join(outdir, "会议记录.md")
        if os.path.exists(m) and os.path.exists(r):
            JOBS[job]["minutes"] = open(m, encoding="utf-8").read()
            JOBS[job]["record"] = open(r, encoding="utf-8").read()
            ep = os.path.join(outdir, "增强笔记.md")
            if os.path.exists(ep):                   # 有手记才有这份,可选
                JOBS[job]["enhanced"] = open(ep, encoding="utf-8").read()
            try:                        # 删 pipeline 中间产物 audio.wav(大),录音已另存 opus
                os.remove(os.path.join(outdir, "audio.wav"))
            except OSError:
                pass
            ok = True
        else:
            JOBS[job]["error"] = fail_reason(rc, list(tail))
    except Exception as e:
        JOBS[job]["error"] = str(e)[:120]
    # 更新 session 状态:成功→done;失败→退回 recorded(别卡在"生成中",可重试)
    if JOBS[job].get("sid"):
        sd = sess_dir(JOBS[job]["sid"])
        if sd:
            meta = read_meta(sd) or {}
            meta["status"] = "done" if ok else "recorded"
            write_meta(sd, meta)
    JOBS[job]["done"] = True
