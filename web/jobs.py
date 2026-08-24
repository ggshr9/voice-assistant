"""会议纪要生成任务:子进程跑 meeting/meeting_pipeline.py(分人 + ASR + LLM),轮询取进度。"""
import os, asyncio
from config import HERE
from sessions import sess_dir, read_meta, write_meta

JOBS = {}
MEETING_PY = os.path.join(HERE, "meeting", "meeting_pipeline.py")
PY_MEETING = os.path.expanduser("~/voice-svc/.venv-meeting/bin/python")
_KEEP = ("准备", "转写", "分人", "合并", "生成", "纪要", "记录", "完成", "transcribed", "diarized")


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
        async for line in proc.stdout:
            s = line.decode("utf-8", "ignore").strip()
            if s and any(k in s for k in _KEEP):
                JOBS[job]["progress"].append(s)
        await proc.wait()
        m, r = os.path.join(outdir, "会议纪要.md"), os.path.join(outdir, "会议记录.md")
        if os.path.exists(m) and os.path.exists(r):
            JOBS[job]["minutes"] = open(m, encoding="utf-8").read()
            JOBS[job]["record"] = open(r, encoding="utf-8").read()
            try:                        # 删 pipeline 中间产物 audio.wav(大),录音已另存 opus
                os.remove(os.path.join(outdir, "audio.wav"))
            except OSError:
                pass
            ok = True
        else:
            JOBS[job]["error"] = "处理未产出结果"
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
