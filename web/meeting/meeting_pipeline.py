"""会议批处理编排:音频 → 转写(ctranslate2 venv) + 分人(torch venv) → 合并 → 网关出纪要+记录。
只用 stdlib + 子进程,任何 python3 都能跑。
用法: python meeting_pipeline.py <音频> <输出目录> [我是说话人X]"""
import sys, os, re, json, glob, subprocess

HOME = os.path.expanduser("~")
SVC = os.path.join(HOME, "voice-svc")
HERE = os.path.dirname(os.path.abspath(__file__))
PY_STT = os.path.join(SVC, ".venv/bin/python")          # faster-whisper(ctranslate2)
PY_DIA = os.path.join(SVC, ".venv-meeting/bin/python")  # pyannote(torch)

sys.path.insert(0, HERE)
import minutes_lib


def _ld():
    cublas = glob.glob(os.path.join(SVC, ".venv/**/libcublas.so*"), recursive=True)
    cudnn = glob.glob(os.path.join(SVC, ".venv/**/libcudnn.so*"), recursive=True)
    p = []
    if cublas: p.append(os.path.dirname(cublas[0]))
    if cudnn: p.append(os.path.dirname(cudnn[0]))
    return ":".join(p)


_LANG = {"zh": "Chinese", "en": "English", "auto": "auto"}


def run(audio, outdir, me=None, lang="zh", log=print):
    os.makedirs(outdir, exist_ok=True)
    wav = os.path.join(outdir, "audio.wav")
    log("准备：转 16k wav")
    subprocess.run(["ffmpeg", "-y", "-i", audio, "-ac", "1", "-ar", "16000", wav],
                   check=True, capture_output=True)

    ann = os.path.join(outdir, "annotated.txt")
    log(f"分人 + 转写中（pyannote + Qwen3-ASR / 4090，语言={lang}）")
    subprocess.run([PY_DIA, os.path.join(HERE, "asr_diarize_step.py"), wav, ann, _LANG.get(lang, "Chinese")],
                   check=True, env=os.environ)
    text = open(ann, encoding="utf-8").read().strip()
    nspk = len(set(re.findall(r"^说话人[A-Z]", text, re.M)))
    log(f"转写完成：{nspk} 人")

    log("生成会议纪要（网关 Qwen3.6）")
    minutes = minutes_lib.make_minutes(text, me, log)
    open(os.path.join(outdir, "会议纪要.md"), "w", encoding="utf-8").write(minutes)

    log("生成会议记录（忠实还原）")
    record = minutes_lib.make_record(text, log)
    open(os.path.join(outdir, "会议记录.md"), "w", encoding="utf-8").write(record)

    log("完成")
    return {"speakers": nspk, "minutes": minutes, "record": record}


if __name__ == "__main__":
    audio = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/meeting_out"
    me = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    lang = sys.argv[4] if len(sys.argv) > 4 else "zh"
    r = run(audio, outdir, me, lang)
    print("DONE", r["speakers"], "speakers")
