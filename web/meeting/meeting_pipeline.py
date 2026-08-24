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
    # 线上会议(双声道 L=对方/R=我)走声道分人;其余下混走 pyannote
    scene = ""
    try:
        scene = json.load(open(os.path.join(outdir, "meta.json"), encoding="utf-8")).get("scene", "")
    except Exception:
        pass
    roles = "1" if scene == "线上" else "0"
    wav = os.path.join(outdir, "audio.wav")
    log("准备：转 16k wav")
    ff = ["ffmpeg", "-y", "-i", audio, "-ar", "16000", wav]
    if roles != "1":                       # 非线上:下混单声道
        ff[5:5] = ["-ac", "1"]
    subprocess.run(ff, check=True, capture_output=True)
    if roles == "1" and not me:            # 线上:转写里"我"就是本人
        me = "我"

    ann = os.path.join(outdir, "annotated.txt")
    log(f"分人 + 转写中（{'声道分轨 ' if roles == '1' else ''}pyannote + Qwen3-ASR / 4090，语言={lang}）")
    subprocess.run([PY_DIA, os.path.join(HERE, "asr_diarize_step.py"), wav, ann, _LANG.get(lang, "Chinese"), roles],
                   check=True, env=os.environ)
    text = open(ann, encoding="utf-8").read().strip()
    nspk = len(set(re.findall(r"^(?:说话人[A-Z]|我|对方)", text, re.M)))
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
