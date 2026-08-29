"""会议批处理编排:音频 → 转写(ctranslate2 venv) + 分人(torch venv) → 合并 → 网关出纪要+记录。
只用 stdlib + 子进程,任何 python3 都能跑。
用法: python meeting_pipeline.py <音频> <输出目录> [我是说话人X]"""
import sys, os, re, json, glob, subprocess

HOME = os.path.expanduser("~")
SVC = os.path.join(HOME, "voice-svc")
HERE = os.path.dirname(os.path.abspath(__file__))
# 解释器可用环境变量顶掉:裸机部署是两个 venv,容器里全栈装在同一个解释器
PY_STT = os.environ.get("MEETING_PY_STT", os.path.join(SVC, ".venv/bin/python"))
PY_DIA = os.environ.get("MEETING_PY_DIA", os.path.join(SVC, ".venv-meeting/bin/python"))

sys.path.insert(0, HERE)
import minutes_lib


def _ld():
    cublas = glob.glob(os.path.join(SVC, ".venv/**/libcublas.so*"), recursive=True)
    cudnn = glob.glob(os.path.join(SVC, ".venv/**/libcudnn.so*"), recursive=True)
    p = []
    if cublas: p.append(os.path.dirname(cublas[0]))
    if cudnn: p.append(os.path.dirname(cudnn[0]))
    return ":".join(p)


# 注意没有 "auto" 键:qwen_asr 不认 "auto"(validate 直接抛),而 asr_diarize_step
# 的 per-clip except 会把异常吞成 text="" —— 用户选「自动」= 整场空转写,还查无此错。
# auto 走 .get 的默认值落到 Chinese。
_LANG = {"zh": "Chinese", "en": "English"}


def to_wav_cmd(audio, wav, roles):
    """转 16k wav 的 ffmpeg 命令。roles=="1"(线上双声道)保留声道,否则下混单声道。

    **为什么单独抽出来**:原先是 `ff[5:5] = ["-ac","1"]` 按下标往里插,插的位置
    正好落在 `-ar` 和 `16000` 中间,生成 `-ar -ac 1 16000` —— ffmpeg 退出 234。
    因为只在非「线上」场景触发,线上会议一路正常,这个 bug 活了两个月才被发现,
    期间**每一次上传都必然失败**。改成显式拼接,并让它可被测试直接断言。
    """
    cmd = ["ffmpeg", "-y", "-i", audio, "-ar", "16000"]
    if roles != "1":
        cmd += ["-ac", "1"]
    return cmd + [wav]


# 素材质检阈值:语音占比低于此值,产物头部打显式警告。
# 取 0.25:正常会议实测 0.5+,背景音乐 0.0;夹在中间的是"少量说话+大量放东西"。
SPEECH_RATIO_WARN = float(os.environ.get("MEETING_SPEECH_RATIO_WARN", "0.25"))


def material_warning(stats):
    """边车统计 → 该打进产物头部的警告文本;素材正常返回 None。

    警告必须写进【产物本身】而不是日志:用户只看纪要,曾拿着一场被硬转的
    背景音乐评估了半天系统 —— 整条链路当时零提示。
    """
    if not stats:
        return None
    ratio = stats.get("speech_ratio", 1)
    fallback = stats.get("diarization_fallback")
    if ratio < SPEECH_RATIO_WARN:
        return (f"> ⚠️ **素材可疑**:本段音频仅 {ratio:.0%} 被判定为人声"
                f"(正常会议约 50%)。大概率是背景音乐/播放内容,以下转写不可尽信。\n")
    if fallback:
        return ("> ⚠️ 说话人分离未成功,已降级为单说话人 —— "
                "「说话人A」不代表真的只有一个人。\n")
    return None


def run(audio, outdir, me=None, lang="zh", log=print):
    os.makedirs(outdir, exist_ok=True)
    # 线上会议(双声道 L=对方/R=我)走声道分人;其余下混走 pyannote
    scene = ""
    try:
        scene = json.load(open(os.path.join(outdir, "meta.json"), encoding="utf-8")).get("scene", "")
    except Exception:
        pass
    roles = "1" if scene == "线上" else "0"
    template = ""
    try:
        template = json.load(open(os.path.join(outdir, "meta.json"), encoding="utf-8")).get("template", "")
    except Exception:
        pass
    wav = os.path.join(outdir, "audio.wav")
    log("准备：转 16k wav")
    if os.path.realpath(audio) == os.path.realpath(wav):
        # 对同一会话重跑时输入就是上次产出的 audio.wav —— ffmpeg 拒绝原地覆盖,
        # 而它本来就是我们自己转出的 16k 单声道,直接复用
        log("输入已是 16k wav,跳过转换")
    else:
        r = subprocess.run(to_wav_cmd(audio, wav, roles), capture_output=True)
        if r.returncode != 0:              # 带上 ffmpeg 自己的话,否则只剩一句"处理未产出结果"
            raise RuntimeError("转 wav 失败：" + r.stderr.decode("utf-8", "ignore")[-400:])
    if roles == "1" and not me:            # 线上:转写里"我"就是本人
        me = "我"

    ann = os.path.join(outdir, "annotated.txt")
    log(f"分人 + 转写中（{'声道分轨 ' if roles == '1' else ''}pyannote + Qwen3-ASR / 4090，语言={lang}）")
    subprocess.run([PY_DIA, os.path.join(HERE, "asr_diarize_step.py"), wav, ann, _LANG.get(lang, "Chinese"), roles],
                   check=True, env=os.environ)
    text = open(ann, encoding="utf-8").read().strip()
    stats = {}
    try:
        stats = json.load(open(os.path.join(outdir, "diar_stats.json"), encoding="utf-8"))
        m0 = json.load(open(os.path.join(outdir, "meta.json"), encoding="utf-8"))
        m0["speech_ratio"] = stats.get("speech_ratio")
        m0["diarization_fallback"] = stats.get("diarization_fallback")
        json.dump(m0, open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
    except Exception:                          # noqa: BLE001  质检缺失不拦主流程
        pass
    warn = material_warning(stats)
    nspk = len(set(re.findall(r"^(?:说话人[A-Z]|我|对方)", text, re.M)))
    log(f"转写完成：{nspk} 人")

    log("生成会议纪要（网关 Qwen3.6）")
    minutes = minutes_lib.make_minutes(text, me, log, template=template)
    if warn:
        minutes = warn + "\n" + minutes
    open(os.path.join(outdir, "会议纪要.md"), "w", encoding="utf-8").write(minutes)

    log("生成会议记录（忠实还原）")
    record = minutes_lib.make_record(text, log)
    if warn:
        record = warn + "\n" + record
    open(os.path.join(outdir, "会议记录.md"), "w", encoding="utf-8").write(record)

    # 用户开会时记了手记 → 多产出一份「增强笔记」(用户笔记为骨架,转写补全)
    enhanced = ""
    notes_p = os.path.join(outdir, "notes.md")
    if os.path.exists(notes_p):
        user_notes = open(notes_p, encoding="utf-8").read()
        if user_notes.strip():
            enhanced = minutes_lib.make_enhanced(user_notes, text, log)
            if enhanced:
                tmp = os.path.join(outdir, ".增强笔记.md.part")
                open(tmp, "w", encoding="utf-8").write(enhanced)
                os.replace(tmp, os.path.join(outdir, "增强笔记.md"))

    log("完成")
    return {"speakers": nspk, "minutes": minutes, "record": record, "enhanced": enhanced}


if __name__ == "__main__":
    audio = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/meeting_out"
    me = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    lang = sys.argv[4] if len(sys.argv) > 4 else "zh"
    r = run(audio, outdir, me, lang)
    print("DONE", r["speakers"], "speakers")
