"""分人 + 逐轮 Qwen3-ASR(都 torch,同进程跑)→ 直接产出 说话人A/B/C(或 我/对方) 标注稿。
中文用 Qwen3-ASR(中文 SOTA,无 whisper 幻觉)。
用法: python asr_diarize_step.py <wav> <out.txt> [语言] [声道角色]
语言: Chinese / English / auto(默认 Chinese)
声道角色: 1=双声道分轨(L=对方/R=我,线上会议)按声道贴"我/对方";其余=纯 pyannote 说话人A/B"""
import sys, os, string, torch, numpy as np
from scipy.io import wavfile
from pyannote.audio import Pipeline
from qwen_asr import Qwen3ASRModel

wav_path, out_txt = sys.argv[1], sys.argv[2]
lang_arg = sys.argv[3] if len(sys.argv) > 3 else "Chinese"
roles_arg = sys.argv[4] if len(sys.argv) > 4 else "0"
LANG = None if lang_arg in ("auto", "", None) else lang_arg

sr, raw = wavfile.read(wav_path)
if raw.ndim == 2 and raw.shape[1] >= 2:          # 双声道:L=对方, R=我
    Lf = raw[:, 0].astype("float32") / 32768.0
    Rf = raw[:, 1].astype("float32") / 32768.0
    wav = (Lf + Rf) / 2.0                         # pyannote/ASR 用下混
    stereo = True
else:
    wav = raw.astype("float32") / 32768.0
    Lf = Rf = None
    stereo = False
ROLES = (roles_arg == "1") and stereo             # 启用声道贴标
print(f"input: stereo={stereo} roles={ROLES}", flush=True)

# ---- 分人 ----
dpipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                 token=os.environ["HF_TOKEN"]).to(torch.device("cuda"))
# 调高声纹聚类阈值(默认 0.6→0.72):声音相近就不拆开,从源头少把一个人误拆成多人(VBx,越高越保守)
try:
    _p = dpipe.parameters(instantiated=True)
    _p["clustering"]["threshold"] = float(os.environ.get("DIA_CLUSTER_THRESHOLD", "0.72"))
    dpipe.instantiate(_p)
    print(f"clustering threshold = {_p['clustering']['threshold']}", flush=True)
except Exception as e:
    print("instantiate threshold skipped:", e, flush=True)


EMBEDDINGS = {}          # 本次分人的 {说话人: 256维向量},供声纹认人用


def diarize(signal):
    """对单声道信号跑 pyannote,返回合并后的轮次 [[st,en,sp],...]。

    顺带把 speaker_embeddings 存进 EMBEDDINGS —— 必须用这一次调用的向量,
    事后另跑一遍 pyannote 的聚类不保证一致,标签会对不上。
    """
    try:
        dia = dpipe({"waveform": torch.from_numpy(signal).unsqueeze(0), "sample_rate": sr})
    except Exception:
        return []
    ann = getattr(dia, "speaker_diarization", dia)
    try:
        emb = getattr(dia, "speaker_embeddings", None)
        if emb is not None:
            for i, lbl in enumerate(ann.labels()):
                EMBEDDINGS[lbl] = np.asarray(emb)[i]
    except Exception as e:
        print("embeddings skipped:", e, flush=True)
    turns = sorted((t.start, t.end, s) for t, _, s in ann.itertracks(yield_label=True))
    merged = []
    for st, en, sp in turns:                     # 同人且间隔<1s 合并一轮
        if merged and merged[-1][2] == sp and st - merged[-1][1] < 1.0:
            merged[-1][1] = en
        else:
            merged.append([st, en, sp])
    return merged


def _letter(i):
    return string.ascii_uppercase[i] if i < 26 else str(i + 1)


def _collapse(turns, min_dur=4.0, min_frac=0.18):
    """安全网:把说话时长过短的'幽灵说话人'(超短句 embedding 不稳致 pyannote 误拆)并进
    该声道主要说话人。与聚类阈值互补——阈值治音色相近的误拆,这个治超短句的误拆。"""
    if not turns:
        return turns, []
    dur = {}
    for st, en, sp in turns:
        dur[sp] = dur.get(sp, 0.0) + (en - st)
    total = sum(dur.values())
    dom = max(dur, key=dur.get)
    keep = {sp for sp, d in dur.items() if d >= max(min_dur, min_frac * total)} or {dom}
    remap = {sp: (sp if sp in keep else dom) for sp in dur}
    out = [[st, en, remap[sp]] for st, en, sp in turns]
    uniq = list(dict.fromkeys(remap[sp] for _, _, sp in turns))
    return out, uniq


# 收集待转写任务: (st, en, 取音信号, 标签)。每段从其所属声道取音,更干净。
tasks = []
if ROLES:                                        # 左右声道各自分人,按人数自动贴标(零选项)
    Lturns, Lspk = _collapse(diarize(Lf))        # L=远端=对方
    Rturns, Rspk = _collapse(diarize(Rf))        # R=本地=我/现场
    for st, en, sp in Lturns:
        lab = ("对方" + _letter(Lspk.index(sp))) if len(Lspk) > 1 else "对方"
        tasks.append((st, en, Lf, lab))
    for st, en, sp in Rturns:
        lab = ("现场" + _letter(Rspk.index(sp))) if len(Rspk) > 1 else "我"
        tasks.append((st, en, Rf, lab))
    print(f"diarized 对方{len(Lspk)}人 / 本地{len(Rspk)}人", flush=True)
else:                                            # 单声道(线下/上传):纯 pyannote 说话人A/B
    turns = diarize(wav)
    identified = {}
    try:                                         # 声纹认人:认出来用真名,认不出保留匿名牌
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))))
        import _voiceprint as _vp
        people = _vp.load_registry(os.path.expanduser("~/voice-svc/voiceprints.json"))
        if people and EMBEDDINGS:
            identified = _vp.match_speakers(EMBEDDINGS, people)
            if identified:
                print(f"voiceprint matched: {identified}", flush=True)
    except Exception as e:
        print("voiceprint skipped:", e, flush=True)

    order, anon = {}, 0
    for st, en, sp in turns:
        if sp not in order:
            if sp in identified:                 # 真名不占匿名牌字母,免得跳号
                order[sp] = identified[sp]
            else:
                order[sp] = f"说话人{_letter(anon)}"
                anon += 1
        tasks.append((st, en, wav, order[sp]))
    print(f"diarized {len(order)} speakers", flush=True)
tasks.sort(key=lambda x: x[0])

# ---- 逐轮 ASR ----
asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16,
                                    device_map="cuda:0", max_new_tokens=2048)
lines, spks = [], set()
for st, en, sig, lab in tasks:
    if en - st < 0.4:
        continue
    clip = sig[int(st * sr):int(en * sr)]
    try:
        r = asr.transcribe(audio=(clip, sr), language=LANG)
        text = (r[0].text if r else "").strip()
    except Exception:
        text = ""
    if not text:
        continue
    spks.add(lab)
    lines.append(f"{lab}：{text}")

open(out_txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"ASR done: {len(spks)} speakers, {len(lines)} lines", flush=True)
