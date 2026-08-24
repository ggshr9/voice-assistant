"""分人 + 逐轮 Qwen3-ASR(都 torch,同进程跑)→ 直接产出 说话人A/B/C 标注稿。
中文用 Qwen3-ASR(中文 SOTA,无 whisper 幻觉)。用法: python asr_diarize_step.py <wav> <out.txt> [语言]
语言: Chinese / English / auto(默认 Chinese)"""
import sys, os, string, torch, numpy as np
from scipy.io import wavfile
from pyannote.audio import Pipeline
from qwen_asr import Qwen3ASRModel

wav_path, out_txt = sys.argv[1], sys.argv[2]
lang_arg = sys.argv[3] if len(sys.argv) > 3 else "Chinese"
LANG = None if lang_arg in ("auto", "", None) else lang_arg

sr, raw = wavfile.read(wav_path)
wav = raw.astype("float32") / 32768.0

# ---- 分人 ----
dpipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                 token=os.environ["HF_TOKEN"]).to(torch.device("cuda"))
dia = dpipe({"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": sr})
ann = getattr(dia, "speaker_diarization", dia)
turns = sorted((t.start, t.end, s) for t, _, s in ann.itertracks(yield_label=True))
# 连续同人且间隔<1s 合并成一轮(减少 ASR 次数 + 更通顺)
merged = []
for st, en, sp in turns:
    if merged and merged[-1][2] == sp and st - merged[-1][1] < 1.0:
        merged[-1][1] = en
    else:
        merged.append([st, en, sp])
print(f"diarized {len(turns)} turns -> {len(merged)} merged", flush=True)

# ---- 逐轮 ASR ----
asr = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16,
                                    device_map="cuda:0", max_new_tokens=2048)
order, lines = {}, []
for st, en, sp in merged:
    if en - st < 0.4:
        continue
    clip = wav[int(st * sr):int(en * sr)]
    try:
        r = asr.transcribe(audio=(clip, sr), language=LANG)
        text = (r[0].text if r else "").strip()
    except Exception as e:
        text = ""
    if not text:
        continue
    if sp not in order:
        i = len(order)
        order[sp] = f"说话人{string.ascii_uppercase[i]}" if i < 26 else f"说话人{i + 1}"
    lines.append(f"{order[sp]}：{text}")

open(out_txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"ASR done: {len(order)} speakers, {len(lines)} lines", flush=True)
