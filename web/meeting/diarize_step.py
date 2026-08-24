"""分人步(在 torch/pyannote venv 跑):wav → JSON 说话人轮次。
用法: HF_TOKEN=... python diarize_step.py <wav> <out.json>"""
import sys, json, os, torch
from scipy.io import wavfile
from pyannote.audio import Pipeline

sr, wav = wavfile.read(sys.argv[1])
wav = wav.astype("float32") / 32768.0
wt = torch.from_numpy(wav).unsqueeze(0)
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                token=os.environ["HF_TOKEN"]).to(torch.device("cuda"))
dia = pipe({"waveform": wt, "sample_rate": sr})
ann = getattr(dia, "speaker_diarization", dia)
turns = [{"start": round(t.start, 2), "end": round(t.end, 2), "speaker": s}
         for t, _, s in ann.itertracks(yield_label=True)]
json.dump(turns, open(sys.argv[2], "w"))
print(f"diarized {len(turns)} turns, {len(set(t['speaker'] for t in turns))} speakers", flush=True)
