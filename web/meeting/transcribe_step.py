"""转写步(在 ctranslate2 venv 跑):wav → JSON 段(带时间戳)。
用法: python transcribe_step.py <wav> <out.json>"""
import sys, json
from faster_whisper import WhisperModel

lang = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("auto", "") else None
m = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segs, info = m.transcribe(sys.argv[1], language=lang, condition_on_previous_text=False)
out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in segs]
json.dump({"language": info.language, "segments": out},
          open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
print(f"transcribed {len(out)} segments, lang={info.language}", flush=True)
