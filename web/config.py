"""会议工作台共享配置(环境变量 + 常量)。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.expanduser("~/voice-svc/sessions")
os.makedirs(SESSIONS, exist_ok=True)

MODEL = os.environ.get("STT_MODEL", "large-v3-turbo")
LLM_URL = os.environ.get("CAPTION_LLM_URL", "")   # 必须显式配置(线上由 secrets.env 注入)
LLM_KEY = os.environ.get("CAPTION_LLM_KEY", "")
LLM_MODEL = os.environ.get("CAPTION_LLM_MODEL", "Qwen3.6")

SR, FRAME = 16000, 480           # 16k 采样, 30ms 帧
SILENCE_TAIL, MIN_SPEECH = 0.8, 0.4
ACCESS_PW = os.environ.get("CAPTION_ACCESS_PW", "")   # 设了就要口令

# 录音存储:停止后把 wav 转 Opus(~10x 小);保留策略与磁盘告警
# 一段最长攒这么久就硬断。只靠静音断句不够:有人一口气说两分钟,
# 就攒出一个两分钟的段 —— 既拖垮延迟,也让模型更容易跑飞。
MAX_SEG_SEC = float(os.environ.get("CAPTION_MAX_SEG_SEC", "15"))
RETENTION_DAYS = int(os.environ.get("CAPTION_RETENTION_DAYS", "0"))      # >0 启用:删超期录音文件(保留文字/纪要)
SESSIONS_WARN_GB = float(os.environ.get("CAPTION_SESSIONS_WARN_GB", "50"))
# 单次上传上限。没有上限的话,一个请求就能把盘写满(而且从前连口令都不用对)。
MAX_UPLOAD_BYTES = int(float(os.environ.get("CAPTION_MAX_UPLOAD_GB", "4")) * (1 << 30))
