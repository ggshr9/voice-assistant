# voice-assistant — 全本地语音栈（转写 / 会议纪要 / 实时字幕 / 语音对话 / 声音克隆）

一套跑在 Apple Silicon（M5 Max / 128GB）上的**全本地**语音工具集：录音 → ASR → 本地 LLM → 结构化纪要 / 同传字幕 / 克隆音回答。
除可选的服务器版流水线外，不依赖任何云服务。

## 三种用法

| 形态 | 入口 | 说明 |
|---|---|---|
| **CLI** | `bin/` | 主力。`rec` 录音 → `meeting` 转写分人 → `minutes` 出纪要 |
| **菜单栏 App** | `meeting_app.py` | macOS 顶栏 🎙️ 图标，点两下出纪要。只是壳，内部仍调 CLI |
| **网页版** | `web/` | 浏览器共享标签页声音 → 实时字幕；上传音频 → 会议纪要（服务器 CUDA 版，见下） |

## 快速开始

```bash
./install.sh          # 把 bin/* 软链到 ~/.local/bin
llm start             # 拉起本地大脑（vllm-mlx + Qwen3.6-35B-A3B-8bit，:8080）

rec online            # 录线上会议（需音频MIDI里建好聚合设备「会议录制」）
meeting ~/会议录音/线上会议_20260619_2301.m4a 8 --me 说话人1
# → 转写 + 声纹分人 + 自动生成 纪要_xxx.md
```

## 组件

### CLI（`bin/`）
- `rec` — 录音。`rec` 录麦克风，`rec online` 录聚合设备（线上会议：对方+自己）
- `meeting <文件> [说话人数|-] [nominutes] [--me 说话人N]` — 转写（mlx-qwen3-asr / Qwen3-ASR-1.7B）+ pyannote 声纹分人 + 自动调 `minutes`
- `minutes <转写目录|.txt|.json>` — 结构化纪要 Markdown：一句话摘要 / 关键决议 / 待办表格（分「我的-他人」）/ 讨论要点 / 风险。长会议自动 map-reduce 分块
- `llm {start|stop|status|log|watch|test|vision}` — 统一大脑服务，一个端口同时给 OpenAI 格式（`:8080/v1`）和 Anthropic 格式（`:8080/v1/messages`）
- `caption` — 实时字幕（外语→中文），底部双语浮窗
- `dictate` — 全局语音听写，热键切换式，转写后粘贴到光标处
- `chat` / `ask` — 连续对话，语音版 / 打字版双胞胎
- `va` — 全本地语音助手「小麦」
- `clone <参考音> <原文> <新内容>` — 声音克隆（IndexTTS-1.5）

### 本机服务（Python）
- `stt_server.py` — 常驻 STT（mlx-whisper large-v3-turbo），`:8082/transcribe`
- `clone_tts_server.py` — 常驻克隆 TTS，`:8083`
- `caption_core.py` / `caption_overlay.py` / `caption.py` — 实时字幕的纯逻辑 / 浮窗 / 编排器
- `assistant.py` / `chat.py` — 语音助手与连续对话实现
- `meeting_app.py` + `meeting_app_launch.sh` — rumps 菜单栏壳

### 网页版（`web/`）
- `web_caption.py` — aiohttp + WSS：浏览器推 16k PCM → VAD 切句 → faster-whisper(CUDA) → 网关翻译 → 推回 `{orig, zh}`
- `web/meeting/` — 会议批处理流水线：`transcribe_step` / `diarize_step` / `asr_diarize_step` / `minutes_lib` / `meeting_pipeline`

> ⚠️ 网页版是**面向 GPU 服务器**写的：走 `~/voice-svc` 下的 venv、faster-whisper + pyannote + CUDA，纪要走 litellm 网关。与本机 MLX 那套是两条独立实现，本机目前没有 `~/voice-svc`，跑不起来。

## 配置

所有外部地址与 key 都走环境变量，仓库内不含任何凭据：

| 变量 | 用途 |
|---|---|
| `CAPTION_STT_URL` / `CAPTION_STT_MODE` | STT 服务地址；`upload` 传字节（远端）/ `path` 传路径（本机） |
| `CAPTION_LLM_URL` / `CAPTION_LLM_KEY` / `CAPTION_LLM_MODEL` | LLM 网关 |
| `CAPTION_ACCESS_PW` | 网页版访问口令，设了才要 |
| `HF_TOKEN` | pyannote 声纹模型（门控） |
| `MEETING_MODEL` | 换 ASR 模型，如 `Qwen/Qwen3-ASR-0.6B` |

`caption` 从 `~/.config/caption.env` 读取（该文件不入库）。

## 依赖

Python 脚本用 [PEP 723 内联依赖](https://peps.python.org/pep-0723/)，`uv run xxx.py` 会自动装。
模型放 `~/models/`：Qwen3.6-35B-A3B-8bit（主力 LLM，带视觉编码器）、Kokoro-82M-bf16（TTS）、IndexTTS-1.5（克隆），Whisper / Qwen3-ASR 走 HF 缓存。

## 测试

```bash
uv run tests/test_caption_core.py      # 纯逻辑
uv run tests/test_stt_lang.py          # 需 stt_server 在跑
uv run tests/test_caption_pipeline.py  # stub 网络，验编排
```
