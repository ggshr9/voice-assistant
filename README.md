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
  - 持续静音 5 分钟自动停（`REC_SILENCE_SEC=0` 关闭、`REC_SILENCE_DB` 调灵敏度）。会后忘了停录会让 ASR 把静音幻觉成一长串「嗯」
- `meeting <文件> [说话人数|-] [nominutes] [--me 说话人N]` — 转写（mlx-qwen3-asr / Qwen3-ASR-1.7B-8bit）+ pyannote 声纹分人（走 GPU）+ 自动调 `minutes`
  - 内部两个小模块：`bin/_asr_mps.py`（把分人搬到 MPS，快 26.7 倍）、`bin/_annotate.py`（词级 segment → 说话人标注稿）
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
- `config.py` / `jobs.py` / `sessions.py` / `stt.py` — 共享配置、任务队列、会话与录音留存、STT 封装
- `run_web.sh` — 启动脚本（systemd unit 在服务器 `~/voice-svc/systemd/`）
- `web/meeting/` — 会议批处理流水线：`transcribe_step` / `diarize_step` / `asr_diarize_step` / `minutes_lib` / `meeting_pipeline`

> 网页版跑在公司内网那台 GPU 机器上（`ssh gpu-server` → `gpu-box`，2× RTX 4090），代码在那边的 `~/voice-svc/`：faster-whisper + pyannote + CUDA，纪要走 litellm 网关。与本机 MLX 那套是两条独立实现。
>
> **那台机器是 `web/` 的真源，仓库这份是镜像**（与 `bin/` 相反 —— 那边是仓库为真源）。用 `sync-web` 拉回，别手动 rsync：
>
> ```bash
> sync-web --check   # 只看差异
> sync-web           # 拉回
> ```
>
> 脚本把安全边界固化了：排除 `secrets.env`（含 `CAPTION_LLM_KEY` / `CAPTION_ACCESS_PW`）和 TLS 私钥，并**自动把服务器版写死的内网网关默认值洗成空**——线上靠 `secrets.env` 注入，仓库必须保持「默认空 + 缺失即报错」，否则每拉一次就把内网拓扑带回来一次。最后会复查一遍，有残留就退出非零。

## 常驻服务（LaunchAgent）

`launchagents/` 里两个 plist，`cp` 到 `~/Library/LaunchAgents/` 后 `launchctl load` 即可：

- `com.local.meeting-app.plist` — 菜单栏 App（`RunAtLoad=false`，由菜单项按需启停）
- `com.local.caption.plist` — 实时字幕，菜单栏「🌐 实时字幕」开关它

## 配置

所有外部地址与 key 都走环境变量，仓库内**不含任何凭据、也不含任何内网默认地址**——
地址没配就当场报错，不会静默打到错的主机。

| 变量 | 必需 | 用途 |
|---|---|---|
| `CAPTION_STT_URL` | ✅ | STT 服务地址（如本机 `http://127.0.0.1:8082/transcribe`） |
| `CAPTION_STT_MODE` | | `upload` 传字节（远端服务）/ `path` 传路径（本机 mlx 服务），默认 `upload` |
| `CAPTION_LLM_URL` | ✅ | LLM 网关的 `/v1/chat/completions` |
| `CAPTION_LLM_KEY` / `CAPTION_LLM_MODEL` | | 网关鉴权与模型名，默认模型 `Qwen3.6` |
| `CAPTION_ACCESS_PW` | | 网页版访问口令，设了才要 |
| `HF_TOKEN` | | pyannote 声纹模型（门控） |
| `MEETING_MODEL` | | 换 ASR 模型，如 `Qwen/Qwen3-ASR-0.6B` |

`caption` 从 `~/.config/caption.env` 读取（该文件不入库）。

## 依赖

Python 脚本用 [PEP 723 内联依赖](https://peps.python.org/pep-0723/)，`uv run xxx.py` 会自动装。
模型放 `~/models/`：Qwen3.6-35B-A3B-8bit（主力 LLM，带视觉编码器）、Kokoro-82M-bf16（TTS）、IndexTTS-1.5（克隆），Whisper / Qwen3-ASR 走 HF 缓存。

## 测试

```bash
uv run tests/test_minutes.py           # 纪要分块/去噪/文件定位/待办归属（不联网）
uv run tests/test_annotate.py          # 词级 segment → 说话人标注稿的拼接规则
uv run tests/test_asr_mps.py           # 分人走 GPU 的 shim（假模块，不需 torch）
bash   tests/test_rec.sh                # 录音分支（桩 ffmpeg，不需音频设备）
uv run tests/test_caption_core.py      # 字幕纯逻辑
uv run tests/test_caption_pipeline.py  # stub 网络，验编排
uv run tests/test_stt_lang.py          # 需 stt_server 在跑
```

`test_minutes.py` 用变异测试验过牙齿：改坏 `_denoise` 的叠词阈值、拆掉
`split_chunks` 的行边界保护、把「我是谁」指令误塞进逐块笔记，三种注入都会失败。
