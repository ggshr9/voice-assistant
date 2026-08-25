# voice-assistant — 全本地语音栈（转写 / 会议纪要 / 实时字幕 / 语音对话 / 声音克隆）

一套跑在 Apple Silicon 上的语音工具集：录音 → ASR → LLM → 结构化纪要 / 声纹识别 / 同传字幕 / 克隆音回答。

**默认全本地**：转写、声纹分人、声纹识别、TTS 全部在本机跑，不联网。纪要与检索走本机
大脑（Qwen3.6-35B-8bit），这一档需要 64GB 内存。

**也可以混合**：内存不够或不想常驻 38GB，把 `CAPTION_LLM_URL` 指向任何 OpenAI 兼容端点
（OpenAI / DeepSeek / Ollama / vLLM）即可 —— **转写、分人、声纹这些仍然全在本地**，
只有生成纪要那一步出去。代码会自动适配严格端点，不用改任何东西。

## 三种用法

| 形态 | 入口 | 说明 |
|---|---|---|
| **CLI** | `bin/` | 主力。`rec` 录音 → `meeting` 转写分人 → `minutes` 出纪要 |
| **菜单栏 App** | `meeting_app.py` | macOS 顶栏 🎙️ 图标，点两下出纪要。只是壳，内部仍调 CLI |
| **MCP** | `bin/meetings_mcp.py` | 让 agent（Claude Code 等）直接查会议档案，4 个工具 |
| **网页版** | `web/` | 浏览器共享标签页声音 → 实时字幕；上传音频 → 会议纪要（服务器 CUDA 版，见下） |

## 快速开始

> 完整的前置条件、服务器部署、灾难恢复清单见 **[docs/DEPLOY.md](docs/DEPLOY.md)**。
> 下面这段假设你已经装好 Homebrew / ffmpeg / uv，建好了聚合设备，并同意了 pyannote 的模型条款。

```bash
brew install ffmpeg switchaudio-osx
brew install --cask blackhole-2ch     # 录线上会议必需
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install mlx-qwen3-asr vllm-mlx

./install.sh          # 把 bin/* 软链到 ~/.local/bin
setup                 # 自检：逐项告诉你缺什么、怎么补
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
- `caption` — 实时字幕（外语→中文），底部双语浮窗。**默认全本地**：本机 mlx-whisper STT(8082) + 本机大脑(8080)，先起这两个服务再跑
  - 远端方案（2026-08-24 实测）：STT `10.0.0.2:8090` 走 Tailscale 直连可用、**不需要 SSH 隧道**；公司 LLM 网关 4000 的 `/v1/chat/completions` 是 404、8088 在 404/429 之间跳，6 月后变过，没深挖
- `dictate` — 全局语音听写，热键切换式，转写后粘贴到光标处
- `chat` / `ask` — 连续对话，语音版 / 打字版双胞胎
- `va` — 全本地语音助手「小麦」
- `setup` — **装完跑一次自检**。逐项实际去连、去查：工具链、模型、HF token、pyannote 条款是否已同意、两个音频设备在不在、大脑起没起。缺什么直接给命令。`setup --token hf_xxx` 写配置
- `who` — **声纹注册表**：`who label <转写目录> 说话人C 李四` 从一场会里认领某人（不需要谁念稿子）、`who enroll <音频> 我 --me`、`who list`、`who forget`
- `recall "问题"` — **用自然语言问过去的会议**。两段式检索：先用标题+摘要选出相关的会，再把那几场全文喂给大脑作答并标出处。`recall --list` 列出全部
- `clone <参考音> <原文> <新内容>` — 声音克隆（IndexTTS-1.5）

### 会议索引

`minutes` 生成纪要后自动登记，也可 `_index.py rebuild` 重建：

```
~/会议录音/索引.json   机器读（recall 与将来的 MCP 直接吃这个）
~/会议录音/索引.md     人读（Finder 里点开就能扫）
```

每场会一行：标题（本地大脑起的）、日期、时长、说话人数、待办数（我的）、一句话摘要、路径。json 是真源，md 由它渲染。

**为什么不上向量库**：会议是几十到几百场量级，全部摘要加起来才几 KB，本来就塞得进上下文；而检索真正需要的是「读懂」不是「找相似」。多一个嵌入模型 + 索引库，换不来准确率，只增加两个会坏的东西。

### 声纹识别

注册后，会议转写里的 `说话人A/B/C` 自动换成真名，`minutes` 也不再需要手动传 `--me`。

**注册不需要任何人念稿子** —— 分人本来就为每个说话人产出一条 256 维向量，给某场会的说话人打个名字就等于注册了，素材还是真实对话（比念稿更贴近实际发音状态）。

```bash
who label 转写_线上会议_20260619_2301 说话人C 李四
who enroll ~/会议录音/降噪_我的声音.wav 张三 --me
```

实测（pyannote community-1）：同一个人两段独立录音 **0.876**、跨独立分人 **0.88~0.94**、陌生人 **≤0.068**。门槛 0.55 且需高出次高者 0.15，**认不出就保留匿名牌，永不瞎猜**。打错了对同一场会重打即可，新向量会追加（既纠正也让它越用越准）。

网页版（服务器）也认：`asr_diarize_step.py` 在分人那一次调用里取向量比对，单声道/上传路径生效（双声道线上会议本来就按声道知道谁是「我」）。

⚠️ 声纹是生物特征，`~/.config/voiceprints.json`（权限 600）**不进 git** —— 代码要可追溯、这类数据要可删除，生命周期相反。

推到服务器要**显式开关**，默认不推：

```bash
SYNC_VOICEPRINTS=1 sync-web     # 声纹会留存在公司那台机器上
```

撤回就删两处：本机 `~/.config/voiceprints.json` 和服务器 `~/voice-svc/voiceprints.json`。

### 共享 prompt

`prompts.py`（仓库根）是纪要 prompt 的**唯一真源**，本机 CLI 与服务器网页版都从它导入。

两边曾各存一份、谁也没同步谁 —— 实测 `FINAL_SYS` 漂移到只剩 66% 相似，而服务器那份是活的（`sessions/` 里最近一场真实会议 2026-07-02），也就是走网页版的会一直用着旧 prompt。`tests/test_prompt_drift.py` 现在会在任何一边重新自存 prompt 时失败。

⚠️ **它的同步方向与 `web/` 相反**：`web/` 的真源在服务器（拉回），`prompts.py` 的真源在仓库（`sync-web` 推过去）。理由是 prompt 属于产品决策，该跟仓库走版本。

### MCP server（给 agent 用）

```bash
claude mcp add -s user meetings -- python3 ~/voice-assistant/bin/meetings_mcp.py
```

stdlib 手写 JSON-RPC 2.0 over stdio，不依赖 mcp SDK，零外部依赖（同 `wxvault_mcp.py` 的路子）。4 个工具：

| 工具 | 用途 | 需要大脑 |
|---|---|---|
| `list_meetings(query?, limit?)` | 列会议，可按标题/摘要过滤 | 否 |
| `get_meeting(id, part?)` | 取纪要 / 逐字记录 / 转写原文；id 支持标题或日期片段模糊定位 | 否 |
| `search_meetings(query, limit?)` | 转写全文字面检索，返回命中上下文 | 否 |
| `ask_meetings(question, top?)` | 自然语言问答，两段式检索后作答并标出处 | ✅ `llm start` |

`search` 和 `ask` 是互补的，别只留一个：前者精确、秒回，适合「谁提过 某支付平台」这种找原话；后者要过大脑、十几秒，适合「当时结论是什么」这种需要读懂再归纳的。

**全部同步秒回** —— 转写/纪要是 `meeting` 和 `minutes` 离线跑完的成品，MCP 只读，所以不需要异步 job + 轮询那套。

### 本机服务（Python）
- `stt_server.py` — 常驻 STT（mlx-whisper large-v3-turbo），`:8082/transcribe`
- `clone_tts_server.py` — 常驻克隆 TTS，`:8083`
- `caption_core.py` / `caption_overlay.py` / `caption.py` — 实时字幕的纯逻辑 / 浮窗 / 编排器
- `assistant.py` / `chat.py` — 语音助手与连续对话实现
- `meeting_app.py` + `meeting_app_launch.sh` — rumps 菜单栏壳

### 网页版（`web/`）
- `web_caption.py` — aiohttp + WSS：浏览器推 16k PCM → VAD 切句 → faster-whisper(CUDA) → 网关翻译 → 推回 `{orig, zh}`
- `config.py` / `jobs.py` / `sessions.py` / `stt.py` — 共享配置、任务队列、会话与录音留存、STT 封装
- `run_web.sh` — 启动脚本；`server/` 是 `~/voice-svc` 根上的服务代码（`stt_server_cuda.py` / `run_stt.sh` / 3 个 systemd unit），也由 `sync-web` 拉回
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
| `CAPTION_LLM_URL` | ✅ | 任何 OpenAI 兼容端点的 `/v1/chat/completions`（OpenAI / DeepSeek / Ollama / vLLM 都行，不必自建网关）|
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
uv run tests/test_index.py             # 会议索引：解析纪要/起标题/去重/渲染
uv run tests/test_recall.py            # 检索：目录渲染/抠编号/关键词兜底/拼上下文
uv run tests/test_mcp.py               # MCP 协议握手/工具分发/错误路径
uv run tests/test_prompt_drift.py      # prompt 只有一份（两边都不许自存）
uv run tests/test_voiceprint.py        # 声纹：余弦/门槛/一对一分配/注册表
uv run --with rumps tests/test_stage_contract.py  # 菜单栏进度与 CLI 的 stage 契约
uv run tests/test_asr_mps.py           # 分人走 GPU 的 shim（假模块，不需 torch）
bash   tests/test_rec.sh                # 录音分支（桩 ffmpeg，不需音频设备）
uv run tests/test_caption_core.py      # 字幕纯逻辑
uv run tests/test_caption_pipeline.py  # stub 网络，验编排
uv run tests/test_stt_lang.py          # 需 stt_server 在跑
```

`test_minutes.py` 用变异测试验过牙齿：改坏 `_denoise` 的叠词阈值、拆掉
`split_chunks` 的行边界保护、把「我是谁」指令误塞进逐块笔记，三种注入都会失败。

## License

MIT
