# 部署

两套完全独立的部署：**本机（Apple Silicon）** 跑 CLI 与菜单栏 App；**GPU 服务器（Linux + CUDA）** 跑网页版。
两边只共享 `prompts.py` 和 `_voiceprint.py`（由 `sync-web` 从仓库推过去），其余各管各的。

---

## 一、本机（macOS / Apple Silicon）

### 前置

```bash
# 包管理与基础工具
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg switchaudio-osx
curl -LsSf https://astral.sh/uv/install.sh | sh      # 所有 Python 脚本用 uv 跑，不用管虚拟环境

# ASR / 分人 / 大脑
uv tool install mlx-qwen3-asr
uv tool install vllm-mlx
```

**BlackHole（录线上会议必需）**

```bash
brew install --cask blackhole-2ch
```

装完要在「音频 MIDI 设置」里手工建两个设备，名字必须一字不差：

| 设备 | 类型 | 组成 | 用途 |
|---|---|---|---|
| `会议录制` | 聚合设备 | BlackHole 2ch + 麦克风 | `rec online` 从这里录（对方声音 + 你自己） |
| `会议外放` | 多输出设备 | BlackHole 2ch + 扬声器/耳机 | 录制时系统输出切到这里，对方声音才进得了 BlackHole，同时你还听得见 |

`rec online` 会自动把系统输出切到「会议外放」，停止时切回（靠 `switchaudio-osx`）。
腾讯会议要确认 设置→音频→扬声器 = 「默认/系统默认」，被钉到具体设备就录不到对方。

**pyannote 声纹分人（门控模型，必须先手工同意条款）**

1. 打开 <https://hf.co/pyannote/speaker-diarization-community-1>，登录后点同意
2. 拿一个有 read 权限的 HF token，写进 shell 配置：`export HF_TOKEN=hf_xxx`
3. 没有 token 或没同意条款时，`meeting` 会**自动降级为纯转写**，不会报错卡住

### 模型

**先看硬件够不够** —— 这是真正的门槛，不是配置问题：

| | 磁盘 | 常驻内存 |
|---|---|---|
| `Qwen3.6-35B-A3B-8bit` 主力大脑 | 35 GB | **36.8 GB**（实测） |
| ASR / 分人 / 对齐器（HF 缓存，自动下载） | ~7 GB | 用时才加载 |
| TTS 与声音克隆（可选） | ~6 GB | 用时才加载 |

**64 GB 内存是下限，32 GB 跑不动 35B。** 内存不够就把 `minutes` / `recall` 指向别的
OpenAI 兼容端点（改 `bin/minutes` 顶部的 `API`），或换个小模型——转写、分人、声纹
这些都跟大脑无关，照样能用。

**国内先配镜像**（HF 直连很慢或不通）：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 写进 ~/.zshrc
```

> ⚠️ 但 **pyannote 门控模型要走官方直连** —— 镜像拿不到门控仓库。
> `bin/meeting` 里已经写死了 `HF_ENDPOINT=https://huggingface.co` 来绕开这点。

**自动下载的（不用管）**：`mlx-community/Qwen3-ASR-1.7B-8bit`（转写）、
`Qwen/Qwen3-ForcedAligner-0.6B`（词级时间戳）、`pyannote/speaker-diarization-community-1`
（分人，需先同意条款）、`mlx-community/whisper-large-v3-turbo`（实时字幕的本机 STT）。

**要手工放进 `~/models/` 的**：

```bash
export HF_ENDPOINT=https://hf-mirror.com
uv tool install huggingface-hub

# 主力大脑：MoE 架构、8bit(group_size 64)、含视觉编码器（名字没 VL 但权重里有 vision_tower）
hf download mlx-community/Qwen3.6-35B-A3B-8bit --local-dir ~/models/Qwen3.6-35B-A3B-8bit

# 以下都是可选，只影响 clone / va / chat 这几个语音功能
hf download mlx-community/Kokoro-82M-bf16 --local-dir ~/models/Kokoro-82M-bf16
hf download IndexTeam/IndexTTS-1.5       --local-dir ~/models/IndexTTS-1.5
```

35 GB 那个建议用 `aria2c` 多线程拉（`hf download` 支持 `HF_HUB_ENABLE_HF_TRANSFER=1`），
不然国内单线程要很久。目录名必须与上面一致——`bin/llm`、`bin/clone` 里是写死的。

### 安装

```bash
git clone <repo> ~/voice-assistant && cd ~/voice-assistant
./install.sh          # 把 bin/* 软链到 ~/.local/bin
```

确认 `~/.local/bin` 在 PATH 里，然后跑自检：

```bash
setup
```

它会**实际去连、去查**每一项（不是让你填表）：工具链在不在、模型齐不齐、HF token 有没有、
pyannote 条款是否已同意（真发一次带 token 的请求）、两个音频设备在不在、大脑起没起。
缺什么直接把命令给你。

配置统一写在 `~/.config/voice-assistant.env`（权限 600）：`setup --token hf_xxx`、
`setup --set K=V`、`setup --show`。已有的 `caption.env` 等仍可覆盖它。

### 常驻服务（可选）

plist 里的 `__HOME__` **是占位符，必须先渲染**（跟服务器 systemd 同理）：

```bash
for f in launchagents/*.plist; do
  sed "s|__HOME__|$HOME|g" "$f" > ~/Library/LaunchAgents/"$(basename "$f")"
done
launchctl load ~/Library/LaunchAgents/com.local.meeting-app.plist
launchctl load ~/Library/LaunchAgents/com.local.caption.plist
```

两个都是 `RunAtLoad=false`，由菜单栏按需启停，不会开机自启。

### 冒烟测试

```bash
llm start                          # 大脑，8bit 权重加载 30-60 秒
rec                                # 说几句，Ctrl+C 或静音 5 分钟自停
meeting ~/会议录音/会议_*.m4a       # 转写 → 分人 → 纪要 → 登记索引
recall "刚才说了什么"
```

---

## 二、GPU 服务器（Linux + CUDA）

网页版：浏览器共享标签页声音 → 实时字幕；上传音频 → 会议纪要。

### 前置

- NVIDIA GPU + 驱动（本部署用 2× RTX 4090，单卡够用）
- Python 3.12、`uv`
- 一个域名 + DNS 托管在 Cloudflare（证书用 DNS-01 签，机器不用暴露 80 端口）

### 两个虚拟环境，别混

它们的依赖互相冲突，所以是两个：

```bash
cd ~/voice-svc

uv venv .venv --python 3.12                 # 实时字幕 + STT 服务
uv pip install --python .venv/bin/python \
    faster-whisper aiohttp webrtcvad numpy flask

uv venv .venv-meeting --python 3.12         # 会议批处理：pyannote + torch + qwen-asr
uv pip install --python .venv-meeting/bin/python \
    pyannote.audio torch torchaudio torchcodec qwen-asr scipy
```

> ⚠️ 两处容易踩：
> - `meeting_pipeline.py` 里**写死了这两个路径**。用 `.venv` 跑分人会 `ModuleNotFoundError: torch`——torch 只在 `.venv-meeting` 里。
> - `flask` 是 `stt_server_cuda.py` 要的，漏了 STT 服务起不来。

**代码放哪也是写死的**：全套代码必须在 `~/voice-svc/`（`web/config.py`、`web/jobs.py`、
`asr_diarize_step.py`、`minutes_lib.py` 里都是这个路径）。换目录要一起改。

### 配置

`~/voice-svc/web/secrets.env`（**不进仓库**，权限 600）：

**LLM 不需要自建网关** —— 任何 OpenAI 兼容端点填上 URL 和 key 就行：官方 OpenAI、
DeepSeek、月之暗面，或本机的 Ollama / vLLM / LM Studio。

```bash
export CAPTION_LLM_URL=https://api.deepseek.com/v1/chat/completions   # 举例
export CAPTION_LLM_KEY=sk-...
export CAPTION_LLM_MODEL=deepseek-chat
export CAPTION_ACCESS_PW=...          # 网页访问口令，留空则不设防
export CAPTION_RETENTION_DAYS=0       # >0 则删除超期录音（文字与纪要保留）
export DIA_CLUSTER_THRESHOLD=0.72     # 声纹聚类阈值，越高越保守、越不容易把一个人拆成多人
```

`~/voice-svc/hf.env`：`export HF_TOKEN=hf_xxx`（同样要先同意 pyannote 条款）

> 本机的 `bin/minutes` 与 `caption_core.py` 会带两个 vLLM 扩展字段
> （`repetition_penalty` / `chat_template_kwargs`，用于关思考和防 8bit 重复退化）。
> 严格端点会对未知参数回 400 —— 代码会**自动脱掉这两个字段重试**，所以填云端 API
> 也能直接用，不必改代码。只有 400/422 才降级；401/500 照常抛出，免得把配置错误
> 伪装成成功。

### 证书

```bash
uv tool install certbot certbot-dns-cloudflare
cat > ~/voice-svc/cf.ini <<'EOF'
dns_cloudflare_api_token = ...
EOF
chmod 600 ~/voice-svc/cf.ini

certbot certonly --dns-cloudflare --dns-cloudflare-credentials ~/voice-svc/cf.ini \
  -d caption.你的域名 \
  --config-dir ~/voice-svc/le --work-dir ~/voice-svc/le-work --logs-dir ~/voice-svc/le-logs
cp ~/voice-svc/le/live/caption.你的域名/{fullchain,privkey}.pem ~/voice-svc/web/
```

### systemd

unit 里的 `__USER__` / `__HOME__` **是占位符，必须先渲染**：

```bash
for f in server/systemd/*.service server/systemd/*.timer; do
  sed -e "s|__USER__|$USER|g" -e "s|__HOME__|$HOME|g" "$f" \
    | sudo tee /etc/systemd/system/"$(basename "$f")" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable --now caption-web.service caption-stt.service caption-renew.timer
```

> 为什么不用 systemd 自己的 `%h` / `%i`：系统级 unit 里 `%h` 解析成 `/root`（不是
> `User=` 的家目录），`%i` 只在模板单元（`name@.service`）里有效。用显式占位更不容易错。

| unit | 作用 |
|---|---|
| `caption-web.service` | 网页版。`Restart=always`；靠 `AmbientCapabilities=CAP_NET_BIND_SERVICE` 以普通用户绑 443 |
| `caption-stt.service` | STT 服务（faster-whisper on CUDA）。默认绑 `127.0.0.1`，要给别的机器用就设 `STT_HOST` |
| `caption-renew.timer` | 每天续证；续完 `pkill web_caption.py`，靠 `Restart=always` 拉起新证书 |

### 证书路径

`web_caption.py` 从 `CAPTION_CERT` / `CAPTION_KEY` 读，默认值指向 `caption.example.com`
这个**占位域名**。用自己的域名就在 `secrets.env` 里加：

```bash
export CAPTION_CERT=$HOME/voice-svc/le/live/caption.你的域名/fullchain.pem
export CAPTION_KEY=$HOME/voice-svc/le/live/caption.你的域名/privkey.pem
```

### 从本机同步代码

```bash
sync-web --check     # 只看差异
sync-web             # 拉回 web/ 与 server/，并推 prompts.py + _voiceprint.py
```

方向不对称，别搞反：

| 方向 | 内容 | 真源 |
|---|---|---|
| 拉 | `web/`、`server/` | **服务器**（在那台机器上迭代） |
| 推 | `prompts.py`、`bin/_voiceprint.py` | **仓库**（代码，跟版本走） |
| 推（需显式开关） | `~/.config/voiceprints.json` | 本机，`SYNC_VOICEPRINTS=1` 才推 |

`sync-web` 排除 `secrets.env` 与 `*.pem`，并把服务器版写死的网关默认地址洗成空——
线上靠 `secrets.env` 注入，仓库必须保持「默认空 + 缺失即报错」。

### 冒烟测试

```bash
cd ~/voice-svc/web/meeting
set -a; . ~/voice-svc/web/secrets.env; . ~/voice-svc/hf.env; set +a
~/voice-svc/.venv-meeting/bin/python asr_diarize_step.py /path/to.wav /tmp/out.txt Chinese 0
```

应看到 `diarized N speakers`；注册过声纹的话还会有 `voiceprint matched: {...}`。

---

## 三、灾难恢复清单

服务器重装时，这些**不在仓库里**、必须另外备份：

- `~/voice-svc/web/secrets.env`、`~/voice-svc/hf.env`、`~/voice-svc/cf.ini`（凭据）
- `~/voice-svc/le/`（证书，也可重新签）
- `~/voice-svc/sessions/`（历史会议录音与纪要）
- `~/voice-svc/voiceprints.json`（声纹，可从本机重推）

本机侧：`~/.config/caption.env`、`~/.config/voiceprints.json`、`~/会议录音/`。
