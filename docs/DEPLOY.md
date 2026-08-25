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

**模型**放 `~/models/`：`Qwen3.6-35B-A3B-8bit`（主力大脑，~38GB，带视觉编码器）、`Kokoro-82M-bf16`（TTS）、`IndexTTS-1.5`（克隆）。
ASR 权重走 HF 缓存，首次运行自动下载。

### 安装

```bash
git clone <repo> ~/voice-assistant && cd ~/voice-assistant
./install.sh          # 把 bin/* 软链到 ~/.local/bin
```

确认 `~/.local/bin` 在 PATH 里。

### 常驻服务（可选）

```bash
cp launchagents/*.plist ~/Library/LaunchAgents/
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

uv venv .venv --python 3.12                 # 实时字幕：faster-whisper 走 ctranslate2
uv pip install --python .venv/bin/python \
    faster-whisper aiohttp webrtcvad numpy

uv venv .venv-meeting --python 3.12         # 会议批处理：pyannote + torch + qwen-asr
uv pip install --python .venv-meeting/bin/python \
    pyannote.audio torch torchaudio torchcodec qwen-asr scipy
```

> ⚠️ `meeting_pipeline.py` 里写死了这两个路径。用 `.venv` 跑分人会 `ModuleNotFoundError: torch`——
> torch 只在 `.venv-meeting` 里。

### 配置

`~/voice-svc/web/secrets.env`（**不进仓库**，权限 600）：

```bash
export CAPTION_LLM_URL=https://你的网关/v1/chat/completions
export CAPTION_LLM_KEY=...
export CAPTION_LLM_MODEL=...
export CAPTION_ACCESS_PW=...          # 网页访问口令，留空则不设防
export CAPTION_RETENTION_DAYS=0       # >0 则删除超期录音（文字与纪要保留）
export DIA_CLUSTER_THRESHOLD=0.72     # 声纹聚类阈值，越高越保守、越不容易把一个人拆成多人
```

`~/voice-svc/hf.env`：`export HF_TOKEN=hf_xxx`（同样要先同意 pyannote 条款）

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

```bash
cp server/systemd/*.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now caption-web.service caption-renew.timer
```

- `caption-web.service` — 网页版，`Restart=always`，用 `AmbientCapabilities=CAP_NET_BIND_SERVICE` 以普通用户绑 443
- `caption-renew.timer` — 每天续证，续完 `pkill web_caption.py` 让 `Restart=always` 拉起新证书

STT 服务（`stt_server_cuda.py`）由 `run_stt.sh` 启动，默认绑内网/Tailscale 地址，用 `STT_HOST` / `STT_PORT` 覆盖。

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
