# 部署

两套完全独立的部署：**本机（macOS + Apple Silicon）** 跑 CLI 与菜单栏 App；
**GPU 服务器（Linux + NVIDIA）** 跑网页版。
两边只共享 `prompts.py` 和 `_voiceprint.py`（由 `sync-web` 从仓库推过去），其余各管各的。

> **只想让一群人（各种系统）能用，就只部署服务器版。** 部署平台是 Linux+NVIDIA，
> 但使用端是浏览器 —— Windows / macOS / Linux / 手机都行，团队里不必人人有 Mac。
> 上传音频出纪要在任何浏览器上都能用；只有「共享标签页声音做实时字幕」需要
> 桌面版 Chrome / Edge（Safari 与移动端拿不到标签页音频）。

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

> **别被「2」误导**：HF 上有 `speaker-diarization-precision-2`，看着像 community-1 的
> 下一版，其实是**另一条产品线**——它仓库里只有 `README.md` 和 `config.yaml`、
> **没有权重**，`pipeline.name` 指向 `pyannoteai.sdk.SDK`，也就是把音频传到
> pyannote 云上算，要 API key。
>
> `speaker-diarization-community-1` 才是当前最新的**本地**版（2025-09-29 更新，
> 下载 525 万 vs precision-2 的 3.7 万），且已从商业线回移了部分改进。
> 换 precision-2 等于放弃「音频不出自己的机器」这个前提。

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

---

## 勘误

**关于「pyannote 过分割」（2026-08-25 更正）**

早期 commit（`5f9f97d` 及其 README 措辞）声称 pyannote 在多人会议上会把同一个人
拆成多簇，并引用了「标称 8 人的会里两两相似度有 4 对在 0.88~0.93」作为证据。

**那个现象是我们自己造成的** —— 那次跑的是 `meeting xxx.wav 8`，强制指定了 8 人。
同一段音频、同一模型、同一台机器的对照：

| | 说话人数 | 最大簇间相似度 | >0.8 的对数 |
|---|---|---|---|
| 不指定人数（自动） | 2 | 0.212 | 0 |
| 强制 8 人 | 8 | 0.927 | 7 |

pyannote 自动判断是准的。**使用建议：不确定几个人就别传那个参数。**
`merge_clusters` 保留为兜底但默认关闭。

---

## 分人器选型：为什么留在 pyannote（2026-08-26 实测）

DiariZen（BUT-FIT，2025）在公开榜单上 DER 明显优于 pyannote 3.1，所以实测了一轮。
**同一台服务器、同一段 265 秒双人音频**：

| | pyannote community-1 | DiariZen (wavlm-large-s80-md) |
|---|---|---|
| 判定说话人数 | **2** | **3**（多出一个只说 1 秒的簇） |
| 主说话人时长 | 141s / 78s | 147s / 76s（两者一致） |
| 分人耗时 | **2.8s** | 5.4s |
| 模型加载 | ~2s | 3.2s |
| 模型体积 | 32M + 25M | 266M |
| 运行环境 | 复用 ASR 的 venv | **独立 venv 5.3G**（torch 2.1.1+cu121） |
| 默认参数 | 开箱可用 | `batch_size=32` 直接 OOM，要手工调到 4 |
| 设备选择 | 可 `.to(device)` | `pipelines/inference.py:57` **写死 `cuda:0`**，只能靠 `CUDA_VISIBLE_DEVICES` 绕开 |
| Apple Silicon | ✅ MPS 已跑通（比 CPU 快 26.7×） | ❌ 未验证，且依赖 CUDA 版 torch |

**结论：不换。** 理由不是 DiariZen 差，是换的代价远大于收益 ——

1. 声纹库要全部重建（embedding 模型不同，历史注册全部作废）
2. Mac 那半边大概率跑不了（CUDA 依赖）→ 两个平台用不同分人器 → 声纹不通用
3. 部署重量级差一个数量级，而本项目最看重「本地、轻」
4. 那个 1 秒的幽灵簇恰好是 `web/` 侧 `_collapse` 安全网在处理的东西 —— 换过去同样要打补丁

**这个结论的适用范围有限（n=1，且是双人音频）**：DiariZen 的宣称优势区间是 5 人以上，
这次没有测到它擅长的场景。准确说法是「在少数人、追求轻量的场景 pyannote 更优」，
**不是**「DiariZen 不如 pyannote」。

环境保留在服务器 `~/voice-svc/DiariZen/`，将来真遇到多人场景可直接复测。
复现脚本见该目录，跑的时候记得 `CUDA_VISIBLE_DEVICES` 避开在用的卡。

---

## 部署 web/ 的正确姿势（2026-08-26 事故记录）

**仓库里的 `web/` 是脱敏镜像，不能直接 scp 覆盖服务器。**

为公开而做的脱敏把真实值换成了占位符：

| 文件 | 仓库（镜像） | 服务器（生产） |
|---|---|---|
| `web_caption.py` | `caption.example.com/fullchain.pem` | 真实域名证书路径 |
| `config.py` | `CAPTION_LLM_URL` 默认空 | 内网网关地址 |

实测事故：直接 `scp web/web_caption.py` 覆盖后，证书路径指向 `example.com`，
`ctx.load_cert_chain` 抛 `FileNotFoundError`，服务进入 **3 秒一次的重启循环**
（restart counter 从 20 涨到 34），直到从备份把真实路径改回来。

**正确做法**：用 `bin/push-web`，它会

1. 先在服务器上把待覆盖的文件备份到 `~/voice-svc/_bak_<时间戳>/`
2. 逐文件扫描脱敏占位值（`example.com` / `__HOME__` / `YOUR_` 等），**命中就拒推**
3. 全部干净才推送，然后重启并轮询到服务真的返回 200

含占位值的文件必须**在服务器上打补丁**（`python3 -` + 精确字符串替换），
不要整文件覆盖。护栏本身有测试：`tests/test_push_web.sh`。

## 网页 UI 的自动化测试

`tests/e2e/web.spec.js`（Playwright，13 例）覆盖口令门、会议库、页面健康度、
以及**完整的上传 → 纪要**链路。需要两个环境变量，都不写进仓库：

```bash
ssh -N -L 127.0.0.1:21443:127.0.0.1:443 <你的服务器> &     # 隧道，避免依赖公网 DNS
CAPTION_URL=https://127.0.0.1:21443 \
CAPTION_PW=<口令> \
CAPTION_FIXTURE=/path/to/一段短音频.m4a \
  npx playwright test tests/e2e/web.spec.js
```

⚠️ 本机若已有服务占着目标端口，`ssh -L` 可能只绑到 IPv6 `[::1]`，
而 curl/浏览器走 IPv4 会打到那个无关服务上（踩过，表现为莫名其妙的 426）。
所以显式写 `-L 127.0.0.1:<port>:...`，并先 `lsof -nP -iTCP:<port>` 确认端口空闲。

---

## 容器化部署（2026-08-27,issue #2）

```bash
cd deploy/docker && cp .env.example .env    # 填口令与 LLM 网关(任何 OpenAI 兼容端点)
docker compose --profile cpu up -d          # 无卡机器:whisper int8,慢但全功能
docker compose --profile gpu up -d          # NVIDIA:Qwen3-ASR(需 nvidia-container-toolkit)
```

打开 `https://<主机>:8443`。容器首启自签 TLS 证书（`CAPTION_TLS_SELFSIGN=1`）——
浏览器警告点「继续」；**必须是 https**，`getUserMedia` 在 http 下连麦克风都拿不到。
会话与模型缓存都在 `va-data` volume 里，容器可随意重建。

Windows 用户：装 Docker Desktop 后双击 `windows/start.cmd` —— 首次生成 `.env`
让你填 API 配置，再次运行即构建+启动+开浏览器。
（Mac CLI 那套是 MLX/Apple Silicon 专属，Windows 走网页版。）

三个为容器做的代码改动（对裸机零影响）：
- `asr_backends`:whisper 的 compute_type 不再写死 float16（CPU 要 int8，写死当场报错）
- `meeting_pipeline`:两个解释器路径可用 `MEETING_PY_STT/PY_DIA` 顶掉（容器里全栈同一解释器）
- `web_caption`:显式 `CAPTION_TLS_SELFSIGN=1` 才自签，裸机缺证书仍大声报错

---

## 实时 ASR 选型复核（2026-08-27,用户真实录音 A/B）

用户反馈实时字幕效果不佳(多人口语闲聊场景)。用其 123 秒真实录音做了两组实验:

**① FireRedASR2-AED(榜单中文 CER 2.89 vs Qwen 3.76)** —— 在我们的
「任意 VAD 段」输入下**完败**:六个 10 秒切片里两个空、两个含义错乱
("花钱是贵"被转成"两千四百四十五个男司机"),个别切片耗时 4 秒。
榜单成绩出自它自家全套管线(专用 VAD+LID+标点);脱离那套喂裸段直接崩。
LLM 版(8B)无流式,不适合实时;留作将来批处理侧评估。
环境保留在服务器 `~/voice-svc/FireRedASR2S/`(py3.11 venv + AED 4.5G)。

**② Qwen 的 context 参数(滚动上文)** —— 模型把上文**原样复读进输出**
(经典 prompt 回声),越滚越长。弃用。

**结论:实时侧 Qwen3-ASR-1.7B 维持现役。** 体感差的真实构成:
实时=无上下文短段草稿,本就是质量阶梯的下层;多人口语闲聊是 ASR 最难场景。
真正的质量在「停止 → 生成纪要」后的批处理重转(整段+分人),评估应以那份为准。

**② 附:pyannote 对某路浏览器采集音频输出精确全零(2026-08-27,悬案)**
某场网页版实录(健康电平 -26dB、webrtcvad 正常触发、Qwen 每片成句),
pyannote segmentation-3.0/community-1 对其输出**精确 0.000** —— 切片、响度归一、
高通 80/200Hz、内存波形直喂全部无效;同环境对上传文件一直正常。原因未明。
已在 asr_diarize_step 加降级:分人空手而归时用 webrtcvad 切段、全部记「说话人A」,
保住转写(此前是整场空纪要且查无此错)。悬案线索:该音频经 Chrome
echoCancellation+noiseSuppression+AGC 处理,0-1k 能量占比 67.7%(对照 51.9%)。

**③ VibeVoice-ASR-7B 评测(2026-08-28/29,微软,转写+分人+时间戳一体)**
- 真人语音(英文 2 人对话 60s):**质量优秀** —— 说话人全对、轮次边界干净、
  逐字级忠实(连 "uh, you know" 都保留),结构化 JSON 一趟出「谁-何时-说什么」。
- 音乐段:输出 `[Lyric]` 标签 —— 三家模型里唯一明说「这是歌」的(Qwen 硬转、
  FireRed 编歌词、pyannote 沉默),可当音乐检测器,但素材质检用 pyannote
  语音帧占比更便宜(实测:真人语音 47.9% vs 音乐 0.0%)。
- **速度勘误**:首测"60s 跑 682s"是音乐段幻觉循环烧满 max_new_tokens 的假象;
  真人语音实测 **7.68s/60s(RTF≈0.13)**,且当时实时 Qwen 同卡驻留未受影响。
- **真实瓶颈只剩长音频显存**:60s 段与 Qwen 共存没问题,185s 整段的激活把
  24GB 撑爆(OOM)。可选出路:8bit 量化(~10GB)、60s 分窗(损失长上下文优势)、
  批处理期间临时卸载实时模型。—— 因此它是【批处理后端的真候选】
  (一个模型替掉 pyannote+Qwen 接力,还自带音乐标注),不是"留档不采用"。
  环境保留:~/voice-svc/VibeVoice(py3.12 venv + 模型 ~15G)。

**④ VibeVoice-ASR 8bit 量化 spike(2026-08-29,结论:现有硬件可行)**
- 整模型 8bit → 全部 `[Unintelligible]`:**量化毁的是音频塔**(bf16 对照完好锁定归因)。
- 解法:`llm_int8_skip_modules=[acoustic/semantic_tokenizer, connectors, diffusion_head,
  lm_head]` —— 只量化 LLM 解码器。输出与 bf16 逐字级一致(仅 "C-bus/C-balls" 级微差)。
- 整段 265s 真实会议:**峰值 13.65GB、135.5s(RTF≈0.51)、与实时 Qwen 同卡共存无碍**
  (bf16 在 185s 就 OOM 的场景)。判 3 个说话人(pyannote 判 2,素材实为 8 人会切片,
  孰对未裁)。
- **判词:批处理后端的可行候选,不再受硬件阻塞。** 换它前必须解决:声纹注册与
  pyannote 向量的兼容(方案候选:VV 出轮次 + pyannote 只做声纹比对)、RTF 0.5 对
  2h 长会的耗时(需分窗)。两条 prompt 级坑:processor 必须传
  language_model_pretrained_name="Qwen/Qwen2.5-7B",否则特征全错静默输出
  [Unintelligible];在退化输入(音乐幻觉循环)上量的性能数字不能当模型性能。

## 导出(2026-08-29)

- **Word(.docx)**:详情页每张文档卡的「Word」按钮,服务端 pandoc 转换
  (静态二进制 `~/voice-svc/bin/pandoc`,免 root;容器里走 apt 的 pandoc,
  `CAPTION_PANDOC` 可指路径)。
- **PDF**:「打印/PDF」按钮开打印视图,用浏览器原生「另存为 PDF」——
  刻意不在服务端做:那要拖一整套 LaTeX,而浏览器路径零依赖、离线可用。
