# voice-assistant 维护地图

本地会议语音栈,双平台:Mac CLI(`bin/`,MLX/Apple Silicon 专属)+ Linux 服务器网页版(`web/`,CUDA)。

## 不可违背的约定
- **共享知识只有一份**:候选链语义 `llm_chain.py`、幻觉过滤 `noise_filter.py`、检索内核 `recall_core.py`、
  全部 LLM prompt `prompts.py` —— 都在仓库根,两端共用。改功能先看该不该改在共享层,别在两端各写一遍
  (曾因三份 model_chain 漂移导致翻译死了一周)。
- **写文件必须原子**:用 `bin/_atomicio.atomic_write`(tmp→fsync→replace)。`open(path,"w")` 那一刻文件
  就被截断了,本项目在这上面栽过四次(录音/索引/声纹库/纪要)。`bin/_voiceprint.py` 里的副本是刻意的
  (它被单独推到服务器),改一处看另一处。
- **代码里不留域名/内网地址/密钥**:全部走环境变量(服务器由 secrets.env 注入)。仓库与服务器文件逐字节一致。
- **误删真话的代价高于显示垃圾**:幻觉过滤、任何"自动清理"都往保守调。

## 测试
`tests/test_*.py` 全部是 uv 单文件脚本:`uv run tests/test_xxx.py` 逐个跑。
`tests/test_rec.sh`、`tests/test_push_web.sh` 用 bash。`tests/e2e/web.spec.js` 是 Playwright,
需要私有服务器,公共 CI 跑不了。改动必须配测试;新增守卫建议做变异验证(把守卫破坏掉确认测试会红)。

## 注释风格
中文;写「为什么」不写「做什么」;踩过的坑连同实测数据留在代码旁边。
