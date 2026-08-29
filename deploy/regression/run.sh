#!/bin/bash
# 真实素材回归集:全管线跑一遍真实音频,验「合成测试抓不到」的那类回归。
#
# 为什么存在:auto 语言黑洞(双路径整场空转写)、音乐悬案(背景音乐被硬转成
# "会议"且零提示)、浏览器采集音频 —— 全是合成 say 音频测不出来的。
# 语料放 ~/voice-svc/regression/corpus/,期望写在下面的 CASES 里。
# 用法(服务器上): bash ~/voice-assistant-src/deploy/regression/run.sh
set -u
CORPUS="$HOME/voice-svc/regression/corpus"
WORK="$HOME/voice-svc/regression/work"
WEB="$HOME/voice-svc/web"
PY="$HOME/voice-svc/.venv-meeting/bin/python"
cd "$WEB"; set -a; . secrets.env; . "$HOME/voice-svc/hf.env"; set +a
export HF_ENDPOINT=https://huggingface.co

# 名字|文件|语音占比区间|应否有素材警告|说话人数区间|转写字数下限
CASES="
音乐素材|cursed.wav|0.00-0.10|yes|1-1|100
真实多人会议|cmp.wav|0.30-1.00|no|2-9|300
真人中文单人|zh_solo.wav|0.30-1.00|no|1-1|10
"
pass=0; fail=0
while IFS="|" read -r name file rband warn sband minlen; do
  [ -z "$name" ] && continue
  src="$CORPUS/$file"
  [ -f "$src" ] || { echo "SKIP $name(语料缺失 $file)"; continue; }
  d="$WORK/$name"; rm -rf "$d"; mkdir -p "$d"
  printf '{"id":"%s","title":"回归_%s","scene":"上传","started":0}' "$name" "$name" > "$d/meta.json"
  if ! $PY meeting/meeting_pipeline.py "$src" "$d" - zh > "$d/run.log" 2>&1; then
    echo "FAIL $name: 管线崩了(见 $d/run.log)"; fail=$((fail+1)); continue
  fi
  ratio=$($PY -c "import json;print(json.load(open('$d/diar_stats.json')).get('speech_ratio',-1))" 2>/dev/null || echo -1)
  lo=${rband%-*}; hi=${rband#*-}
  ok=1; why=""
  awk -v r="$ratio" -v lo="$lo" -v hi="$hi" 'BEGIN{exit !(r>=lo && r<=hi)}' || { ok=0; why="占比 $ratio 不在 $rband"; }
  has_warn=$(grep -c "素材可疑" "$d/会议纪要.md" 2>/dev/null || true)
  if [ "$warn" = yes ] && [ "${has_warn:-0}" -eq 0 ]; then ok=0; why="$why;该警告没警告"; fi
  if [ "$warn" = no ] && [ "${has_warn:-0}" -gt 0 ]; then ok=0; why="$why;不该警告却警告了"; fi
  # 说话人计数认【所有轮次标签】,不只匿名牌 —— 声纹认出真名时标签是名字
  # (首跑就这么挂的:样本恰是声纹注册源,被跨平台认出「顾时瑞」,匿名牌计数=0)
  spk=$(grep -oE "^[^：#>]{1,20}：" "$d/会议记录.md" 2>/dev/null | sort -u | wc -l | tr -d " ")
  slo=${sband%-*}; shi=${sband#*-}
  [ "$spk" -ge "$slo" ] 2>/dev/null && [ "$spk" -le "$shi" ] || { ok=0; why="$why;说话人 $spk 不在 $sband"; }
  chars=$(wc -m < "$d/会议记录.md" 2>/dev/null | tr -d " ")
  [ "${chars:-0}" -ge "$minlen" ] || { ok=0; why="$why;转写仅 $chars 字(<$minlen)"; }
  if [ $ok -eq 1 ]; then echo "PASS $name (占比=$ratio 说话人=$spk 字数=$chars)"; pass=$((pass+1))
  else echo "FAIL $name: $why"; fail=$((fail+1)); fi
done <<< "$CASES"
echo "—— 回归集: $pass 过 / $fail 挂 ——"
[ $fail -eq 0 ]
