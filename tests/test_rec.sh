#!/bin/bash
# bin/rec 的分支测试:用桩 ffmpeg 顶掉真录音,只验「传了什么参数、走了哪条路」。
# 不需要音频设备,可在任何机器上跑。  用法: bash tests/test_rec.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "  ✓ $1"; }
bad()  { FAIL=$((FAIL+1)); echo "  ✗ $1"; echo "      $2"; }

STUB="$(mktemp -d)"
ARGLOG="$STUB/args.txt"

# 桩 ffmpeg:列设备时假装有麦克风;录音时把参数记下来就立刻退出
cat > "$STUB/ffmpeg" <<STUBEOF
#!/bin/bash
if [[ "\$*" == *"-list_devices"* ]]; then
  echo "[AVFoundation indev @ 0x1] AVFoundation audio devices:" >&2
  echo "[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone" >&2
  echo "[AVFoundation indev @ 0x1] [1] 会议录制" >&2
  exit 0
fi
printf '%s\n' "\$*" >> "$ARGLOG"
exit 0
STUBEOF
chmod +x "$STUB/ffmpeg"
# ffprobe/meeting 也顶掉,免得测试触发真转写
printf '#!/bin/bash\nexit 0\n' > "$STUB/meeting"; chmod +x "$STUB/meeting"

run_rec() {  # run_rec <REC_SILENCE_SEC> [rec 的参数...]
  : > "$ARGLOG"
  local sec="$1"; shift
  HOME="$STUB" FFMPEG="$STUB/ffmpeg" REC_SILENCE_SEC="$sec" \
    bash "$REPO/bin/rec" "$@" >"$STUB/out.txt" 2>&1
}

echo "bin/rec 分支测试"

run_rec 0
if grep -q "silencedetect" "$ARGLOG"; then
  bad "REC_SILENCE_SEC=0 应完全不加检测滤镜" "$(cat "$ARGLOG")"
else
  ok "REC_SILENCE_SEC=0 走原始路径,不加 silencedetect"
fi
grep -q "静音自动停：已关闭" "$STUB/out.txt" \
  && ok "关闭时有明确提示" \
  || bad "关闭时应提示已关闭" "$(cat "$STUB/out.txt")"

run_rec 120
if grep -q "silencedetect=noise=-35dB:d=120" "$ARGLOG"; then
  ok "阈值透传到 silencedetect(d=120, 默认 -35dB)"
else
  bad "silencedetect 参数不对" "$(cat "$ARGLOG")"
fi

: > "$ARGLOG"
HOME="$STUB" FFMPEG="$STUB/ffmpeg" REC_SILENCE_SEC=90 REC_SILENCE_DB=-45 \
  bash "$REPO/bin/rec" >/dev/null 2>&1
grep -q "silencedetect=noise=-45dB:d=90" "$ARGLOG" \
  && ok "REC_SILENCE_DB 可覆盖灵敏度" \
  || bad "REC_SILENCE_DB 未生效" "$(cat "$ARGLOG")"

run_rec 60 online
if grep -q -- "-i :会议录制" "$ARGLOG"; then
  ok "online 模式仍选聚合设备「会议录制」"
else
  bad "online 模式设备选错" "$(cat "$ARGLOG")"
fi
grep -q -- "-ar 16000" "$ARGLOG" && grep -q -- "-ac 1" "$ARGLOG" \
  && ok "采样率/声道未被改动(16k 单声道)" \
  || bad "音频参数被改动" "$(cat "$ARGLOG")"

# ---- 静音检测必须先下混单声道 ----
# 「会议录制」聚合设备给的是 2.1,实测第 3 路(LFE)常年 RMS -39dB。silencedetect
# 要求【所有声道】都低于阈值,于是自动停从来没触发过(设 8 秒、跑满 7 分钟不停)。
# 命令行上的 `-ac 1` 是【输出】选项,在滤镜链之后才生效,救不了这个。
run_rec 60 online
if grep -qE "aformat=channel_layouts=mono,silencedetect" "$ARGLOG"; then
  ok "silencedetect 之前先 aformat 成单声道"
else
  bad "滤镜链里缺少下混,多声道设备上静音自动停会永不触发" "$(cat "$ARGLOG")"
fi
# 顺序不能反:先 silencedetect 再 aformat 等于没修
if grep -qE "silencedetect[^\"]*,aformat" "$ARGLOG"; then
  bad "aformat 排在 silencedetect 之后,等于没修" "$(cat "$ARGLOG")"
else
  ok "滤镜顺序正确(aformat 在前)"
fi

# ---- 停止时不能连同转写器一起杀 ----
# 转写器一死就没人读 FIFO,ffmpeg 立刻阻塞在写管道上,处理不了信号,
# moov 永远写不出来 —— 整个 m4a 报废(真实踩过:786KB 音频在、索引没有、救不回来)。
if grep -qE "trap '[^']*LIVEPID" "$REPO/bin/rec"; then
  bad "trap 里同时杀了 LIVEPID,会让 ffmpeg 阻塞在 FIFO 上封不上文件" \
      "$(grep -n "trap '" "$REPO/bin/rec")"
else
  ok "trap 只转发信号给 ffmpeg,不提前杀转写器"
fi

# ---- 落地校验:读不出的文件不能打绿勾 ----
if grep -q "录音文件读不出来" "$REPO/bin/rec"; then
  ok "结束后校验产物,损坏时明确报错"
else
  bad "结束后没有校验,损坏文件也会显示「已保存」" ""
fi

rm -rf "$STUB"
echo
echo "通过 $PASS,失败 $FAIL"
[ "$FAIL" -eq 0 ]
