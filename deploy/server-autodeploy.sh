#!/bin/bash
# 服务器自拉取部署:cron 每 5 分钟跑一次。main 有新合并 → 同步文件 → 重启 → 健康检查,
# 失败自动回滚。目的:合并(手机上点也行)之后服务器自己更新,不依赖那台会关机的 Mac。
#
# 文件清单与 Mac 侧 push-web/sync-web 完全一致 —— 两条部署路径必须搬同一批文件,
# 少一个共享模块服务就起不来(stt.py 要 import 仓库根的 llm_chain/noise_filter)。
set -u
SRC="$HOME/voice-assistant-src"          # 仓库克隆(公开仓库,拉取免认证)
SVC="$HOME/voice-svc"
LOG="$SVC/autodeploy.log"
LOCK="$SVC/.autodeploy.lock"
MARK="$SVC/.deployed-commit"

exec 9>"$LOCK"; flock -n 9 || exit 0     # 与手动部署/上一轮自己互斥,拿不到锁就下轮再来

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

cd "$SRC" || { log "仓库目录不在:$SRC"; exit 1; }
git fetch -q origin main 2>>"$LOG" || { log "fetch 失败(网络?)"; exit 0; }
NEW=$(git rev-parse origin/main)
[ "$NEW" = "$(cat "$MARK" 2>/dev/null)" ] && exit 0          # 没新东西,安静退出

log "发现新提交 $NEW,开始部署"
git checkout -q "$NEW" 2>>"$LOG"

# 备份现状(健康检查失败时回滚用)
TS=$(date +%Y%m%d_%H%M%S); BAK="$SVC/_bak_auto_$TS"; mkdir -p "$BAK/web/meeting"
cp -p "$SVC"/web/*.py "$SVC"/web/index.html "$BAK/web/" 2>/dev/null
cp -p "$SVC"/web/meeting/*.py "$BAK/web/meeting/" 2>/dev/null
cp -p "$SVC"/{prompts,llm_chain,noise_filter,recall_core,_voiceprint}.py "$BAK/" 2>/dev/null

# 同步(清单 = push-web ∪ sync-web 的推送集)
cp -p web/*.py web/index.html "$SVC/web/"
cp -p web/meeting/*.py "$SVC/web/meeting/"
cp -p prompts.py llm_chain.py noise_filter.py recall_core.py "$SVC/"
cp -p bin/_voiceprint.py "$SVC/_voiceprint.py"

restart_and_check(){
  local OLD PID C
  OLD=$(pgrep -f "python web_caption.py" | head -1)
  kill -TERM "$OLD" 2>/dev/null
  for i in $(seq 1 20); do
    sleep 3
    PID=$(pgrep -f "python web_caption.py" | head -1)
    [ -n "$PID" ] && [ "$PID" != "$OLD" ] && break
    [ "$i" = 20 ] && return 1
  done
  for i in $(seq 1 20); do
    C=$(curl -sk -o /dev/null -w %{http_code} --max-time 5 https://127.0.0.1/ 2>/dev/null || true)
    [ "$C" = "200" ] && return 0
    sleep 3
  done
  return 1
}

if restart_and_check; then
  echo "$NEW" > "$MARK"
  log "✅ 部署完成 $NEW(新进程 $(pgrep -f 'python web_caption.py' | head -1))"
  ls -dt "$SVC"/_bak_auto_* 2>/dev/null | tail -n +6 | xargs rm -rf 2>/dev/null  # 备份留 5 份
else
  log "❌ 健康检查失败,回滚到部署前"
  cp -p "$BAK"/web/*.py "$BAK"/web/index.html "$SVC/web/" 2>/dev/null
  cp -p "$BAK"/web/meeting/*.py "$SVC/web/meeting/" 2>/dev/null
  cp -p "$BAK"/*.py "$SVC/" 2>/dev/null
  restart_and_check && log "回滚后服务已恢复" || log "‼️ 回滚后仍未恢复,需要人工:journalctl -u caption-web"
fi
