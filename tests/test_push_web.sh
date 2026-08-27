#!/bin/bash
# push-web 的护栏测试:含脱敏占位值的文件必须被拒推。
# 这条护栏是拿生产事故换来的 —— 直接 scp 仓库版 web_caption.py 覆盖服务器,
# 证书路径变成 caption.example.com,服务进入 3 秒一次的重启循环。
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ✓ $1"; }
bad() { FAIL=$((FAIL+1)); echo "  ✗ $1"; echo "      $2"; }

echo "bin/push-web 护栏测试"

# 假的 remote:任何 ssh/scp 都不该真跑到 —— 检查不过就应该在那之前退出
STUB="$(mktemp -d)"
printf '#!/bin/bash\necho "REACHED_NETWORK" >&2\nexit 0\n' > "$STUB/ssh";  chmod +x "$STUB/ssh"
printf '#!/bin/bash\necho "REACHED_NETWORK" >&2\nexit 0\n' > "$STUB/scp";  chmod +x "$STUB/scp"

# 合成一个含占位值的脏文件 —— 从前直接拿 web_caption.py 当测试对象,
# 配置外移(2026-08-27)后它干净了,这些用例便集体失效:
# 护栏测试不能依赖「仓库文件恰好是脏的」。
mkdir -p "$STUB/repo/web"
cp -r "$REPO/bin" "$STUB/repo/bin"
printf 'CERT = "~/le/live/caption.example.com/fullchain.pem"\n' > "$STUB/repo/web/dirty.py"
cp "$REPO/web/jobs.py" "$STUB/repo/web/jobs.py"
out="$(PATH="$STUB:$PATH" SYNC_WEB_REMOTE=fake-remote bash "$STUB/repo/bin/push-web" dirty.py 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then ok "含占位值的文件被拒推(退出码 $rc)"; else bad "占位值文件竟然推出去了" "$out"; fi
echo "$out" | grep -q "example.com" && ok "报错点名了是哪一行占位值" || bad "没指出具体占位值" "$out"
echo "$out" | grep -q "REACHED_NETWORK" && bad "检查未通过却已经动了网络" "$out" || ok "拒推发生在任何网络动作之前"
echo "$out" | grep -q "打补丁" && ok "给出了正确做法(打补丁而非覆盖)" || bad "没告诉用户该怎么办" "$out"

# 干净文件应通过检查阶段
out2="$(PATH="$STUB:$PATH" SYNC_WEB_REMOTE=fake-remote bash "$STUB/repo/bin/push-web" jobs.py 2>&1)"
echo "$out2" | grep -q "✓ web/jobs.py" && ok "干净文件通过检查" || bad "干净文件被误拦" "$out2"

rm -rf "$STUB"
echo
echo "通过 $PASS,失败 $FAIL"
[ "$FAIL" -eq 0 ]
