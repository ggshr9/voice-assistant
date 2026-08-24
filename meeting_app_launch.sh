#!/bin/bash
# 由「会议纪要.app」(AppleScript applet)调用。
# 关键：经 launchd 启动菜单栏应用，让 launchd 成为它的「负责进程」，
# 这样 applet 退出后 macOS 不会把它当 applet 的子进程一起回收。
UID_=$(id -u)
SVC="gui/$UID_/com.local.meeting-app"
PLIST="$HOME/Library/LaunchAgents/com.local.meeting-app.plist"

# 已在跑就不重复起(用 launchd 状态判断，避免 pgrep 误匹配到含同名字符串的别的进程)
if launchctl print "$SVC" 2>/dev/null | grep -q "state = running"; then
  exit 0
fi
launchctl bootstrap "gui/$UID_" "$PLIST" 2>/dev/null   # 首次加载(已加载会报错，忽略)
launchctl kickstart "$SVC" 2>/dev/null                 # RunAtLoad=false → 主动拉起
exit 0
