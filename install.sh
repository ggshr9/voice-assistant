#!/bin/bash
# 把 bin/* 软链到 ~/.local/bin，让本仓库成为 CLI 的唯一真源。
# 已存在的同名文件会先备份到 ~/.local/bin/<名>.pre-repo。
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HOME/.local/bin"
mkdir -p "$DEST"

for src in "$REPO"/bin/*; do
  name=$(basename "$src")
  target="$DEST/$name"
  if [ -L "$target" ]; then
    rm "$target"
  elif [ -e "$target" ]; then
    mv "$target" "$target.pre-repo"
    echo "  已备份原文件 → $name.pre-repo"
  fi
  ln -s "$src" "$target"
  echo "  ✓ $name"
done

echo
echo "完成。确认 PATH 里有 $DEST："
echo '  export PATH="$HOME/.local/bin:$PATH"'
