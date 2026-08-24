#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

pkg update -y
pkg install -y python git

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
chmod +x "$PROJECT_DIR/bin/nabd" "$PROJECT_DIR/bin/nabd-verify"
ln -sf "$PROJECT_DIR/bin/nabd" "$PREFIX/bin/nabd"
ln -sf "$PROJECT_DIR/bin/nabd-verify" "$PREFIX/bin/nabd-verify"

printf '\nتم تثبيت Nabd بنجاح.\n'
printf 'ضع مفتاحك في البيئة ثم نفّذ: nabd "وصف المهمة"\n'
