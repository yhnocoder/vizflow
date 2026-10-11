#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv run scripts/check_comment.py
uv run scripts/check_map.py
if command -v shellcheck >/dev/null 2>&1; then
  git ls-files -z '*.sh' | xargs -0 -r shellcheck
else
  echo "shellcheck not found, skipped"
fi
npx tsc --noEmit
npx vitest run
