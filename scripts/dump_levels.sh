#!/bin/sh
set -eu
if [ "$#" -lt 2 ]; then
  echo "usage: sh scripts/dump_levels.sh <out-dir> <page.yaml under examples/>..." >&2
  exit 1
fi
out=$1
shift
pages=$(printf '%s,' "$@")
cd "$(dirname "$0")/.."
VIZFLOW_DUMP_OUT="$out" VIZFLOW_DUMP_PAGES="${pages%,}" npx vitest run test/scene3d/dump_levels.test.ts
