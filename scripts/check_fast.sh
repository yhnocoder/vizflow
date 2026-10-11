#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
npx tsgo --noEmit
npx vitest run --changed "${1:-HEAD}" --passWithNoTests
