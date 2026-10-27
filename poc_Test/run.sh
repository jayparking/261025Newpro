#!/usr/bin/env bash
# Run from any directory; use the GPU environment prepared by 1.setup.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
poc_python="${PYTHON:-$PWD/.venv-gpu/bin/python}"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h || "$arg" == --dry-run ]]; then
    if [[ ! -x "$poc_python" ]] && ! command -v "$poc_python" >/dev/null; then
      poc_python=python3
    fi
    break
  fi
done
[[ -x "$poc_python" ]] || command -v "$poc_python" >/dev/null || {
  echo '먼저 프로젝트의 bash 1.setup.sh를 완료하세요. GPU Python 환경이 없습니다.' >&2
  exit 2
}
exec "$poc_python" poc_Test/run.py "$@"
