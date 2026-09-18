#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/Users/timo/.pyenv/versions/3.13.7/bin/python" ]]; then
    PYTHON="/Users/timo/.pyenv/versions/3.13.7/bin/python"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]] && "${ROOT_DIR}/.venv/bin/python" -c 'import litsearch' >/dev/null 2>&1; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || true)"
  fi
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  printf 'Python interpreter not found. Set PYTHON to the project interpreter.\n' >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  exec "${PYTHON}" -m litsearch.cli --help
fi

exec "${PYTHON}" -m litsearch.cli "$@"
