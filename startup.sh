#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
PID_FILE="${DATA_DIR}/chroma.pid"
LOG_FILE="${DATA_DIR}/chroma.log"
HOST="${CHROMA_HOST:-127.0.0.1}"
PORT="${CHROMA_PORT:-8000}"
CHROMA_PATH="${CHROMA_PATH:-${DATA_DIR}/chroma}"
HEALTH_URL="http://${HOST}:${PORT}/api/v2/heartbeat"
CURL_BIN="${CURL_BIN:-/usr/bin/curl}"

if [[ ! -x "${CURL_BIN}" ]]; then
  CURL_BIN="$(command -v curl || true)"
fi
if [[ -z "${CURL_BIN}" ]]; then
  printf 'curl is required for Chroma readiness checks.\n' >&2
  exit 1
fi

if [[ -x "${ROOT_DIR}/.venv/bin/chroma" ]]; then
  CHROMA_BIN="${ROOT_DIR}/.venv/bin/chroma"
elif [[ -x "/Users/timo/.pyenv/versions/3.13.7/bin/chroma" ]]; then
  CHROMA_BIN="/Users/timo/.pyenv/versions/3.13.7/bin/chroma"
elif command -v chroma >/dev/null 2>&1; then
  CHROMA_BIN="$(command -v chroma)"
else
  printf 'Chroma executable not found. Install the project dependencies first.\n' >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${CHROMA_PATH}"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(<"${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    printf 'Chroma is already running (PID %s).\n' "${pid}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if "${CURL_BIN}" -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
  printf 'Chroma is already responding at %s.\n' "${HEALTH_URL}"
  exit 0
fi

nohup "${CHROMA_BIN}" run \
  --path "${CHROMA_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  >"${LOG_FILE}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"

for _ in {1..30}; do
  if "${CURL_BIN}" -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    printf 'Chroma started (PID %s) at http://%s:%s.\n' "${pid}" "${HOST}" "${PORT}"
    printf 'Log: %s\n' "${LOG_FILE}"
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'Chroma exited during startup. See %s.\n' "${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 1
done

printf 'Chroma did not become ready within 30 seconds. See %s.\n' "${LOG_FILE}" >&2
exit 1
