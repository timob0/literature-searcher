#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/data/chroma.pid"
HOST="${CHROMA_HOST:-127.0.0.1}"
PORT="${CHROMA_PORT:-8000}"
HEALTH_URL="http://${HOST}:${PORT}/api/v2/heartbeat"
CURL_BIN="${CURL_BIN:-/usr/bin/curl}"

if [[ ! -x "${CURL_BIN}" ]]; then
  CURL_BIN="$(command -v curl || true)"
fi
if [[ -z "${CURL_BIN}" ]]; then
  printf 'curl is required for Chroma health checks.\n' >&2
  exit 1
fi

if [[ ! -f "${PID_FILE}" ]]; then
  if "${CURL_BIN}" -fsS --max-time 2 "${HEALTH_URL}" >/dev/null 2>&1; then
    printf 'Chroma is responding, but no managed PID file exists.\n' >&2
    printf 'Stop the process that owns port %s manually if needed.\n' "${PORT}" >&2
    exit 1
  fi
  printf 'Chroma is not running.\n'
  exit 0
fi

pid="$(<"${PID_FILE}")"
if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  printf 'Removed stale Chroma PID file.\n'
  exit 0
fi

kill "${pid}"
for _ in {1..15}; do
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    printf 'Chroma stopped cleanly (PID %s).\n' "${pid}"
    exit 0
  fi
  sleep 1
done

printf 'Chroma did not stop after 15 seconds; sending SIGTERM again.\n' >&2
kill -TERM "${pid}" 2>/dev/null || true
rm -f "${PID_FILE}"
exit 1
