#!/usr/bin/env bash
# Start the Controlled Testbed application server (FastAPI/Uvicorn) on :8000.
#
# Hosts guardrails + ChromaDB RAG + the local Ollama engine. The Test Client
# Environment reaches it over HTTP, so it never needs GPU/Ollama/Chroma access.
#
# Usage: bash scripts/setup_app.sh [--restart]
LOG_TAG="app"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

RESTART=0
[[ "${1:-}" == "--restart" ]] && RESTART=1

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_LOG="${LOG_DIR}/app_server.log"
APP_PID="${RUNTIME_DIR}/app_server.pid"
HEALTH_URL="http://127.0.0.1:${APP_PORT}/health"

install_deps() {
  log "installing application-server dependencies"
  python3 -m pip install --quiet --no-input "fastapi>=0.110" "uvicorn>=0.29"
}

app_up() {
  curl -fsS -m 3 "${HEALTH_URL}" >/dev/null 2>&1
}

main() {
  install_deps

  if app_up && (( ! RESTART )); then
    ok "app server already healthy at ${HEALTH_URL}"
    curl -fsS "${HEALTH_URL}" | python3 -m json.tool
    return 0
  fi

  if [[ -f "${APP_PID}" ]]; then
    kill "$(cat "${APP_PID}")" 2>/dev/null || true
    sleep 1
  fi

  log "starting app server on ${APP_HOST}:${APP_PORT}"
  (
    cd "${PROJECT_DIR}"
    APP_HOST="${APP_HOST}" \
    APP_PORT="${APP_PORT}" \
    OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
    LLM_MODEL_NAME="${LLM_MODEL_NAME}" \
    RAG_TOP_K="${RAG_TOP_K}" \
    CHROMA_DIR="${CHROMA_DIR}" \
    nohup python3 app_server.py >>"${APP_LOG}" 2>&1 &
    echo $! > "${APP_PID}"
  )

  wait_for_http "${HEALTH_URL}" 90 "app server"
  ok "app server ready"
  curl -fsS "${HEALTH_URL}" | python3 -m json.tool
  echo "  log : ${APP_LOG}"
  echo "  pid : $(cat "${APP_PID}")"
}

main "$@"
