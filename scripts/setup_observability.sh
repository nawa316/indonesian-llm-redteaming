#!/usr/bin/env bash
# Start the observability layer for the LLM stack.
#
# Always available (no Docker/root required):
#   * observability/gpu_exporter.py -> Prometheus metrics on :9101
#
# Optional Docker stack (set OBS_DOCKER=1 and have a usable Docker daemon):
#   * Prometheus :9090 + Grafana :3000, provisioned from observability/
#
# Usage: bash scripts/setup_observability.sh [--restart]
LOG_TAG="obs"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

RESTART=0
[[ "${1:-}" == "--restart" ]] && RESTART=1

EXPORTER_LOG="${LOG_DIR}/gpu_exporter.log"
EXPORTER_PID="${RUNTIME_DIR}/gpu_exporter.pid"

install_deps() {
  log "installing Python observability dependencies"
  python3 -m pip install --quiet --no-input -r "${PROJECT_DIR}/requirements.txt"
}

exporter_up() {
  curl -fsS -m 3 "http://127.0.0.1:${EXPORTER_PORT}/metrics" >/dev/null 2>&1
}

start_exporter() {
  if exporter_up && (( ! RESTART )); then
    ok "exporter already serving on :${EXPORTER_PORT}"
    return 0
  fi
  if (( RESTART )) && [[ -f "${EXPORTER_PID}" ]]; then
    kill "$(cat "${EXPORTER_PID}")" 2>/dev/null || true
    sleep 1
  fi
  [[ -f "${PROJECT_DIR}/observability/gpu_exporter.py" ]] || die "gpu_exporter.py missing"

  log "starting GPU/Ollama exporter on :${EXPORTER_PORT}"
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" EXPORTER_PORT="${EXPORTER_PORT}" \
    nohup python3 "${PROJECT_DIR}/observability/gpu_exporter.py" \
    >>"${EXPORTER_LOG}" 2>&1 &
  echo $! > "${EXPORTER_PID}"

  wait_for_http "http://127.0.0.1:${EXPORTER_PORT}/metrics" 30 "exporter"
}

start_docker_stack() {
  docker_usable || die "OBS_DOCKER=1 but docker is not usable"
  log "starting Prometheus + Grafana via docker compose"
  OLLAMA_BASE_URL="${OLLAMA_BASE_URL}" \
  EXPORTER_PORT="${EXPORTER_PORT}" \
  PROMETHEUS_PORT="${PROMETHEUS_PORT}" \
  GRAFANA_PORT="${GRAFANA_PORT}" \
    docker compose -f "${PROJECT_DIR}/observability/docker-compose.yml" up -d
  wait_for_http "http://127.0.0.1:${PROMETHEUS_PORT}/-/ready" 90 "Prometheus"
  ok "Grafana: http://127.0.0.1:${GRAFANA_PORT} (admin / ${GRAFANA_ADMIN_PASSWORD:-admin})"
}

main() {
  install_deps
  start_exporter

  if [[ "${OBS_DOCKER:-0}" == "1" ]]; then
    start_docker_stack
  else
    log "OBS_DOCKER!=1 -> metrics only (exporter is scrape-ready for any Prometheus)"
  fi

  echo
  ok "observability ready"
  echo "  metrics : http://127.0.0.1:${EXPORTER_PORT}/metrics"
  echo "  log     : ${EXPORTER_LOG}"
  echo "  telemetry helper: observability/telemetry.py"
}

main "$@"
