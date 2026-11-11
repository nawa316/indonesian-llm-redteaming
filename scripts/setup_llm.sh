#!/usr/bin/env bash
# Install and start the Ollama serving engine.
#
# Runtime selection (LLM_RUNTIME):
#   auto   -> Docker when the daemon is usable, otherwise a native install
#   docker -> force Docker (fails if the daemon/GPU passthrough is unavailable)
#   native -> force the systemd-less native binary
#
# Runpod containers commonly lack CAP_SYS_ADMIN, so Docker-in-Docker is not
# available there; `auto` transparently falls back to native Ollama.
#
# Usage:  bash scripts/setup_llm.sh [--force]
LOG_TAG="llm"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

ollama_api_up() { curl -fsS -m 3 "${OLLAMA_BASE_URL}/api/version" >/dev/null 2>&1; }

start_docker_ollama() {
  docker_usable || die "docker daemon is not usable (LLM_RUNTIME=docker)"
  has_gpu || warn "nvidia-smi unavailable; the container may start without GPU"

  if docker ps -a --format '{{.Names}}' | grep -qx "${OLLAMA_CONTAINER}"; then
    if (( FORCE )); then
      log "removing existing container ${OLLAMA_CONTAINER}"
      docker rm -f "${OLLAMA_CONTAINER}" >/dev/null
    else
      log "starting existing container ${OLLAMA_CONTAINER}"
      docker start "${OLLAMA_CONTAINER}" >/dev/null
    fi
  fi

  if ! docker ps --format '{{.Names}}' | grep -qx "${OLLAMA_CONTAINER}"; then
    log "creating Ollama container from ${OLLAMA_DOCKER_IMAGE}"
    docker run -d \
      --name "${OLLAMA_CONTAINER}" \
      --restart unless-stopped \
      --gpus all \
      -p "${OLLAMA_PORT}:11434" \
      -e OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
      -e OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
      -e OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS}" \
      -v "${RUNTIME_DIR}/ollama:/root/.ollama" \
      "${OLLAMA_DOCKER_IMAGE}" >/dev/null
  fi

  wait_for_http "${OLLAMA_BASE_URL}/api/version" 120 "Ollama (docker)"
  docker exec "${OLLAMA_CONTAINER}" ollama --version 2>/dev/null | tail -1 || true
}

install_native_ollama() {
  if command -v ollama >/dev/null 2>&1 && (( ! FORCE )); then
    log "native ollama already installed: $(ollama --version 2>/dev/null | tail -1 || echo unknown)"
    return 0
  fi
  log "installing Ollama natively via install.sh"
  curl -fsSL https://ollama.com/install.sh -o "${RUNTIME_DIR}/ollama_install.sh"
  sh "${RUNTIME_DIR}/ollama_install.sh" >"${LOG_DIR}/ollama_install.log" 2>&1 \
    || die "Ollama install failed; see ${LOG_DIR}/ollama_install.log"
  command -v ollama >/dev/null 2>&1 || die "ollama binary missing after install"
}

start_native_ollama() {
  if ollama_api_up; then
    ok "Ollama already serving at ${OLLAMA_BASE_URL}"
    return 0
  fi

  # kill a stale server started outside this script (only when forced)
  if (( FORCE )) && pgrep -f "ollama serve" >/dev/null 2>&1; then
    log "stopping stale ollama serve (--force)"
    pkill -f "ollama serve" || true
    sleep 2
  fi

  log "starting native ollama serve on ${OLLAMA_HOST_BIND}:${OLLAMA_PORT}"
  OLLAMA_HOST="${OLLAMA_HOST_BIND}:${OLLAMA_PORT}" \
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
  OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
  OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS}" \
  nohup ollama serve >>"${LOG_DIR}/ollama.log" 2>&1 &
  echo $! > "${RUNTIME_DIR}/ollama.pid"

  wait_for_http "${OLLAMA_BASE_URL}/api/version" 90 "Ollama (native)"
}

main() {
  log "runtime=${LLM_RUNTIME}  workspace=${WORKSPACE}"
  gpu_summary

  case "${LLM_RUNTIME}" in
    docker) start_docker_ollama ;;
    native) install_native_ollama; start_native_ollama ;;
    auto)
      if docker_usable; then
        log "docker daemon detected -> container runtime"
        start_docker_ollama
      else
        log "docker unavailable -> native runtime (expected on unprivileged Runpod pods)"
        install_native_ollama
        start_native_ollama
      fi
      ;;
    *) die "invalid LLM_RUNTIME=${LLM_RUNTIME} (auto|docker|native)" ;;
  esac

  local version
  version="$(http_get "${OLLAMA_BASE_URL}/api/version" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])')"
  ok "Ollama ${version} ready at ${OLLAMA_BASE_URL}"
  http_get "${OLLAMA_BASE_URL}/api/tags" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  models:", [m["name"] for m in d.get("models", [])] or "none")'
}

main "$@"
