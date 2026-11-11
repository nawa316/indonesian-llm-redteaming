#!/usr/bin/env bash
# One entry point for the whole stack.
#
#   ./setup.sh all             # engine + model + observability + RAG + verify
#   ./setup.sh llm             # Ollama only
#   ./setup.sh model           # GGUF download + model registration
#   ./setup.sh observability   # metrics exporter (+ optional Grafana stack)
#   ./setup.sh rag             # seed ChromaDB RAG corpus
#   ./setup.sh verify          # readiness checks
#   ./setup.sh status          # human-readable current state
#
# Everything is idempotent: safe to re-run after a pod restart.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

cmd="${1:-all}"

case "${cmd}" in
  all)
    bash scripts/setup_llm.sh
    bash scripts/setup_model.sh
    bash scripts/setup_observability.sh
    python3 rag/seed_chroma.py
    bash scripts/verify_stack.sh --full
    ;;
  llm)           bash scripts/setup_llm.sh ;;
  model)         shift; bash scripts/setup_model.sh "$@" ;;
  observability) bash scripts/setup_observability.sh ;;
  rag)           shift; python3 rag/seed_chroma.py --reset "$@" ;;
  verify)        shift; bash scripts/verify_stack.sh "$@" ;;
  status)
    source scripts/lib/common.sh
    echo "Ollama : $(curl -fsS -m 3 "${OLLAMA_BASE_URL}/api/version" 2>/dev/null || echo DOWN)"
    echo "Models :"
    curl -fsS -m 3 "${OLLAMA_BASE_URL}/api/tags" 2>/dev/null \
      | python3 -c 'import json,sys; [print(f"  - {m[\"name\"]} ({round(m[\"size\"]/1e9,1)} GB)") for m in json.load(sys.stdin).get("models",[])]' \
      || echo "  none"
    echo "Stats  :"
    curl -fsS -m 3 "http://127.0.0.1:${EXPORTER_PORT}/metrics" 2>/dev/null \
      | grep -E '^llm_(gpu_utilization_percent|gpu_memory_used_bytes|ollama_up)' \
      || echo "  exporter DOWN"
    ;;
  -h|--help|help) usage 0 ;;
  *) usage 1 ;;
esac
