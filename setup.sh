#!/usr/bin/env bash
# One entry point for the whole stack.
#
#   ./setup.sh all             # engine + model + observability + RAG + verify
#   ./setup.sh llm             # Ollama only
#   ./setup.sh model           # GGUF download + model registration
#   ./setup.sh observability   # metrics exporter (+ optional Grafana stack)
#   ./setup.sh rag             # seed ChromaDB RAG corpus
#   ./setup.sh app             # start the application server on :8000
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

_run_timed_step() {
  local name="$1"; shift
  local start_ts end_ts start_ns end_ns elapsed_s status
  start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  start_ns="$(python3 -c 'import time; print(time.perf_counter())')"
  if "$@"; then
    status="ok"
  else
    status="failed"
  fi
  end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  end_ns="$(python3 -c 'import time; print(time.perf_counter())')"
  elapsed_s="$(python3 -c "print(round(${end_ns} - ${start_ns}, 3))")"
  STEP_RECORDS+=("${name}|${status}|${start_ts}|${end_ts}|${elapsed_s}")
  if [[ "${status}" != "ok" ]]; then
    return 1
  fi
}

_write_setup_report() {
  local setup_cmd="$1" setup_start_utc="$2" setup_start_perf="$3"
  local setup_end_utc setup_total_s
  setup_end_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  setup_total_s="$(python3 -c "import time; print(round(time.perf_counter() - ${setup_start_perf}, 3))")"
  source scripts/lib/common.sh
  mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}/reports"
  python3 - "${setup_cmd}" "${setup_start_utc}" "${setup_end_utc}" "${setup_total_s}" \
    "${LOG_DIR}/setup_report.json" "${SCRIPT_DIR}/reports/setup_report.json" \
    "${STEP_RECORDS[@]}" <<'PY'
import json, sys
from pathlib import Path

cmd, started, finished, total_s, log_path, report_path, *raw_steps = sys.argv[1:]
steps = []
for item in raw_steps:
    name, status, s_utc, e_utc, elapsed = item.split("|", 4)
    steps.append({
        "name": name,
        "status": status,
        "started_utc": s_utc,
        "finished_utc": e_utc,
        "elapsed_seconds": float(elapsed),
    })

payload = {
    "command": cmd,
    "started_utc": started,
    "finished_utc": finished,
    "total_seconds": float(total_s),
    "steps": steps,
}
try:
    from observability.system_report import collect_system_snapshot
    payload["system_after_setup"] = collect_system_snapshot()
except Exception:
    pass

text = json.dumps(payload, indent=2) + "\n"
for target in (log_path, report_path):
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(text, encoding="utf-8")
print(f"setup report -> {log_path} (total {float(total_s):.2f}s)")
PY
}

case "${cmd}" in
  all)
    STEP_RECORDS=()
    SETUP_START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    SETUP_START_PERF="$(python3 -c 'import time; print(time.perf_counter())')"
    _run_timed_step llm bash scripts/setup_llm.sh
    _run_timed_step model bash scripts/setup_model.sh
    _run_timed_step observability bash scripts/setup_observability.sh
    _run_timed_step rag python3 rag/seed_chroma.py
    _run_timed_step app bash scripts/setup_app.sh
    _run_timed_step verify bash scripts/verify_stack.sh --full
    _write_setup_report "all" "${SETUP_START_UTC}" "${SETUP_START_PERF}"
    ;;
  llm)           bash scripts/setup_llm.sh ;;
  model)         shift; bash scripts/setup_model.sh "$@" ;;
  observability) bash scripts/setup_observability.sh ;;
  rag)           shift; python3 rag/seed_chroma.py --reset "$@" ;;
  app)           bash scripts/setup_app.sh ;;
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
