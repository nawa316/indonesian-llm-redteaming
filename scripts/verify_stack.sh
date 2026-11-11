#!/usr/bin/env bash
# End-to-end readiness check for the LLM engine + observability stack.
# Exits non-zero on the first failed check, so it is safe for CI/smoke gates.
#
# Usage: bash scripts/verify_stack.sh [--full]
LOG_TAG="verify"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1
FAILURES=0

check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "${label}"; else warn "${label}"; FAILURES=$(( FAILURES + 1 )); fi
}

section() { echo; echo "== $* =="; }

section "hardware"
gpu_summary
echo "  CPU threads: $(nproc)"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Disk /workspace: $(df -h /workspace 2>/dev/null | awk 'NR==2 {print $4" free"}')"

section "services"
check "Ollama API" curl -fsS -m 5 "${OLLAMA_BASE_URL}/api/version"
check "Exporter metrics" curl -fsS -m 5 "http://127.0.0.1:${EXPORTER_PORT}/metrics"

section "model"
if tags_json="$(curl -fsS -m 5 "${OLLAMA_BASE_URL}/api/tags")"; then
  if ! python3 -c '
import json, sys
want, raw = sys.argv[1], sys.argv[2]
models = json.loads(raw).get("models", [])
names = [m.get("name", "") for m in models]
present = any(n == want or n.startswith(want + ":") for n in names)
print("  registered:", names or "none")
print("  target", repr(want), ":", "FOUND" if present else "MISSING")
sys.exit(0 if present else 1)
' "${LLM_MODEL_NAME}" "${tags_json}"
  then
    FAILURES=$(( FAILURES + 1 ))
  fi
else
  warn "cannot list models: Ollama API unreachable"
  FAILURES=$(( FAILURES + 1 ))
fi

if (( FULL )); then
  section "inference smoke test"
  body="$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Balas dengan satu kata: aman"}],"stream":False,"options":{"num_predict":16}}))' "${LLM_MODEL_NAME}")"
  started="$(date +%s%3N)"
  if resp="$(curl -fsS -m 600 -H 'Content-Type: application/json' -d "${body}" "${OLLAMA_BASE_URL}/api/chat")"; then
    ended="$(date +%s%3N)"
    python3 - "${resp}" "$(( ended - started ))" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
print(f"  latency_ms={sys.argv[2]} tokens={d.get('eval_count')}")
print(f"  output: {(d.get('message') or {}).get('content','').strip()[:200]}")
PY
  else
    warn "chat completion failed"; FAILURES=$(( FAILURES + 1 ))
  fi
fi

section "result"
if (( FAILURES > 0 )); then
  die "${FAILURES} check(s) failed"
fi
ok "all checks passed"
