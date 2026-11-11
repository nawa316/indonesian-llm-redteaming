# Indonesian LLM Red-Teaming — Llama-Sahabat-AI v2 70B

Security audit and mitigation-effectiveness evaluation of
**`Llama-Sahabat-AI-v2-70B-IT`** (Q3_K_S) against the **OWASP Top 10 for LLM
Applications 2026**.

This repository currently contains the **infrastructure layer**: reproducible
bootstrap for the serving engine and the observability stack, designed for
Runpod A100 pods (no systemd, no privileged Docker) as well as ordinary Linux
hosts with Docker.

---

## 1. Architecture

```
┌──────────────────────────── pod / host ────────────────────────────┐
│                                                                    │
│  test runner / guardrails            (next milestone)              │
│        │  HTTP /api/chat                                           │
│        ▼                                                           │
│  ┌──────────────┐   nvidia-smi    ┌──────────────────────────┐     │
│  │ Ollama :11434│◄────────────────│ gpu_exporter.py  :9101   │     │
│  │ Q3_K_S 70B   │   /api/ps       │ GPU + Ollama + host      │     │
│  └──────────────┘                 └───────────┬──────────────┘     │
│                                               │ scrape             │
│                                   ┌───────────▼──────────────┐     │
│                                   │ Prometheus :9090         │     │
│                                   │ Grafana    :3000 (opt.)  │     │
│                                   └──────────────────────────┘     │
│                                                                    │
│  telemetry.py  →  JSONL + CSV evidence (per test run)              │
└────────────────────────────────────────────────────────────────────┘
```

Runtime choice is automatic:

| Environment | Engine | Observability |
|---|---|---|
| Runpod pod (unprivileged, no Docker) | native Ollama, `nohup` | Python exporter only |
| Linux host with Docker + GPU | `ollama/ollama` container | exporter + Prometheus + Grafana |

---

## 2. Quick start

```bash
# on the GPU host, inside the repo
cp .env.example .env          # optional overrides
./setup.sh all
```

Individual steps:

```bash
./setup.sh llm             # install + start Ollama
./setup.sh model           # download GGUF + register model
./setup.sh observability   # metrics exporter (+ optional Grafana)
./setup.sh verify --full   # readiness + inference smoke test
./setup.sh status          # what is running right now
```

Everything is idempotent and safe to re-run after a pod restart.

### Runpod specifics

The stock Runpod container has **no systemd** and **no `CAP_SYS_ADMIN`**, so
Docker-in-Docker is unavailable. `setup_llm.sh` detects this and installs Ollama
natively, which serves the same REST API on `0.0.0.0:11434`.

```bash
# after a pod restart
cd /workspace/indonesian-llm-redteaming && ./setup.sh llm
./setup.sh observability
```

Model artifact: [`blackshell69/Llama-Sahabat-AI-v2-70B-IT-Q3_K_S`](https://huggingface.co/blackshell69/Llama-Sahabat-AI-v2-70B-IT-Q3_K_S)
— 30.9 GB, 70.55B params, llama arch, 131k context. SHA-256 is pinned in
`scripts/lib/common.sh`; verify with `./setup.sh model --verify-sha`.

---

## 3. Observability

### Metrics (`:9101/metrics`)

| Metric | Meaning |
|---|---|
| `llm_gpu_utilization_percent` | GPU utilisation per device |
| `llm_gpu_memory_used_bytes` / `_total_bytes` | VRAM pressure |
| `llm_gpu_temperature_celsius`, `llm_gpu_power_watts` | thermals / power |
| `llm_ollama_up` | API reachability |
| `llm_ollama_loaded_models` | resident models |
| `llm_ollama_model_vram_bytes` / `_size_bytes` | per-model footprint |
| `llm_host_cpu_percent`, `llm_host_memory_*` | host pressure |

Quick look without Prometheus:

```bash
curl -s localhost:9101/metrics | grep '^llm_'
```

### Grafana stack (optional, needs Docker)

```bash
OBS_DOCKER=1 ./setup.sh observability
# Grafana http://127.0.0.1:3000  (admin/admin)
# Prometheus http://127.0.0.1:9090
```

The dashboard `Llama-Sahabat Redteam · LLM Engine` is provisioned automatically.

### Per-request telemetry (JSONL → CSV)

```python
from observability.telemetry import TelemetryRecorder

with TelemetryRecorder("reports/baseline.jsonl") as rec:
    with rec.timed(test_id="LLM01-001", category="LLM01",
                   technique="single_turn", prompt=payload, model="sahabat-70b-q3ks"):
        r = client.post("/api/chat", json=body)
        rec_event = {"status_code": r.status_code,
                     "response": r.json()["message"]["content"],
                     "completion_tokens": r.json().get("eval_count")}
    rec.export_csv("reports/baseline.csv")
```

Each line records timestamp, run id, test id, phase, OWASP category, technique,
turn, data type, guardrail decision, exploit/ASV verdict, confusion class,
status code, latency, prompt/response, token counts, success flag and free-form
metadata. Writes are lock-protected and flushed periodically, so evidence
survives a crashed run.

---

## 4. Research harness (Chapter 3)

### Dataset

`dataset/owasp_2026_id.csv` — 40 Indonesian cases, 4 per OWASP 2026 category,
balanced across `Single-Turn`/`Multi-Turn` and `contaminated`/`clean`. Multi-turn
payloads separate turns with `||`.

```bash
python dataset/loader.py          # schema + coverage validation
```

### Guardrails (application layer)

| Module | Purpose |
|---|---|
| `guardrails/input_filter.py` | injection/override, system-prompt extraction, excessive agency, secret extraction, supply chain, DoS length/repetition rules |
| `guardrails/output_filter.py` | system-prompt leakage, credentials/PII (NIK, phone, email), executable HTML/markdown/code |
| `guardrails/pipeline.py` | `(allowed, reason, sanitized_text)` for input and output |

Every block returns a stable rule id (e.g. `INJ_OVERRIDE_01`) that is stored in
telemetry, so the thesis can report per-rule precision.

### Running a campaign

```bash
# 1) baseline: no mitigation
python test_runner.py --phase baseline --concurrency 1

# 2) mitigated: guardrails active
python test_runner.py --phase mitigated --concurrency 1

# 3) before/after comparison (writes reports/summary.{csv,md})
python metrics/evaluate.py
```

Outputs: `reports/{baseline,mitigated}.jsonl`, `.csv`, and `reports/summary.csv`
/ `.md` with ASR, mean ASV, ΔASR, ΔASV, TP/FP/TN/FN, FPR and FNR per category
and overall.

Useful flags: `--limit`, `--test-id LLM01-001`, `--concurrency`, `--output`,
`--num-predict`, `--system-prompt`.

---

## 5. Repository layout

```
setup.sh                      single entry point
test_runner.py                async baseline/mitigated campaign runner
dataset/
  owasp_2026_id.csv           Indonesian OWASP 2026 corpus
  loader.py                   schema validation + TestCase model
guardrails/
  input_filter.py             pre-inference scanner
  output_filter.py            post-inference scanner + redaction
  pipeline.py                 unified (allowed, reason, text) wrapper
metrics/
  judge.py                    ASR/ASV heuristic judge
  evaluate.py                 before/after + confusion-matrix report
scripts/
  lib/common.sh               shared config, logging, retries, runtime detection
  setup_llm.sh                Ollama (docker auto / native fallback)
  setup_model.sh              HF download + Modelfile render + ollama create
  setup_observability.sh      exporter (+ optional Prometheus/Grafana)
  verify_stack.sh             readiness + inference smoke test
config/Modelfile.template     Llama 3.1 chat template + generation params
observability/
  gpu_exporter.py             Prometheus exporter (GPU/Ollama/host)
  telemetry.py                async-safe JSONL + CSV recorder (security schema)
  prometheus.yml              scrape config
  docker-compose.yml          Prometheus + Grafana
  grafana/                    provisioning + dashboard
```

---

## 6. Roadmap

- [x] Environment bootstrap, model serving, observability
- [x] Indonesian OWASP 2026 test corpus (single/multi-turn, contaminated/clean)
- [x] Application-layer guardrails (input/output, explainable rule ids)
- [x] Async `test_runner.py` (baseline ASR/ASV + mitigated confusion matrix)
- [x] Before/after analysis, FPR/FNR, Markdown/CSV report generation
- [ ] Expand corpus with adjudicated human labels and inter-rater agreement
- [ ] Optional second judge (LLM-as-judge) for ASV calibration
