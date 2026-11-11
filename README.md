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
payloads separate turns with `||`. Each case carries a `severity` label
(`Critical`/`High`/`Medium`/`Low`) used for per-severity reporting.

```bash
python dataset/loader.py          # schema + coverage validation
```

### Deterministic control variables (v1.8.2)

Baseline and mitigated campaigns share identical decoding settings:

| Parameter | Value | Reason |
|---|---|---|
| `temperature` | `0.0` | greedy decoding, no sampling variance |
| `top_p` | `1.0` | full distribution under greedy |
| `top_k` | `1` | strict top-1 candidate |
| `seed` | `42` | fixed RNG |
| `repeat_penalty` | `1.1` | avoids 3-bit repetition loops |
| `num_ctx` | `8192` | multi-turn history + RAG chunks |
| `num_predict` | `512` | per-turn generation cap |

The fixed Indonesian system prompt lives in `.env.example`, `config/Modelfile.template`
(`SYSTEM """..."""`), and `test_runner.py`; all three are byte-identical. To
apply it to the served model:

```bash
bash scripts/setup_model.sh --force   # re-renders Modelfile + ollama create
```

### RAG (ChromaDB)

`rag/chroma_store.py` provides a persistent Chroma collection with Ollama
embeddings (`nomic-embed-text`) and a deterministic scikit-learn character
n-gram fallback when the embedding model is absent.

```bash
python rag/seed_chroma.py --reset --verify
```

(`./setup.sh rag` performs the same reset and seed.)

Documents carry `access_level` (`public`/`confidential`) and `is_poisoned`.
In **baseline** the retriever has no access filter and injects every top-k
chunk; in **mitigated** it queries `access_level=public`, drops poisoned chunks,
and scans the rest with `RET_POISON_*` rules (`guardrails/rag_filter.py`).

RAG activates automatically for `LLM09`/vector cases, or globally:

```bash
python test_runner.py --phase baseline --use-rag
```

### Guardrails (application layer)

| Module | Purpose |
|---|---|
| `guardrails/input_filter.py` | injection/override, system-prompt extraction, excessive agency, secret extraction, supply chain, DoS length/repetition rules |
| `guardrails/output_filter.py` | system-prompt leakage, credentials/PII (NIK, phone, email), executable HTML/markdown/code |
| `guardrails/rag_filter.py` | tenant access control + indirect injection scanning on retrieved chunks |
| `guardrails/pipeline.py` | `(allowed, reason, sanitized_text)` for input, output and RAG |

Every block returns a stable rule id (e.g. `INJ_OVERRIDE_01`, `RET_POISON_EXFIL`)
that is stored in telemetry, so the thesis can report per-rule precision.

### Metrics

`metrics/judge.py` combines keyword coverage, TF-IDF/cosine semantic similarity,
refusal detection and toxicity markers to produce `exploit_success` and
`asv_score` (0–1), plus the three complementary metrics from Section 2.2.6.3:

| Metric | Meaning |
|---|---|
| `compliance_score` | adherence to safety/role boundaries (1 = compliant) |
| `relevance_score` | relevance of the answer to the prompt |
| `toxicity_flag` | harmful/dangerous content markers present |

### Running a campaign

```bash
python rag/seed_chroma.py --reset

# 1) baseline: no mitigation, unfiltered RAG
python test_runner.py --phase baseline --concurrency 1

# 2) mitigated: guardrails + filtered RAG
python test_runner.py --phase mitigated --concurrency 1

# 3) before/after comparison (writes reports/summary.{csv,md})
python metrics/evaluate.py
```

Outputs: `reports/{baseline,mitigated}.jsonl`, `.csv`, and `reports/summary.csv`
/ `.md` with ASR, mean ASV, compliance, relevance, toxicity, ΔASR, ΔASV,
TP/FP/TN/FN, FPR and FNR per category, per severity and overall.

Useful flags: `--limit`, `--test-id LLM01-001`, `--concurrency`, `--output`,
`--use-rag`, `--rag-top-k`, `--temperature`, `--seed`, `--num-predict`,
`--system-prompt`.

---

## 5. Repository layout

```
setup.sh                      single entry point
test_runner.py                async baseline/mitigated campaign runner
dataset/
  owasp_2026_id.csv           Indonesian OWASP 2026 corpus (+ severity)
  rag_documents_id.json       public / confidential / poisoned RAG docs
  loader.py                   schema validation + TestCase model
rag/
  chroma_store.py             ChromaDB + Ollama/offline embeddings
  seed_chroma.py              idempotent seeding + retrieval verification
guardrails/
  input_filter.py             pre-inference scanner
  output_filter.py            post-inference scanner + redaction
  rag_filter.py               tenant ACL + indirect-injection scanning
  pipeline.py                 unified (allowed, reason, text) wrapper
metrics/
  judge.py                    semantic ASR/ASV + complementary metrics
  evaluate.py                 before/after, severity, confusion matrix
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
- [x] Deterministic decoding (temp 0 / top_p 1 / top_k 1 / seed 42)
- [x] Indonesian OWASP 2026 test corpus with severity labels
- [x] ChromaDB RAG with tenant ACL + indirect-injection filtering
- [x] Application-layer guardrails (input/output/RAG, explainable rule ids)
- [x] Async `test_runner.py` (baseline ASR/ASV + mitigated confusion matrix)
- [x] Semantic ASV + compliance/relevance/toxicity metrics
- [x] Before/after analysis, per-severity, FPR/FNR, Markdown/CSV reports
- [ ] Expand corpus with adjudicated human labels and inter-rater agreement
- [ ] Optional second judge (LLM-as-judge) for ASV calibration
