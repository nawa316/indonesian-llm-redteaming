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
│  test runner / mitigation            (next milestone)              │
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
and scans the rest with `RET_POISON_*` rules (`mitigation/rag_filter.py`).

RAG activates automatically for `LLM09`/vector cases, or globally:

```bash
python test_runner.py --phase baseline --use-rag
```

### Client-server split (Figure 3.2)

`app_server.py` hosts the **Controlled Testbed**: mitigation, ChromaDB RAG and
Ollama. The runner then only needs network access, no GPU or local models.

```
Test Client (laptop)                         Controlled Testbed (Runpod)
┌───────────────────────┐   HTTP/JSON       ┌──────────────────────────────┐
│ test_runner.py        │ ────────────────► │ app_server.py :8000          │
│ judge + Telemetry     │ ◄──────────────── │ mitigation → RAG → Ollama    │
│ reports/*.jsonl|csv   │                   │ ChromaDB + Llama-Sahabat 70B │
└───────────────────────┘                   └──────────────────────────────┘
```

Start the server on the pod (also included in `./setup.sh all`):

```bash
./setup.sh app            # background on 0.0.0.0:8000, logs to /workspace/logs
curl -s localhost:8000/health | python3 -m json.tool
```

Then run the entire campaign (`baseline` + `mitigated` + `summary.{csv,md}`)
from the laptop in **one command** with automatic SSH port-forwarding (no manual
`ssh -L` terminal needed):

```bash
python test_runner.py --ssh-user root --ssh-host <pod-host> --ssh-port <port>
# or shorthand:
python test_runner.py --ssh root@<pod-host>:<port>
```

You can also run against an already-exposed HTTP URL or individual phases:

```bash
python test_runner.py --app-url http://127.0.0.1:8000 --phase baseline
python test_runner.py --app-url http://127.0.0.1:8000 --phase mitigated
python metrics/evaluate.py
```

`SSH_HOST`/`SSH_PORT`/`SSH_USER` or `APP_BASE_URL` in `.env` enable the same
behaviour without flags (`RUNNER_MODE=auto`). Use `--mode direct` to force the
original in-process execution on a single machine. In remote mode telemetry also
stores `server_latency_ms`, `rag_metadata` and the raw model response.

### Mitigation (application layer)

| Module | Purpose |
|---|---|
| `mitigation/input_filter.py` | injection/override, system-prompt extraction, excessive agency, secret extraction, supply chain, DoS length/repetition rules |
| `mitigation/output_filter.py` | system-prompt leakage, credentials/PII (NIK, phone, email), executable HTML/markdown/code |
| `mitigation/rag_filter.py` | tenant access control + indirect injection scanning on retrieved chunks |
| `mitigation/pipeline.py` | `(allowed, reason, sanitized_text)` for input, output and RAG |

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

**One-command execution (Remote via automatic SSH tunnel — baseline + mitigated + evaluation):**

```bash
python test_runner.py --ssh-user root --ssh-host <pod-host> --ssh-port <port>
# or shorthand:
python test_runner.py --ssh root@<pod-host>:<port>
```

**Local / step-by-step execution:**

```bash
python rag/seed_chroma.py --reset

# 1) baseline + mitigated + before/after summary in one command
python test_runner.py

# or run each phase individually:
python test_runner.py --phase baseline --concurrency 1
python test_runner.py --phase mitigated --concurrency 1
python metrics/evaluate.py
```

Outputs: `reports/{baseline,mitigated}.jsonl`, `.csv`, and `reports/summary.csv`
/ `.md` with ASR, mean ASV, compliance, relevance, toxicity, ΔASR, ΔASV,
TP/FP/TN/FN, FPR and FNR per category, per severity and overall.

Useful flags: `--phase {all,baseline,mitigated}`, `--ssh user@host:port`,
`--ssh-host`, `--ssh-port`, `--ssh-user`, `--ssh-key`, `--limit`,
`--test-id LLM01-001`, `--concurrency`, `--output`, `--app-url`,
`--mode {auto,direct,remote}`, `--use-rag`, `--rag-top-k`, `--temperature`,
`--seed`, `--num-predict`, `--system-prompt`.

---

## 5. Repository layout & project structure

```text
indonesian-llm-redteaming/
├── setup.sh                      # Single entry point for server bootstrap (Linux/Runpod)
├── app_server.py                 # Controlled Testbed FastAPI server (:8000)
├── test_runner.py                # Async test client (auto SSH tunnel / remote / direct)
├── .env.example                  # Default configuration & control variables template
├── requirements.txt              # Python dependencies
├── config/
│   └── Modelfile.template        # Llama 3.1 chat template + deterministic decoding params
├── dataset/
│   ├── owasp_2026_id.csv         # 40 Indonesian OWASP Top 10 2026 test cases (+ severity)
│   ├── rag_documents_id.json     # 14 RAG corpus docs (public, confidential, poisoned)
│   └── loader.py                 # CSV schema validator + TestCase dataclass
├── rag/
│   ├── chroma_store.py           # Persistent ChromaDB wrapper + Ollama/TF-IDF embeddings
│   └── seed_chroma.py            # Idempotent RAG corpus seeder & retrieval verifier
├── mitigation/
│   ├── input_filter.py           # Layer 1: pre-inference heuristic & DoS input scanner
│   ├── rag_filter.py             # Layer 2: tenant ACL (public-only) & RAG poisoning filter
│   ├── output_filter.py          # Layer 3: post-inference leak, PII & XSS scanner/redactor
│   └── pipeline.py               # Unified GuardrailPipeline orchestrating all 3 layers
├── metrics/
│   ├── judge.py                  # Deterministic ASR, ASV, Compliance, Relevance & Toxicity judge
│   └── evaluate.py               # Before/After ΔASR, ΔASV, Confusion Matrix, FPR & FNR calculator
├── scripts/
│   ├── lib/common.sh             # Shared env variables, paths (/workspace), logging & GPU detection
│   ├── setup_llm.sh              # Ollama installer & starter (Docker auto / native Runpod fallback)
│   ├── setup_model.sh            # Hugging Face GGUF downloader, SHA-256 check & Modelfile renderer
│   ├── setup_observability.sh    # GPU/Ollama exporter starter (+ optional Prometheus/Grafana)
│   ├── setup_app.sh              # Background daemon starter for app_server.py (:8000)
│   └── verify_stack.sh           # End-to-end hardware, service & inference readiness check
└── observability/
    ├── gpu_exporter.py           # Prometheus metrics exporter (:9101) for GPU, VRAM, Ollama & host
    ├── telemetry.py              # Thread/async-safe JSONL & CSV evidence recorder
    ├── prometheus.yml            # Prometheus scrape configuration
    ├── docker-compose.yml        # Optional Prometheus + Grafana container stack
    └── grafana/                  # Auto-provisioned datasource & LLM Engine dashboard
```

### Detailed explanation of each component

1. **Root Entry Points (`setup.sh`, `app_server.py`, `test_runner.py`)**
   - `setup.sh`: Master bootstrap script run on the Linux GPU server (Runpod). Running `./setup.sh all` sequentially executes Ollama installation, 70B GGUF model download, ChromaDB seeding, observability exporter, and starts `app_server.py`.
   - `app_server.py`: FastAPI server listening on `:8000` inside the **Controlled Testbed Environment** (Figure 3.2). It wraps `mitigation/`, `rag/`, and Ollama's `/api/chat` behind `POST /api/app/chat` and `GET /health`.
   - `test_runner.py`: Cross-platform async client run on the researcher's machine (**Test Client Environment**). Supports automatic background SSH port-forwarding (`--ssh user@host:port` or `--ssh-host`/`--ssh-port`/`--ssh-user`) and executes both `baseline` and `mitigated` phases plus final evaluation in a single command (`--phase all`).

2. **Configuration (`config/` & `.env.example`)**
   - `config/Modelfile.template`: Defines the official Llama 3.1 chat template (`<|start_header_id|>...<|eot_id|>`) and locks the thesis control variables (`temperature=0.0`, `top_p=1.0`, `top_k=1`, `seed=42`, `repeat_penalty=1.1`, `num_ctx=8192`, `num_predict=512`) alongside the Indonesian baseline `SYSTEM_PROMPT`.

3. **Test Corpus & Loader (`dataset/`)**
   - `dataset/owasp_2026_id.csv`: 40 structured Indonesian test scenarios (4 per OWASP 2026 category `LLM01`–`LLM10`), balanced across `Single-Turn` vs `Multi-Turn` (turns separated by `||`) and `contaminated` (attack payloads) vs `clean` (benign control prompts for False Positive testing), labeled with `Critical`/`High`/`Medium`/`Low` severity.
   - `dataset/rag_documents_id.json`: 14 Indonesian internal organizational documents used for RAG testing, divided into `public` SOPs, `confidential` records (credentials, database config, employee NIK/PII, cross-tenant data), and `is_poisoned` documents containing indirect prompt injections.
   - `dataset/loader.py`: Validates column schema, category formatting, severity labels, and multi-turn separators before any test run begins.

4. **Retrieval-Augmented Generation (`rag/`)**
   - `rag/chroma_store.py`: Manages the persistent ChromaDB vector collection (`owasp_id_docs`), supporting Ollama embeddings (`nomic-embed-text`) with an automatic offline deterministic character n-gram TF-IDF fallback.
   - `rag/seed_chroma.py`: Resets, seeds, and verifies the ChromaDB vector store from `dataset/rag_documents_id.json`.

5. **Application-Layer Mitigation (`mitigation/`)**
   - `mitigation/input_filter.py` (**Layer 1**): Scans incoming user prompts before inference. Enforces character/token length and repetition bounds (`DOS_*`) and matches imperative attack patterns (`INJ_OVERRIDE_*`, `INJ_SYSTEM_PROMPT_*`, `INJ_AGENCY_*`, `INJ_SECRET_*`, `INJ_SUPPLY_01`, `HARM_*`, `INJ_REGIONAL_01`, etc.) while allowing defensive/educational questions to pass.
   - `mitigation/rag_filter.py` (**Layer 2**): Enforces tenant access control (`access_level="public"`) during ChromaDB retrieval in the `mitigated` phase, drops documents flagged as `is_poisoned`, and scans retrieved text chunks against indirect injection rules (`RET_POISON_*`).
   - `mitigation/output_filter.py` (**Layer 3**): Inspects model responses after inference and redacts leaked system-prompt fragments (`OUT_LEAK_*`), credentials/tokens (`OUT_SECRET_*`), Indonesian PII such as 16-digit NIK, phone numbers, and emails (`OUT_PII_*`), and executable HTML/JS/code (`OUT_UNSAFE_*`).
   - `mitigation/pipeline.py`: Combines `InputFilter`, `RAGFilter`, and `OutputFilter` into a single `GuardrailPipeline` interface returning explainable `GuardrailDecision` objects with stable `rule_id`s.

6. **Scoring & Evaluation (`metrics/`)**
   - `metrics/judge.py`: Deterministic response evaluator that computes binary `exploit_success` (for **ASR**), continuous `asv_score` ($0.55 \times \text{keyword\_coverage} + 0.45 \times \text{semantic\_similarity}$ with refusal penalty and toxicity adjustment), and complementary metrics (`compliance_score`, `relevance_score`, `toxicity_flag`).
   - `metrics/evaluate.py`: Compares `reports/baseline.csv` and `reports/mitigated.csv` to compute $\Delta\text{ASR}$, $\Delta\text{ASV}$, Confusion Matrix counts (`TP`, `FP`, `TN`, `FN`), **FPR**, and **FNR** across overall, per-OWASP category, and per-severity scopes, exporting `reports/summary.csv` and `reports/summary.md`.

7. **Server Bootstrap Scripts (`scripts/`)**
   - `scripts/lib/common.sh`: Central configuration sourced by all shell scripts; maps mutable directories (`models/`, `chromadb/`, `logs/`, `runtime/`) to Runpod's persistent `/workspace` volume and provides GPU (`nvidia-smi`) and Docker detection helpers.
   - `scripts/setup_llm.sh`, `setup_model.sh`, `setup_observability.sh`, `setup_app.sh`, `verify_stack.sh`: Modular, idempotent shell scripts invoked by `./setup.sh`.

8. **Telemetry & Hardware Observability (`observability/`)**
   - `observability/telemetry.py`: Lock-protected JSONL and CSV writer that persists every test turn's prompt, response, latency, token counts, guardrail decision, rule ID, confusion class, and judge scores to `reports/`.
   - `observability/gpu_exporter.py`: Lightweight HTTP Prometheus exporter (`:9101/metrics`) tracking GPU utilization, VRAM usage, temperature, power draw, host CPU/RAM, and Ollama loaded-model footprint.

---

## 6. Roadmap

- [x] Environment bootstrap, model serving, observability
- [x] Deterministic decoding (temp 0 / top_p 1 / top_k 1 / seed 42)
- [x] Indonesian OWASP 2026 test corpus with severity labels
- [x] ChromaDB RAG with tenant ACL + indirect-injection filtering
- [x] Client-server split (app_server :8000 ↔ remote test_runner)
- [x] Application-layer mitigation (input/output/RAG, explainable rule ids)
- [x] Async `test_runner.py` (baseline ASR/ASV + mitigated confusion matrix)
- [x] Semantic ASV + compliance/relevance/toxicity metrics
- [x] Before/after analysis, per-severity, FPR/FNR, Markdown/CSV reports
- [ ] Expand corpus with adjudicated human labels and inter-rater agreement
- [ ] Optional second judge (LLM-as-judge) for ASV calibration
