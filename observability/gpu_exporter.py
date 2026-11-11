#!/usr/bin/env python3
"""Prometheus exporter for the LLM serving stack.

Runs without Docker and without root, which makes it usable on unprivileged
Runpod pods. It samples:

  * GPU utilisation / VRAM / temperature / power via nvidia-smi
  * Ollama availability and currently loaded models via /api/ps
  * Host CPU / RAM via psutil

Exposed on :9101/metrics by default (EXPORTER_PORT).

Usage:
    python3 observability/gpu_exporter.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import httpx
import psutil
from prometheus_client import Gauge, Info, start_http_server

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
PORT = int(os.getenv("EXPORTER_PORT", "9101"))
INTERVAL = float(os.getenv("EXPORTER_INTERVAL", "5"))

GPU_UTIL = Gauge("llm_gpu_utilization_percent", "GPU utilisation", ["gpu", "name"])
GPU_MEM_USED = Gauge("llm_gpu_memory_used_bytes", "VRAM used", ["gpu", "name"])
GPU_MEM_TOTAL = Gauge("llm_gpu_memory_total_bytes", "VRAM total", ["gpu", "name"])
GPU_TEMP = Gauge("llm_gpu_temperature_celsius", "GPU temperature", ["gpu", "name"])
GPU_POWER = Gauge("llm_gpu_power_watts", "GPU power draw", ["gpu", "name"])

OLLAMA_UP = Gauge("llm_ollama_up", "1 when the Ollama API answers /api/version")
OLLAMA_LOADED = Gauge("llm_ollama_loaded_models", "Number of models resident in memory")
OLLAMA_MODEL_VRAM = Gauge(
    "llm_ollama_model_vram_bytes", "VRAM held by a loaded model", ["model"]
)
OLLAMA_MODEL_SIZE = Gauge(
    "llm_ollama_model_size_bytes", "On-disk size of a loaded model", ["model"]
)

HOST_CPU = Gauge("llm_host_cpu_percent", "Host CPU utilisation")
HOST_MEM = Gauge("llm_host_memory_used_bytes", "Host RAM used")
HOST_MEM_TOTAL = Gauge("llm_host_memory_total_bytes", "Host RAM total")

BUILD = Info("llm_exporter", "Exporter build information")
BUILD.info({"version": "1.0.0", "interval_seconds": str(INTERVAL)})


def _to_float(value: str) -> float:
    value = value.strip()
    if value in {"", "[N/A]", "N/A", "[Not Supported]"}:
        return float("nan")
    return float(value)


def sample_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        return
    query = (
        "index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, util, used, total, temp, power = parts[:7]
        labels = {"gpu": idx, "name": name}
        GPU_UTIL.labels(**labels).set(_to_float(util))
        GPU_MEM_USED.labels(**labels).set(_to_float(used) * 1024**2)
        GPU_MEM_TOTAL.labels(**labels).set(_to_float(total) * 1024**2)
        GPU_TEMP.labels(**labels).set(_to_float(temp))
        GPU_POWER.labels(**labels).set(_to_float(power))


def sample_ollama(client: httpx.Client) -> None:
    try:
        r = client.get(f"{OLLAMA_URL}/api/ps")
        r.raise_for_status()
        models = r.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        OLLAMA_UP.set(0)
        OLLAMA_LOADED.set(0)
        return

    OLLAMA_UP.set(1)
    OLLAMA_LOADED.set(len(models))
    for model in models:
        name = model.get("name") or model.get("model") or "unknown"
        OLLAMA_MODEL_VRAM.labels(model=name).set(float(model.get("size_vram", 0)))
        OLLAMA_MODEL_SIZE.labels(model=name).set(float(model.get("size", 0)))


def sample_host() -> None:
    HOST_CPU.set(psutil.cpu_percent(interval=None))
    mem = psutil.virtual_memory()
    HOST_MEM.set(mem.used)
    HOST_MEM_TOTAL.set(mem.total)


def main() -> None:
    start_http_server(PORT)
    print(f"exporter listening on :{PORT}/metrics (ollama={OLLAMA_URL})", flush=True)
    with httpx.Client(timeout=5.0) as client:
        while True:
            started = time.time()
            sample_gpu()
            sample_ollama(client)
            sample_host()
            time.sleep(max(0.0, INTERVAL - (time.time() - started)))


if __name__ == "__main__":
    main()
