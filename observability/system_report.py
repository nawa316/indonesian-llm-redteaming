#!/usr/bin/env python3
"""System, hardware, setup, and API report collector & Markdown renderer.

Used by:
  * ``app_server.py`` (``GET /api/app/report``) to report server-side CPU, RAM,
    GPU, disk, Ollama, setup timing, and API call statistics.
  * ``test_runner.py`` to fetch the server report when a campaign finishes
    (or collect locally in direct mode) and persist ``system_report.json`` and
    ``system_report.md`` on the client.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parent.parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "[N/A]", "N/A", "[Not Supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sample_gpus() -> list[dict[str, Any]]:
    """Query NVIDIA GPU metrics via nvidia-smi (returns [] if unavailable)."""
    if shutil.which("nvidia-smi") is None:
        return []
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
        return []

    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, util, used, total, temp, power = parts[:7]
        gpus.append(
            {
                "index": idx,
                "name": name,
                "utilization_percent": _to_float(util),
                "memory_used_mb": _to_float(used),
                "memory_total_mb": _to_float(total),
                "temperature_celsius": _to_float(temp),
                "power_watts": _to_float(power),
            }
        )
    return gpus


def collect_system_snapshot(disk_path: str | Path | None = None) -> dict[str, Any]:
    """Collect a point-in-time snapshot of host CPU, RAM, disk, and GPU."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    target_disk = str(disk_path or os.getenv("WORKSPACE") or ROOT)
    if not Path(target_disk).exists():
        target_disk = str(ROOT)
    disk = psutil.disk_usage(target_disk)

    load_avg: list[float] | None = None
    if hasattr(os, "getloadavg"):
        try:
            load_avg = [round(v, 2) for v in os.getloadavg()]
        except OSError:
            load_avg = None

    return {
        "ts_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {
            "percent": round(cpu_percent, 2),
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "load_avg": load_avg,
        },
        "memory": {
            "used_mb": round(mem.used / (1024**2), 1),
            "total_mb": round(mem.total / (1024**2), 1),
            "percent": round(mem.percent, 2),
        },
        "disk": {
            "path": target_disk,
            "used_gb": round(disk.used / (1024**3), 2),
            "total_gb": round(disk.total / (1024**3), 2),
            "percent": round(disk.percent, 2),
        },
        "gpus": sample_gpus(),
    }


def load_setup_report() -> dict[str, Any] | None:
    """Load setup_report.json written by setup.sh if present."""
    workspace = Path(os.getenv("WORKSPACE", "/workspace"))
    log_dir = Path(os.getenv("LOG_DIR", str(workspace / "logs")))
    candidates = [
        log_dir / "setup_report.json",
        ROOT / "reports" / "setup_report.json",
        ROOT / ".runtime" / "logs" / "setup_report.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return None


def render_system_report_markdown(report: dict[str, Any]) -> str:
    """Render the combined campaign + server + setup + system report as Markdown."""
    lines: list[str] = [
        "# System, Setup & API Report",
        "",
        f"- Generated (UTC): `{report.get('ts_utc', '')}`",
        f"- Execution mode: `{report.get('execution_mode', 'unknown')}`",
    ]

    campaign = report.get("campaign") or {}
    if campaign:
        lines.extend(
            [
                "",
                "## Campaign Timing",
                "",
                f"- Started (UTC): `{campaign.get('started_utc', '')}`",
                f"- Finished (UTC): `{campaign.get('finished_utc', '')}`",
                f"- Total duration: **{campaign.get('elapsed_seconds', 0):.2f} s**",
            ]
        )
        phases = campaign.get("phases") or []
        if phases:
            lines.extend(
                [
                    "",
                    "| Phase | Cases | Duration (s) | Throughput (cases/s) |",
                    "|---|---:|---:|---:|",
                ]
            )
            for p in phases:
                lines.append(
                    f"| {p.get('phase', '')} | {p.get('cases', 0)} | "
                    f"{p.get('elapsed_seconds', 0.0):.2f} | "
                    f"{p.get('cases_per_second', 0.0):.3f} |"
                )

    server_report = report.get("server_report") or {}
    setup = server_report.get("setup") or report.get("setup")
    if setup:
        lines.extend(
            [
                "",
                "## Setup Timing",
                "",
                f"- Command: `./setup.sh {setup.get('command', 'all')}`",
                f"- Started (UTC): `{setup.get('started_utc', '')}`",
                f"- Finished (UTC): `{setup.get('finished_utc', '')}`",
                f"- Total setup duration: **{setup.get('total_seconds', 0):.2f} s**",
            ]
        )
        steps = setup.get("steps") or []
        if steps:
            lines.extend(
                [
                    "",
                    "| Step | Status | Duration (s) |",
                    "|---|---|---:|",
                ]
            )
            for step in steps:
                lines.append(
                    f"| {step.get('name', '')} | {step.get('status', '')} | "
                    f"{step.get('elapsed_seconds', 0.0):.2f} |"
                )

    api = server_report.get("api") or {}
    if api:
        lat = api.get("latency_ms") or {}
        tokens = api.get("tokens") or {}
        lines.extend(
            [
                "",
                "## Server API Statistics",
                "",
                f"- Total HTTP requests: **{api.get('total_requests', 0)}**",
                f"- Chat requests: **{api.get('chat_requests', 0)}** "
                f"(baseline: {api.get('by_phase', {}).get('baseline', 0)}, "
                f"mitigated: {api.get('by_phase', {}).get('mitigated', 0)})",
                f"- Guardrail blocked: **{api.get('guardrail_blocked', 0)}**",
                f"- Errors: **{api.get('errors', 0)}**",
                f"- Server latency (ms): min={lat.get('min', 0):.1f}, "
                f"avg={lat.get('avg', 0):.1f}, max={lat.get('max', 0):.1f}",
                f"- Tokens: prompt={tokens.get('prompt', 0)}, "
                f"completion={tokens.get('completion', 0)}",
            ]
        )

    sys_info = server_report.get("system") or report.get("system") or {}
    if sys_info:
        cpu = sys_info.get("cpu") or {}
        mem = sys_info.get("memory") or {}
        disk = sys_info.get("disk") or {}
        lines.extend(
            [
                "",
                "## Hardware & System Snapshot",
                "",
                f"- Host: `{sys_info.get('hostname', '')}` ({sys_info.get('platform', '')})",
                f"- CPU: **{cpu.get('percent', 0)}%** "
                f"({cpu.get('logical_cores', '?')} logical / {cpu.get('physical_cores', '?')} physical cores)",
                f"- RAM: **{mem.get('used_mb', 0):.1f} / {mem.get('total_mb', 0):.1f} MB** "
                f"({mem.get('percent', 0)}%)",
                f"- Disk (`{disk.get('path', '')}`): **{disk.get('used_gb', 0):.2f} / "
                f"{disk.get('total_gb', 0):.2f} GB** ({disk.get('percent', 0)}%)",
            ]
        )
        gpus = sys_info.get("gpus") or []
        if gpus:
            lines.extend(
                [
                    "",
                    "| GPU | Name | Util % | VRAM Used / Total (MB) | Temp (°C) | Power (W) |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for g in gpus:
                lines.append(
                    f"| {g.get('index', '')} | {g.get('name', '')} | "
                    f"{g.get('utilization_percent', 'N/A')} | "
                    f"{g.get('memory_used_mb', 'N/A')} / {g.get('memory_total_mb', 'N/A')} | "
                    f"{g.get('temperature_celsius', 'N/A')} | "
                    f"{g.get('power_watts', 'N/A')} |"
                )

    lines.append("")
    return "\n".join(lines)
