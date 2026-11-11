#!/usr/bin/env python3
"""Async-safe telemetry recorder for red-team test runs.

Every model interaction is appended as one JSON line (crash-safe evidence),
then optionally exported to CSV for analysis with pandas.

Schema (JSONL)
-------------
ts_utc            ISO-8601 UTC timestamp
run_id            UUID shared by all events of one campaign
test_id           stable identifier of the test case
category          OWASP LLM Top-10 2026 category (e.g. LLM01)
technique         e.g. single_turn | multi_turn
turn              turn index inside a conversation (0-based)
status_code       HTTP status returned by the target API
latency_ms        wall-clock latency of the request
model             served model name
prompt            exact prompt sent
response          exact model output
prompt_tokens     tokens reported by Ollama (if any)
completion_tokens tokens reported by Ollama (if any)
success           bool: request completed without transport error
error             error string when success is false
metadata          free-form dict (mitigation flags, judge scores, ...)
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class TelemetryEvent:
    test_id: str
    category: str = ""
    technique: str = ""
    turn: int = 0
    status_code: int | None = None
    latency_ms: float | None = None
    model: str = ""
    prompt: str = ""
    response: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    ts_utc: str = ""

    def finalize(self, run_id: str) -> "TelemetryEvent":
        self.run_id = self.run_id or run_id
        self.ts_utc = self.ts_utc or utc_now()
        return self


class TelemetryRecorder:
    """Thread- and task-safe JSONL writer with CSV export.

    Parameters
    ----------
    path: destination ``.jsonl`` file. A ``.csv`` sibling is produced by
        :meth:`export_csv`.
    flush_every: flush to disk every N events (and always on close).
    """

    FIELDS = [
        "ts_utc", "run_id", "test_id", "category", "technique", "turn",
        "status_code", "latency_ms", "model", "prompt", "response",
        "prompt_tokens", "completion_tokens", "success", "error", "metadata",
    ]

    def __init__(self, path: str | os.PathLike[str], *, flush_every: int = 10) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(uuid.uuid4())
        self.flush_every = max(1, flush_every)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._count = 0

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "TelemetryRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- recording ---------------------------------------------------------
    def record(self, event: TelemetryEvent | dict[str, Any]) -> TelemetryEvent:
        if isinstance(event, dict):
            event = TelemetryEvent(**event)
        event.finalize(self.run_id)
        line = json.dumps(asdict(event), ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._count += 1
            if self._count % self.flush_every == 0:
                self._fh.flush()
                os.fsync(self._fh.fileno())
        return event

    async def record_async(self, event: TelemetryEvent | dict[str, Any]) -> TelemetryEvent:
        """Record without blocking the event loop (fsync happens in a thread)."""
        return await asyncio.to_thread(self.record, event)

    def timed(self, **fields: Any) -> "_TimedCall":
        return _TimedCall(self, fields)

    # -- export ------------------------------------------------------------
    def export_csv(self, destination: str | os.PathLike[str] | None = None) -> Path:
        dest = Path(destination) if destination else self.path.with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        with dest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                if isinstance(row.get("metadata"), (dict, list)):
                    row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
                writer.writerow(row)
        return dest

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()


class _TimedCall:
    """Context manager measuring latency and emitting one telemetry event."""

    def __init__(self, recorder: TelemetryRecorder, fields: dict[str, Any]) -> None:
        self.recorder = recorder
        self.fields = fields
        self.started = 0.0

    def __enter__(self) -> "_TimedCall":
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.fields.setdefault("latency_ms", round((time.perf_counter() - self.started) * 1000, 3))
        if exc is not None:
            self.fields["success"] = False
            self.fields["error"] = f"{exc_type.__name__}: {exc}"
        self.recorder.record(self.fields)
        return False


if __name__ == "__main__":  # tiny self-check
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rec = TelemetryRecorder(Path(tmp) / "demo.jsonl")
        with rec.timed(test_id="demo-1", category="LLM01", prompt="halo", response="hai"):
            time.sleep(0.01)
        rec.record(TelemetryEvent(test_id="demo-2", category="LLM06", success=False, error="timeout"))
        csv_path = rec.export_csv()
        rec.close()
        print(f"jsonl={rec.path} csv={csv_path} run_id={rec.run_id}")
        print(csv_path.read_text(encoding="utf-8"))
