#!/usr/bin/env python3
"""Async red-team runner for the Llama-Sahabat-AI v2 deployment.

Phases
------
baseline
    Every payload (including benign ``clean`` controls) goes straight to
    ``/api/chat``. Contaminated cases are scored by ``metrics.judge`` to obtain
    ASR (binary) and ASV (0-1).

mitigated
    Inputs pass through ``guardrails.pipeline`` first. Blocked requests never
    reach the model. Allowed requests are screened again on output. Decisions
    map to a confusion matrix:

        contaminated + blocked -> TP
        contaminated + allowed -> FN   (guardrail missed, regardless of refusal)
        clean        + allowed -> TN
        clean        + blocked -> FP

Usage
-----
    python test_runner.py --phase baseline  --concurrency 1
    python test_runner.py --phase mitigated --concurrency 2
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.loader import TestCase, load_dataset  # noqa: E402
from guardrails.pipeline import GuardrailPipeline  # noqa: E402
from metrics.judge import judge_response  # noqa: E402
from observability.telemetry import TelemetryEvent, TelemetryRecorder  # noqa: E402


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (no external dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "sahabat-70b-q3ks")


def classify(data_type: str, phase: str, blocked: bool) -> str:
    if phase == "baseline":
        return "N/A"
    if data_type == "contaminated":
        return "TP" if blocked else "FN"
    return "FP" if blocked else "TN"


class TestRunner:
    def __init__(
        self,
        *,
        phase: str,
        base_url: str,
        model: str,
        output_dir: Path,
        concurrency: int = 1,
        timeout: float = 600.0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        num_ctx: int = 8192,
        num_predict: int = 512,
        system_prompt: str = "",
        guardrails: GuardrailPipeline | None = None,
        append: bool = False,
        verbose: bool = True,
    ) -> None:
        self.phase = phase
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.system_prompt = system_prompt
        self.pipeline = guardrails or GuardrailPipeline()
        self.append = append
        self.verbose = verbose
        self._semaphore = asyncio.Semaphore(self.concurrency)

    # -- HTTP --------------------------------------------------------------
    async def _chat(
        self, client: httpx.AsyncClient, messages: list[dict[str, str]]
    ) -> tuple[int | None, str, int, int, float, str | None]:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        started = time.perf_counter()
        try:
            response = await client.post(f"{self.base_url}/api/chat", json=body)
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            response.raise_for_status()
            data = response.json()
            message = (data.get("message") or {}).get("content", "") or ""
            return (
                response.status_code,
                message,
                int(data.get("prompt_eval_count") or 0),
                int(data.get("eval_count") or 0),
                latency_ms,
                None,
            )
        except httpx.HTTPStatusError as exc:
            return (
                exc.response.status_code,
                "",
                0,
                0,
                round((time.perf_counter() - started) * 1000, 3),
                f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
            )
        except (httpx.HTTPError, ValueError) as exc:
            return (
                None,
                "",
                0,
                0,
                round((time.perf_counter() - started) * 1000, 3),
                f"{type(exc).__name__}: {exc}",
            )

    # -- one case ----------------------------------------------------------
    def _base_event(self, case: TestCase, *, turn_count: int) -> TelemetryEvent:
        return TelemetryEvent(
            test_id=case.test_id,
            phase=self.phase,
            category=case.owasp_category,
            technique=case.flow.lower().replace("-", "_"),
            turn=turn_count,
            data_type=case.data_type,
            model=self.model,
            prompt=case.payload,
            metadata={
                "scenario_type": case.scenario_type,
                "expected_behavior": case.expected_behavior,
                "target_keywords": case.target_keywords,
            },
        )

    async def run_case(self, client: httpx.AsyncClient, case: TestCase) -> TelemetryEvent:
        async with self._semaphore:
            started = time.perf_counter()
            turns = case.turns
            event = self._base_event(case, turn_count=len(turns))

            blocked = False
            block_stage = ""
            block_reason = ""
            block_rule = ""
            stopped_by_guardrail = False

            # ---- mitigated: input screen -------------------------------
            if self.phase == "mitigated":
                for turn in turns:
                    decision = self.pipeline.screen_input(turn)
                    if not decision.allowed:
                        blocked = True
                        block_stage = "input"
                        block_reason = decision.reason
                        block_rule = decision.rule_id
                        event.metadata["blocked_prompt"] = turn
                        event.metadata["sanitized_prompt"] = decision.sanitized_text
                        stopped_by_guardrail = True
                        break

            messages: list[dict[str, str]] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            responses: list[str] = []
            status_code: int | None = None
            prompt_tokens = 0
            completion_tokens = 0
            transport_error: str | None = None

            if not stopped_by_guardrail:
                for turn in turns:
                    messages.append({"role": "user", "content": turn})
                    status, content, p_tokens, c_tokens, _latency, error = await self._chat(
                        client, messages
                    )
                    status_code = status
                    prompt_tokens += p_tokens
                    completion_tokens += c_tokens

                    if error:
                        transport_error = error
                        break

                    # ---- mitigated: output screen ----------------------
                    if self.phase == "mitigated":
                        out_decision = self.pipeline.screen_output(content)
                        if not out_decision.allowed:
                            blocked = True
                            block_stage = "output"
                            block_reason = out_decision.reason
                            block_rule = out_decision.rule_id
                            event.metadata["raw_response"] = "\n".join(responses + [content])
                            content = out_decision.sanitized_text
                            responses.append(content)
                            break

                    responses.append(content)
                    messages.append({"role": "assistant", "content": content})

            combined_response = "\n".join(responses).strip()

            # ---- scoring ------------------------------------------------
            exploit_success = False
            asv_score = 0.0
            verdict: dict[str, Any] | None = None

            if case.data_type == "contaminated" and not transport_error:
                if blocked:
                    exploit_success = False
                    asv_score = 0.0
                else:
                    judged = judge_response(
                        combined_response,
                        case.target_keywords,
                        expected_behavior=case.expected_behavior,
                    )
                    exploit_success = judged.exploit_success
                    asv_score = judged.asv_score
                    verdict = judged.to_dict()

            event.guardrail_blocked = blocked
            event.guardrail_reason = (
                f"{block_stage}:{block_reason}" if blocked else ""
            )
            event.exploit_success = exploit_success
            event.asv_score = asv_score
            event.confusion_class = classify(case.data_type, self.phase, blocked)
            event.status_code = status_code
            event.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            event.response = combined_response
            event.prompt_tokens = prompt_tokens or None
            event.completion_tokens = completion_tokens or None
            event.success = transport_error is None
            event.error = transport_error
            event.metadata.update({
                "guardrail_stage": block_stage,
                "guardrail_rule": block_rule,
                "turn_responses": responses,
                "judge": verdict,
            })
            return event

    # -- campaign ----------------------------------------------------------
    async def run(self, cases: list[TestCase]) -> tuple[Path, Path]:
        jsonl_path = self.output_dir / f"{self.phase}.jsonl"
        csv_path = self.output_dir / f"{self.phase}.csv"
        recorder = TelemetryRecorder(jsonl_path, append=self.append)

        limits = httpx.Limits(
            max_connections=self.concurrency + 2,
            max_keepalive_connections=self.concurrency + 2,
        )
        timeout = httpx.Timeout(self.timeout, connect=30.0)

        completed = 0
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
                results = await asyncio.gather(
                    *(self.run_case(client, case) for case in cases),
                    return_exceptions=True,
                )
                for case, result in zip(cases, results):
                    if isinstance(result, BaseException):
                        event = self._base_event(case, turn_count=len(case.turns))
                        event.success = False
                        event.error = f"{type(result).__name__}: {result}"
                        event.latency_ms = 0.0
                        event.confusion_class = classify(case.data_type, self.phase, False)
                    else:
                        event = result
                    await recorder.record_async(event)
                    completed += 1
                    if self.verbose:
                        flag = "BLOCK" if event.guardrail_blocked else (
                            "EXPLOIT" if event.exploit_success else "safe"
                        )
                        print(
                            f"[{completed:>3}/{len(cases)}] {event.test_id:<12} "
                            f"{event.confusion_class:<3} {flag:<7} "
                            f"asv={event.asv_score:.2f} {event.latency_ms:>8.0f}ms",
                            file=sys.stderr,
                            flush=True,
                        )
        finally:
            recorder.close()
            recorder.export_csv(csv_path)
        elapsed = time.perf_counter() - started
        print(
            f"\n{self.phase}: {completed} cases in {elapsed:.1f}s "
            f"({completed / elapsed:.2f} cases/s) -> {jsonl_path}",
            flush=True,
        )
        return jsonl_path, csv_path

    def print_summary(self, csv_path: Path) -> None:
        from metrics.evaluate import load_report, render_markdown, summarize

        current = load_report(csv_path)
        baseline_path = self.output_dir / "baseline.csv"

        if self.phase == "mitigated" and baseline_path.exists():
            baseline = load_report(baseline_path)
            table = summarize(baseline, current)
            markdown = render_markdown(
                table,
                baseline_path=str(baseline_path),
                mitigated_path=str(csv_path),
            )
        else:
            table = summarize(current, None)
            markdown = render_markdown(
                table,
                baseline_path=str(csv_path),
                mitigated_path=None,
            )
        print()
        print(markdown)

        summary_csv = self.output_dir / f"summary_{self.phase}.csv"
        table.to_csv(summary_csv, index=False, float_format="%.3f")
        print(f"summary -> {summary_csv}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async OWASP-2026 red-team runner for Llama-Sahabat-AI v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", required=True, choices=["baseline", "mitigated"])
    parser.add_argument("--dataset", default=str(ROOT / "dataset" / "owasp_2026_id.csv"))
    parser.add_argument("--output", default=str(ROOT / "reports"))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    parser.add_argument("--test-id", action="append", default=[], help="run only these ids")
    parser.add_argument("--append", action="store_true", help="append instead of overwriting reports")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


async def amain(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)
    if args.test_id:
        wanted = set(args.test_id)
        cases = [c for c in cases if c.test_id in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no test cases selected", file=sys.stderr)
        return 2

    runner = TestRunner(
        phase=args.phase,
        base_url=args.base_url,
        model=args.model,
        output_dir=Path(args.output),
        concurrency=args.concurrency,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        system_prompt=args.system_prompt,
        append=args.append,
        verbose=not args.quiet,
    )
    _, csv_path = await runner.run(cases)
    runner.print_summary(csv_path)

    # exit non-zero if every request failed (useful in CI)
    from metrics.evaluate import load_report

    df = load_report(csv_path)
    if not bool(df["success"].map(lambda v: str(v).strip().casefold() in {"true", "1"}).any()):
        print("all requests failed", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
