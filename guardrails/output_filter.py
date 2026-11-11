#!/usr/bin/env python3
"""Application-layer output guardrail.

Scans model output for three failure classes before it reaches the user:

* leaked system-prompt fragments / hidden context (LLM08, LLM02)
* credentials and Indonesian PII (LLM02)
* executable HTML, markdown or code (LLM10)

Blocked spans are redacted in ``sanitized_text`` so a caller can choose to
serve a safe variant instead of a hard failure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_RULES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "OUT_LEAK_SYSTEM_01",
        re.compile(
            r"(cutting\s+knowledge\s+date|today\s+date\s*:|environment\s*:\s*ipython|"
            r"<\|start_header_id\|>\s*system|you\s+are\s+a\s+helpful\s+assistant|"
            r"anda\s+adalah\s+asisten\s+AI\s+yang)",
            re.IGNORECASE,
        ),
        "system prompt fragment leaked",
    ),
    (
        "OUT_LEAK_CONTEXT_01",
        re.compile(
            r"(<\|(begin|end)_of_text\|>|<\|eot_id\|>|<\|start_header_id\|>)",
            re.IGNORECASE,
        ),
        "chat template control token leaked",
    ),
    (
        "OUT_SECRET_AWS_01",
        re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        "AWS access key pattern",
    ),
    (
        "OUT_SECRET_APIKEY_01",
        re.compile(r"\b(sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                   r"xox[baprs]-[A-Za-z0-9-]{10,})\b"),
        "API token pattern",
    ),
    (
        "OUT_SECRET_BEARER_01",
        re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*", re.IGNORECASE),
        "bearer token",
    ),
    (
        "OUT_SECRET_ASSIGN_01",
        re.compile(
            r"\b(password|passwd|api[_\s-]?key|secret|token)\b\s*[:=]\s*"
            r"[\"']?([A-Za-z0-9._~+/-]{8,})",
            re.IGNORECASE,
        ),
        "credential assignment",
    ),
    (
        "OUT_PII_NIK_01",
        re.compile(r"(?<!\d)\d{16}(?!\d)"),
        "Indonesian NIK (16 digits)",
    ),
    (
        "OUT_PII_PHONE_01",
        re.compile(r"(?<!\d)(?:\+62|62|0)8\d{8,11}(?!\d)"),
        "Indonesian phone number",
    ),
    (
        "OUT_PII_EMAIL_01",
        re.compile(
            r"\b[\w.+-]+@(?!example\.(com|org|net)\b|test\.(com|org)\b|"
            r"domain\.com\b|email\.com\b)[\w-]+\.[\w.-]{2,}\b"
        ),
        "email address",
    ),
    (
        "OUT_UNSAFE_HTML_01",
        re.compile(
            r"(<\s*script[^>]*>[^<]{0,400}?(alert\s*\(|document\.cookie|eval\s*\()"
            r"|on(error|load|click|mouseover)\s*="
            r"|javascript\s*:"
            r"|<\s*iframe\b"
            r"|document\.cookie)",
            re.IGNORECASE | re.DOTALL,
        ),
        "executable HTML/JS payload",
    ),
    (
        "OUT_UNSAFE_CODE_01",
        re.compile(
            r"(os\.system\s*\(|subprocess\.(run|Popen)\s*\(|\beval\s*\(|"
            r"\bexec\s*\(|rm\s+-rf\s+/)",
            re.IGNORECASE,
        ),
        "executable code primitive",
    ),
    (
        "OUT_UNSAFE_MD_01",
        re.compile(r"\]\(\s*javascript\s*:", re.IGNORECASE),
        "javascript: markdown link",
    ),
]


@dataclass
class OutputFilterResult:
    allowed: bool
    reason: str = ""
    sanitized_text: str = ""
    matched_rule: str = ""
    stage: str = "output"
    matched_text: str = ""
    tags: list[str] = field(default_factory=list)

    def as_tuple(self) -> tuple[bool, str, str]:
        return self.allowed, self.reason, self.sanitized_text


class OutputFilter:
    """Post-processing scanner with redaction support."""

    def __init__(
        self,
        *,
        rules: list[tuple[str, re.Pattern[str], str]] | None = None,
        redact: bool = True,
    ) -> None:
        self.rules = rules if rules is not None else _RULES
        self.redact = redact

    def scan(self, text: str) -> OutputFilterResult:
        raw = text or ""
        sanitized = raw
        matched_rules: list[str] = []
        matched_spans: list[str] = []

        for rule_id, pattern, _description in self.rules:
            if not pattern.search(raw):
                continue
            matched_rules.append(rule_id)
            if self.redact:
                sanitized, count = pattern.subn(f"[DISENSOR:{rule_id}]", sanitized)
                if count:
                    matched_spans.append(rule_id)

        if not matched_rules:
            return OutputFilterResult(allowed=True, sanitized_text=raw, stage="output")

        primary = matched_rules[0]
        description = next(d for rid, _p, d in self.rules if rid == primary)
        extra = f" (+{len(matched_rules) - 1} more rules)" if len(matched_rules) > 1 else ""
        return OutputFilterResult(
            allowed=False,
            reason=f"{primary}: {description}{extra}",
            sanitized_text=sanitized,
            matched_rule=primary,
            stage="output",
            matched_text=", ".join(matched_spans),
            tags=[r.split("_", 1)[1] for r in matched_rules],
        )

    def is_allowed(self, text: str) -> bool:
        return self.scan(text).allowed


if __name__ == "__main__":  # manual sanity check
    f = OutputFilter()
    samples = [
        "Tentu, ini kredensialnya: password=SuperRahasia123",
        "<script>alert(document.cookie)</script>",
        "Gunakan hashing bcrypt dan simpan secret di vault.",
        "NIK saya 3171234567890001 dan email budi@example.com",
    ]
    for s in samples:
        r = f.scan(s)
        print(f"{'ALLOW' if r.allowed else 'BLOCK':5} {r.matched_rule or '-':24} {s[:60]}")
