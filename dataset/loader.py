#!/usr/bin/env python3
"""Loader and schema validator for the OWASP 2026 Indonesian test corpus.

The corpus is a plain CSV so it stays diff-friendly and reviewable in a thesis
appendix. Multi-turn cases keep every user turn in one ``payload`` cell,
separated by ``||`` (see :data:`TURN_SEPARATOR`).

Usage:
    python -m dataset.loader                       # validate the default corpus
    python -m dataset.loader dataset/other.csv     # validate another file
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TURN_SEPARATOR = "||"

# Exact column order required by the methodology.
COLUMNS = [
    "test_id",
    "owasp_category",
    "scenario_type",
    "flow",
    "data_type",
    "payload",
    "expected_behavior",
    "target_keywords",
]

CATEGORIES = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Excessive Agency",
    "LLM04": "Supply Chain",
    "LLM05": "Data and Model Poisoning",
    "LLM06": "Unbounded Consumption",
    "LLM07": "Misinformation",
    "LLM08": "Hidden Context Exposure",
    "LLM09": "Vector and Embedding Weaknesses",
    "LLM10": "Improper Output Handling",
}

ALLOWED_FLOWS = {"Single-Turn", "Multi-Turn"}
ALLOWED_DATA_TYPES = {"contaminated", "clean"}

DEFAULT_DATASET = Path(__file__).resolve().with_name("owasp_2026_id.csv")


class DatasetError(ValueError):
    """Raised when the corpus violates the expected schema."""


@dataclass(frozen=True)
class TestCase:
    test_id: str
    owasp_category: str
    scenario_type: str
    flow: str
    data_type: str
    payload: str
    expected_behavior: str
    target_keywords: list[str]
    line_number: int = 0

    @property
    def turns(self) -> list[str]:
        if self.flow == "Single-Turn":
            return [self.payload.strip()]
        return [t.strip() for t in self.payload.split(TURN_SEPARATOR) if t.strip()]

    @property
    def is_multi_turn(self) -> bool:
        return self.flow == "Multi-Turn"

    @property
    def is_contaminated(self) -> bool:
        return self.data_type == "contaminated"

    @property
    def category_code(self) -> str:
        return self.owasp_category.split(":", 1)[0].strip()

    def to_dict(self) -> dict[str, object]:
        return {
            "test_id": self.test_id,
            "owasp_category": self.owasp_category,
            "scenario_type": self.scenario_type,
            "flow": self.flow,
            "data_type": self.data_type,
            "payload": self.payload,
            "expected_behavior": self.expected_behavior,
            "target_keywords": "|".join(self.target_keywords),
        }


def _parse_keywords(raw: str) -> list[str]:
    return [part.strip() for part in raw.split("|") if part.strip()]


def _validate_row(row: dict[str, str], line: int) -> TestCase:
    where = f"line {line} ({row.get('test_id') or 'no test_id'})"

    if not row.get("test_id", "").strip():
        raise DatasetError(f"{where}: test_id is empty")
    if not row.get("scenario_type", "").strip():
        raise DatasetError(f"{where}: scenario_type is empty")
    if not row.get("expected_behavior", "").strip():
        raise DatasetError(f"{where}: expected_behavior is empty")

    category = row.get("owasp_category", "").strip()
    code = category.split(":", 1)[0].strip()
    if code not in CATEGORIES:
        raise DatasetError(f"{where}: unknown OWASP category {category!r}")
    if category != f"{code}: {CATEGORIES[code]}":
        raise DatasetError(
            f"{where}: category must be formatted as "
            f"{code!r}: {CATEGORIES[code]!r}, got {category!r}"
        )

    flow = row.get("flow", "").strip()
    if flow not in ALLOWED_FLOWS:
        raise DatasetError(f"{where}: flow must be one of {sorted(ALLOWED_FLOWS)}")

    data_type = row.get("data_type", "").strip()
    if data_type not in ALLOWED_DATA_TYPES:
        raise DatasetError(f"{where}: data_type must be one of {sorted(ALLOWED_DATA_TYPES)}")

    payload = row.get("payload", "")
    if not payload.strip():
        raise DatasetError(f"{where}: payload is empty")

    keywords = _parse_keywords(row.get("target_keywords", ""))
    if data_type == "contaminated" and not keywords:
        raise DatasetError(f"{where}: contaminated cases need target_keywords")

    case = TestCase(
        test_id=row["test_id"].strip(),
        owasp_category=category,
        scenario_type=row["scenario_type"].strip(),
        flow=flow,
        data_type=data_type,
        payload=payload.strip(),
        expected_behavior=row["expected_behavior"].strip(),
        target_keywords=keywords,
        line_number=line,
    )

    turns = case.turns
    if flow == "Single-Turn" and len(turns) != 1:
        raise DatasetError(f"{where}: Single-Turn payload must not contain {TURN_SEPARATOR!r}")
    if flow == "Multi-Turn" and len(turns) < 2:
        raise DatasetError(f"{where}: Multi-Turn payload needs at least two turns")
    if any(TURN_SEPARATOR in t for t in turns):
        raise DatasetError(f"{where}: malformed turn separator")

    return case


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[TestCase]:
    """Read, validate and return all test cases in corpus order."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")

    cases: list[TestCase] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DatasetError(f"{path}: empty file")
        missing = [c for c in COLUMNS if c not in reader.fieldnames]
        extra = [c for c in reader.fieldnames if c not in COLUMNS]
        if missing:
            raise DatasetError(f"{path}: missing columns {missing}")
        if extra:
            raise DatasetError(f"{path}: unexpected columns {extra}")

        for line, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue  # tolerate blank lines
            case = _validate_row({k: (v or "") for k, v in row.items()}, line)
            if case.test_id in seen:
                raise DatasetError(f"line {line}: duplicate test_id {case.test_id!r}")
            seen.add(case.test_id)
            cases.append(case)

    if not cases:
        raise DatasetError(f"{path}: no test cases found")
    return cases


def describe(cases: list[TestCase]) -> str:
    per_category = Counter(c.category_code for c in cases)
    per_flow = Counter(c.flow for c in cases)
    per_type = Counter(c.data_type for c in cases)
    lines = [
        f"cases={len(cases)}",
        "  flow: " + ", ".join(f"{k}={v}" for k, v in sorted(per_flow.items())),
        "  type: " + ", ".join(f"{k}={v}" for k, v in sorted(per_type.items())),
        "  categories:",
    ]
    for code in CATEGORIES:
        lines.append(f"    {code}: {per_category.get(code, 0)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(DEFAULT_DATASET))
    args = parser.parse_args(argv)
    try:
        cases = load_dataset(args.path)
    except DatasetError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK {args.path}")
    print(describe(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
