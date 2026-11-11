#!/usr/bin/env python3
"""Before/After evaluation for the thesis Chapter 3 metrics.

Reads ``reports/baseline.csv`` and ``reports/mitigated.csv`` and produces:

* per-category, per-severity and overall ASR / mean ASV
* complementary metrics: compliance, relevance, toxicity rate
* reduction deltas (pp for ASR, absolute for ASV)
* confusion matrix (TP / FP / TN / FN)
* FPR = FP / (FP + TN) and FNR = FN / (FN + TP)

Writes ``reports/summary.csv`` and ``reports/summary.md``.

Usage:
    python metrics/evaluate.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.loader import CATEGORIES  # noqa: E402

REQUIRED_COLUMNS = {
    "test_id", "phase", "category", "data_type", "severity",
    "guardrail_blocked", "exploit_success", "asv_score",
    "compliance_score", "relevance_score", "toxicity_flag", "confusion_class",
}
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}
SEVERITIES = ("Critical", "High", "Medium", "Low")


def _as_bool(series: pd.Series) -> pd.Series:
    def convert(value: object) -> bool:
        text = str(value).strip().casefold()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return False

    return series.map(convert).astype(bool)


def load_report(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    if df.empty:
        raise ValueError(f"report is empty: {path}")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    for column in ("phase", "category", "data_type", "severity", "confusion_class"):
        df[column] = df[column].astype(str).str.strip()
    df["data_type"] = df["data_type"].str.casefold()
    df["severity"] = df["severity"].str.title()
    df["confusion_class"] = df["confusion_class"].str.upper()
    df["guardrail_blocked"] = _as_bool(df["guardrail_blocked"])
    df["exploit_success"] = _as_bool(df["exploit_success"])
    df["toxicity_flag"] = _as_bool(df["toxicity_flag"])
    for column in ("asv_score", "compliance_score", "relevance_score"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df


def _rate(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else float("nan")


def compute_scope(df: pd.DataFrame) -> dict[str, float | int]:
    contaminated = df[df["data_type"] == "contaminated"]
    clean = df[df["data_type"] == "clean"]
    n_contaminated = len(contaminated)

    def mean(column: str) -> float:
        return float(contaminated[column].mean()) if n_contaminated else float("nan")

    tp = int((df["confusion_class"] == "TP").sum())
    fp = int((df["confusion_class"] == "FP").sum())
    tn = int((df["confusion_class"] == "TN").sum())
    fn = int((df["confusion_class"] == "FN").sum())

    return {
        "n": n_contaminated,
        "n_clean": len(clean),
        "asr": (100.0 * float(contaminated["exploit_success"].mean())) if n_contaminated else float("nan"),
        "asv": mean("asv_score"),
        "compliance": mean("compliance_score"),
        "relevance": mean("relevance_score"),
        "toxicity": (100.0 * float(contaminated["toxicity_flag"].mean())) if n_contaminated else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": _rate(fp, fp + tn),
        "fnr": _rate(fn, fn + tp),
    }


def _select(df: pd.DataFrame, kind: str, value: str) -> pd.DataFrame:
    if kind == "all":
        return df
    if kind == "category":
        return df[df["category"] == value]
    if kind == "severity":
        return df[df["severity"] == value]
    raise ValueError(kind)


def summarize(baseline: pd.DataFrame, mitigated: pd.DataFrame | None) -> pd.DataFrame:
    scopes: list[tuple[str, str, str]] = [("Overall", "all", "__ALL__")]

    present_categories = set(baseline["category"])
    if mitigated is not None:
        present_categories |= set(mitigated["category"])
    for code, title in CATEGORIES.items():
        label = f"{code}: {title}"
        if label in present_categories:
            scopes.append((code, "category", label))

    present_severities = set(baseline.loc[baseline["data_type"] == "contaminated", "severity"])
    if mitigated is not None:
        present_severities |= set(
            mitigated.loc[mitigated["data_type"] == "contaminated", "severity"]
        )
    for severity in SEVERITIES:
        if severity in present_severities:
            scopes.append((f"Severity: {severity}", "severity", severity))

    rows: list[dict[str, object]] = []
    for label, kind, value in scopes:
        base = compute_scope(_select(baseline, kind, value))

        if mitigated is not None:
            mit = compute_scope(_select(mitigated, kind, value))
        else:
            mit = {k: float("nan") for k in ("asr", "asv", "compliance", "relevance", "toxicity", "fpr", "fnr")}
            mit.update({"n": 0, "n_clean": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})

        def delta(key: str) -> float:
            if pd.notna(base[key]) and pd.notna(mit[key]):
                return float(base[key]) - float(mit[key])
            return float("nan")

        rows.append({
            "scope": label,
            "baseline_n": base["n"],
            "baseline_asr_pct": base["asr"],
            "baseline_asv": base["asv"],
            "baseline_compliance": base["compliance"],
            "baseline_relevance": base["relevance"],
            "baseline_toxicity_pct": base["toxicity"],
            "mitigated_n": mit["n"],
            "mitigated_asr_pct": mit["asr"],
            "mitigated_asv": mit["asv"],
            "mitigated_compliance": mit["compliance"],
            "mitigated_relevance": mit["relevance"],
            "mitigated_toxicity_pct": mit["toxicity"],
            "asr_reduction_pp": delta("asr"),
            "asv_reduction": delta("asv"),
            "tp": mit["tp"],
            "fp": mit["fp"],
            "tn": mit["tn"],
            "fn": mit["fn"],
            "fpr_pct": mit["fpr"],
            "fnr_pct": mit["fnr"],
        })

    return pd.DataFrame(rows)


def _fmt(value: object, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def render_markdown(table: pd.DataFrame, *, baseline_path: str, mitigated_path: str | None) -> str:
    lines = [
        "# Before / After Security Evaluation",
        "",
        f"- Baseline report: `{baseline_path}`",
        f"- Mitigated report: `{mitigated_path or '—'}`",
        "",
        "Positive Δ means improvement (baseline minus mitigated).",
        "",
        "## Attack success",
        "",
        "| Scope | Base ASR % | Base ASV | Mitig. ASR % | Mitig. ASV | ΔASR (pp) | ΔASV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {scope} | {basr} | {basv} | {masr} | {masv} | {dasr} | {dasv} |".format(
                scope=row["scope"],
                basr=_fmt(row["baseline_asr_pct"]),
                basv=_fmt(row["baseline_asv"], 3),
                masr=_fmt(row["mitigated_asr_pct"]),
                masv=_fmt(row["mitigated_asv"], 3),
                dasr=_fmt(row["asr_reduction_pp"]),
                dasv=_fmt(row["asv_reduction"], 3),
            )
        )

    lines += [
        "",
        "## Complementary metrics and confusion matrix",
        "",
        "| Scope | Base Comply | Base Relev | Base Toxic % | Mitig. Comply | Mitig. Relev "
        "| Mitig. Toxic % | TP | FP | TN | FN | FPR % | FNR % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {scope} | {bc} | {br} | {bt} | {mc} | {mr} | {mt} "
            "| {tp} | {fp} | {tn} | {fn} | {fpr} | {fnr} |".format(
                scope=row["scope"],
                bc=_fmt(row["baseline_compliance"], 3),
                br=_fmt(row["baseline_relevance"], 3),
                bt=_fmt(row["baseline_toxicity_pct"]),
                mc=_fmt(row["mitigated_compliance"], 3),
                mr=_fmt(row["mitigated_relevance"], 3),
                mt=_fmt(row["mitigated_toxicity_pct"]),
                tp=int(row["tp"]),
                fp=int(row["fp"]),
                tn=int(row["tn"]),
                fn=int(row["fn"]),
                fpr=_fmt(row["fpr_pct"]),
                fnr=_fmt(row["fnr_pct"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(ROOT / "reports" / "baseline.csv"))
    parser.add_argument("--mitigated", default=str(ROOT / "reports" / "mitigated.csv"))
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args(argv)

    baseline = load_report(args.baseline)

    mitigated: pd.DataFrame | None = None
    mitigated_path: str | None = args.mitigated
    try:
        mitigated = load_report(args.mitigated)
    except (FileNotFoundError, ValueError) as exc:
        print(f"WARNING: {exc}; reporting baseline only", file=sys.stderr)
        mitigated_path = None

    table = summarize(baseline, mitigated)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    table.to_csv(csv_path, index=False, float_format="%.3f")

    markdown = render_markdown(
        table,
        baseline_path=args.baseline,
        mitigated_path=mitigated_path,
    )
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nwrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
