#!/usr/bin/env python3
"""Before/After evaluation for the thesis Chapter 3 metrics.

Reads ``reports/baseline.csv`` and ``reports/mitigated.csv``, then produces:

* per-category and overall ASR / mean ASV, baseline vs mitigated
* reduction deltas (percentage points for ASR, absolute for ASV)
* confusion-matrix counts (TP / FP / TN / FN)
* FPR = FP / (FP + TN) and FNR = FN / (FN + TP)

Outputs ``reports/summary.csv`` and ``reports/summary.md`` and prints the
Markdown table.

Usage:
    python metrics/evaluate.py
    python metrics/evaluate.py --baseline reports/baseline.csv \
        --mitigated reports/mitigated.csv --out reports
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
    "test_id", "phase", "category", "data_type", "guardrail_blocked",
    "exploit_success", "asv_score", "confusion_class",
}
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


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

    for column in ("phase", "category", "data_type", "confusion_class"):
        df[column] = df[column].astype(str).str.strip()
    df["data_type"] = df["data_type"].str.casefold()
    df["confusion_class"] = df["confusion_class"].str.upper()
    df["guardrail_blocked"] = _as_bool(df["guardrail_blocked"])
    df["exploit_success"] = _as_bool(df["exploit_success"])
    df["asv_score"] = pd.to_numeric(df["asv_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df


def _rate(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else float("nan")


def compute_scope(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute metrics for one subset (overall or a single category)."""
    contaminated = df[df["data_type"] == "contaminated"]
    clean = df[df["data_type"] == "clean"]

    n_contaminated = len(contaminated)
    asr = 100.0 * contaminated["exploit_success"].mean() if n_contaminated else float("nan")
    asv = contaminated["asv_score"].mean() if n_contaminated else float("nan")

    tp = int((df["confusion_class"] == "TP").sum())
    fp = int((df["confusion_class"] == "FP").sum())
    tn = int((df["confusion_class"] == "TN").sum())
    fn = int((df["confusion_class"] == "FN").sum())

    fpr = _rate(fp, fp + tn)
    fnr = _rate(fn, fn + tp)

    return {
        "n": n_contaminated,
        "n_clean": len(clean),
        "asr": asr,
        "asv": asv,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fpr,
        "fnr": fnr,
    }


def summarize(baseline: pd.DataFrame, mitigated: pd.DataFrame | None) -> pd.DataFrame:
    """Build the comparison table, one row per scope (overall first)."""
    scopes: list[tuple[str, str]] = [("Overall", "__ALL__")]
    present = set(baseline["category"]) | (set(mitigated["category"]) if mitigated is not None else set())
    for code, title in CATEGORIES.items():
        label = f"{code}: {title}"
        if label in present:
            scopes.append((code, label))

    rows: list[dict[str, object]] = []
    for scope, selector in scopes:
        base_df = baseline if selector == "__ALL__" else baseline[baseline["category"] == selector]
        base = compute_scope(base_df)

        if mitigated is not None:
            mit_df = mitigated if selector == "__ALL__" else mitigated[mitigated["category"] == selector]
            mit = compute_scope(mit_df)
        else:
            mit = {k: float("nan") for k in ("asr", "asv", "fpr", "fnr")}
            mit.update({"n": 0, "n_clean": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})

        asr_delta = (
            base["asr"] - mit["asr"]
            if pd.notna(base["asr"]) and pd.notna(mit["asr"])
            else float("nan")
        )
        asv_delta = (
            base["asv"] - mit["asv"]
            if pd.notna(base["asv"]) and pd.notna(mit["asv"])
            else float("nan")
        )

        rows.append({
            "scope": scope,
            "baseline_n": base["n"],
            "baseline_asr_pct": base["asr"],
            "baseline_asv": base["asv"],
            "mitigated_n": mit["n"],
            "mitigated_asr_pct": mit["asr"],
            "mitigated_asv": mit["asv"],
            "asr_reduction_pp": asr_delta,
            "asv_reduction": asv_delta,
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
    if isinstance(value, (int,)):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_markdown(table: pd.DataFrame, *, baseline_path: str, mitigated_path: str | None) -> str:
    header = (
        "| Scope | Base ASR % | Base ASV | Mitig. ASR % | Mitig. ASV | ΔASR (pp) | ΔASV "
        "| TP | FP | TN | FN | FPR % | FNR % |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [
        "# Before / After Security Evaluation",
        "",
        f"- Baseline report: `{baseline_path}`",
        f"- Mitigated report: `{mitigated_path or '—'}`",
        "",
        "Positive Δ means improvement (baseline minus mitigated).",
        "",
        header,
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {scope} | {b_asr} | {b_asv} | {m_asr} | {m_asv} | {d_asr} | {d_asv} "
            "| {tp} | {fp} | {tn} | {fn} | {fpr} | {fnr} |".format(
                scope=row["scope"],
                b_asr=_fmt(row["baseline_asr_pct"]),
                b_asv=_fmt(row["baseline_asv"], 3),
                m_asr=_fmt(row["mitigated_asr_pct"]),
                m_asv=_fmt(row["mitigated_asv"], 3),
                d_asr=_fmt(row["asr_reduction_pp"]),
                d_asv=_fmt(row["asv_reduction"], 3),
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
