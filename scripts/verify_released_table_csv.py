#!/usr/bin/env python3
"""Verify the released table CSV artifacts against the paper numbers.

This is a lightweight consistency check, not the experiment recomputation.
For recomputing metrics from raw controller rollouts and ScanNet++ meshes,
use scripts/recompute_paper_tables_from_rollouts.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

METHOD_ORDER = ["SaferSplat", "Ours (ESDF)", "Ours (Semantic ESDF)"]
METHOD_MAP = {
    "3DGS": "SaferSplat",
    "Ours ESDF": "Ours (ESDF)",
    "Ours ESDF semantic-aware": "Ours (Semantic ESDF)",
}
RISK_ORDER = ["high", "mid", "low"]
RISK_LABEL = {"high": "High", "mid": "Mid", "low": "Low"}

EXPECTED_MAIN = {
    "SaferSplat": ("0.149", "0.360", "0.101", "0.216"),
    "Ours (ESDF)": ("0.002", "0.697", "0.106", "0.218"),
    "Ours (Semantic ESDF)": ("0.002", "0.603", "0.108", "0.200"),
}
EXPECTED_RISK = {
    ("high", "Ours (ESDF)"): ("0.669", "0.128", "0.146"),
    ("mid", "Ours (ESDF)"): ("0.777", "0.092", "0.248"),
    ("low", "Ours (ESDF)"): ("0.647", "0.098", "0.261"),
    ("high", "Ours (Semantic ESDF)"): ("0.534", "0.134", "0.116"),
    ("mid", "Ours (Semantic ESDF)"): ("0.631", "0.093", "0.224"),
    ("low", "Ours (Semantic ESDF)"): ("0.647", "0.098", "0.261"),
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(x: float) -> str:
    return f"{x:.3f}"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def latex_table_i(rows: Sequence[Sequence[str]]) -> str:
    body = "\n".join(
        f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\" for r in rows
    )
    return rf"""\begin{{table*}}[t]
\centering
\caption{{\small Equal-time comparison on the six-scene benchmark. MP denotes matched progress.}}
\label{{tab:sim_3methods}}
\begin{{tabular}}{{lcccc}}
\toprule
Method & Computation Time (s) $\downarrow$ & Progress $\uparrow$ & MP Clearance (m) $\uparrow$ & MP Collision $\downarrow$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def latex_table_iii(rows: Sequence[Sequence[str]]) -> str:
    body_lines = []
    for risk, n, e_prog, e_clear, e_col, s_prog, s_clear, s_col in rows:
        body_lines.append(
            f"{risk} & {n} & {e_prog} & {e_clear} & {e_col} & {s_prog} & {s_clear} & {s_col} \\\\"
        )
    return rf"""\begin{{table*}}[t]
\centering
\caption{{\small Risk-group ablation on the six-scene benchmark. MP denotes matched progress.}}
\label{{tab:risk_ablation}}
\begin{{tabular}}{{lccccccc}}
\toprule
Risk & N & \multicolumn{{3}}{{c}}{{Ours (ESDF)}} & \multicolumn{{3}}{{c}}{{Ours (Semantic ESDF)}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}}
Group & & Progress $\uparrow$ & MP Clear. $\uparrow$ & MP Coll. $\downarrow$ & Progress $\uparrow$ & MP Clear. $\uparrow$ & MP Coll. $\downarrow$ \\
\midrule
{chr(10).join(body_lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def build_table_i(table_dir: Path) -> List[Tuple[str, str, str, str, str]]:
    timing_rows = read_csv(table_dir / "success6_3methods_main_table.csv")
    exposure_rows = read_csv(table_dir / "success6_3methods_exposure_main_table.csv")

    timing_by_method = {
        METHOD_MAP.get(row["method"], row["method"]): row for row in timing_rows
    }
    exposure_by_method = {row["method"]: row for row in exposure_rows}

    rows: List[Tuple[str, str, str, str, str]] = []
    for method in METHOD_ORDER:
        timing = timing_by_method[method]
        exposure = exposure_by_method[method]
        rows.append(
            (
                method,
                fmt(as_float(timing, "computation_time_mean_s")),
                fmt(as_float(exposure, "progress_mean")),
                fmt(as_float(exposure, "matched_progress_min_clearance_mean_m")),
                fmt(as_float(exposure, "matched_progress_collision_rate")),
            )
        )
    return rows


def build_table_iii(table_dir: Path) -> List[Tuple[str, str, str, str, str, str, str, str]]:
    rows = read_csv(table_dir / "success6_matchbalanced_subtable_matched_progress.csv")
    by_key = {(row["risk"], row["method"]): row for row in rows}

    out: List[Tuple[str, str, str, str, str, str, str, str]] = []
    for risk in RISK_ORDER:
        esdf = by_key[(risk, "Ours (ESDF)")]
        sem = by_key[(risk, "Ours (Semantic ESDF)")]
        out.append(
            (
                RISK_LABEL[risk],
                esdf["n"],
                fmt(as_float(esdf, "progress_to_goal_mean")),
                fmt(as_float(esdf, "matched_progress_min_clearance_mean_m")),
                fmt(as_float(esdf, "matched_progress_collision_rate")),
                fmt(as_float(sem, "progress_to_goal_mean")),
                fmt(as_float(sem, "matched_progress_min_clearance_mean_m")),
                fmt(as_float(sem, "matched_progress_collision_rate")),
            )
        )
    return out


def validate(table_i: Sequence[Sequence[str]], table_iii: Sequence[Sequence[str]]) -> None:
    errors: List[str] = []
    for row in table_i:
        method = row[0]
        actual = tuple(row[1:5])
        if actual != EXPECTED_MAIN[method]:
            errors.append(f"Table I mismatch for {method}: {actual} != {EXPECTED_MAIN[method]}")

    risk_lookup = {row[0].lower(): row for row in table_iii}
    for risk in RISK_ORDER:
        row = risk_lookup[risk]
        esdf = tuple(row[2:5])
        sem = tuple(row[5:8])
        if esdf != EXPECTED_RISK[(risk, "Ours (ESDF)")]:
            errors.append(f"Table III mismatch for {risk}/ESDF: {esdf}")
        if sem != EXPECTED_RISK[(risk, "Ours (Semantic ESDF)")]:
            errors.append(f"Table III mismatch for {risk}/Semantic ESDF: {sem}")

    if errors:
        raise SystemExit("\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("repro/table_metrics"),
        help="Directory containing the released six-scene metric CSV files.",
    )
    parser.add_argument(
        "--ours-only",
        action="store_true",
        help="Print only the SafeDF ESDF and semantic-aware ESDF rows.",
    )
    parser.add_argument("--save-latex", type=Path, default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    table_i = build_table_i(args.table_dir)
    table_iii = build_table_iii(args.table_dir)
    if args.ours_only:
        table_i = [row for row in table_i if row[0] != "SaferSplat"]

    print("Table I: released CSV check")
    print(
        markdown_table(
            ["Method", "Time (s) ↓", "Progress ↑", "MP Clearance (m) ↑", "MP Collision ↓"],
            table_i,
        )
    )
    print()
    print("Table III: released CSV check")
    print(
        markdown_table(
            [
                "Risk",
                "N",
                "ESDF Progress ↑",
                "ESDF MP Clear. ↑",
                "ESDF MP Coll. ↓",
                "Sem Progress ↑",
                "Sem MP Clear. ↑",
                "Sem MP Coll. ↓",
            ],
            table_iii,
        )
    )

    if not args.no_validate:
        validate(build_table_i(args.table_dir), table_iii)
        print("\nValidation: released CSV rounded values match the paper tables.")

    if args.save_latex is not None:
        latex = latex_table_i(table_i) + "\n\n" + latex_table_iii(table_iii) + "\n"
        args.save_latex.parent.mkdir(parents=True, exist_ok=True)
        args.save_latex.write_text(latex)
        print(f"Saved LaTeX: {args.save_latex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
