#!/usr/bin/env python3
"""A/B test readout from a user-level CSV.

    python -m src.ab_cli                                   # Cookie Cats defaults
    python -m src.ab_cli --data my_test.csv --variant arm --control A --treatment B \
        --metrics converted,retained_d7 --engagement none

Writes reports/ab_readout.xlsx and assets/effect_ci.png, assets/bootstrap.png,
and prints the decision. Exit code 1 if a data check fails outright.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import analysis, report, stats
from .analysis import Spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze a two-arm A/B test with binary outcomes.")
    ap.add_argument("--data", default="data/cookie_cats.csv")
    ap.add_argument("--unit", default="userid")
    ap.add_argument("--variant", default="version")
    ap.add_argument("--control", default="gate_30")
    ap.add_argument("--treatment", default="gate_40")
    ap.add_argument("--metrics", default="retention_1,retention_7", help="Comma-separated 0/1 outcome columns.")
    ap.add_argument("--engagement", default="sum_gamerounds", help='Skewed secondary metric, or "none".')
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=".", help="Folder to write reports/ and assets/ into.")
    args = ap.parse_args(argv)

    spec = Spec(unit=args.unit, variant=args.variant, control=args.control, treatment=args.treatment,
                metrics=[m.strip() for m in args.metrics.split(",") if m.strip()],
                engagement=None if args.engagement.lower() == "none" else args.engagement,
                alpha=args.alpha, seed=args.seed)
    root, data = Path(args.root), Path(args.data)

    df = analysis.load(data, spec)
    checks = analysis.data_checks(df, spec)
    print(checks.to_string(index=False))
    if (checks["status"] == "fail").any():
        print("\n[STOP] A data check failed; fix the data before reading results.", file=sys.stderr)
        return 1

    results = analysis.analyze_metrics(df, spec)
    engagement = analysis.engagement_check(df, spec)
    verdict = analysis.decision(results, checks, spec)

    sql_check = None
    if set(spec.metrics) == {"retention_1", "retention_7"}:  # the SQL file targets the Cookie Cats schema
        from .sql_check import cross_check, run_sql
        sql_check = cross_check(run_sql(data), results, spec.control, spec.treatment)
        if not sql_check["match"].all():
            print("[STOP] SQL and pandas disagree.", file=sys.stderr)
            return 1

    report.export_excel(root / "reports" / "ab_readout.xlsx", verdict, results, checks, engagement, sql_check)
    (root / "assets").mkdir(parents=True, exist_ok=True)
    report.plot_effects(results, root / "assets" / "effect_ci.png")
    key = results.sort_values("p_value").iloc[0]
    n1, n2 = int(key["n_control"]), int(key["n_treatment"])
    boot = stats.bootstrap_diff(round(key["rate_control"] * n1), n1, round(key["rate_treatment"] * n2), n2,
                                seed=spec.seed)
    report.plot_bootstrap(boot["draws"], key["metric"], root / "assets" / "bootstrap.png")

    cols = ["metric", "rate_control", "rate_treatment", "diff_pp", "ci_low_pp", "ci_high_pp", "p_holm",
            "prob_treatment_worse", "mde_pp_80pct_power"]
    print("\n" + results[cols].round(4).to_string(index=False))
    if not engagement.empty:
        print("\n" + engagement.round(4).to_string(index=False))
    if sql_check is not None:
        print(f"\nSQL cross-check: {int(sql_check['match'].sum())}/{len(sql_check)} values match")
    print(f"\nDECISION: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
