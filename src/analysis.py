"""Load an experiment export, check it, and analyze every metric.

Default columns match the public Cookie Cats test; any user-level export with
one row per user, a variant column, and 0/1 outcome columns works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
from scipy import stats as sps

from . import stats


@dataclass
class Spec:
    unit: str = "userid"
    variant: str = "version"
    control: str = "gate_30"
    treatment: str = "gate_40"
    metrics: List[str] = field(default_factory=lambda: ["retention_1", "retention_7"])
    engagement: str | None = "sum_gamerounds"
    alpha: float = 0.05
    expected_share: float = 0.5
    seed: int = 42


def load(path: Path, spec: Spec) -> pd.DataFrame:
    df = pd.read_csv(path)
    for m in spec.metrics:
        df[m] = df[m].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).astype("Int8")
    if spec.engagement:
        df[spec.engagement] = pd.to_numeric(df[spec.engagement], errors="coerce")
    return df


def data_checks(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Checks to run before reading any result.

    status: ok; note (worth knowing, does not affect the decision); warn (caveat
    on the decision); fail (stop: results would be meaningless).
    """
    rows = []

    def add(check, value, status, note):
        rows.append({"check": check, "value": value, "status": status, "note": note})

    variants = set(df[spec.variant].dropna().unique())
    expected = {spec.control, spec.treatment}
    add("variants", ", ".join(sorted(map(str, variants))), "ok" if variants == expected else "fail",
        f"expected exactly {spec.control} and {spec.treatment}")
    dups = int(df[spec.unit].duplicated().sum())
    add("duplicate users", dups, "ok" if dups == 0 else "fail", "each user must appear once")
    missing = int(df[spec.metrics].isna().sum().sum())
    add("missing outcomes", missing, "ok" if missing == 0 else "warn", "rows with a blank metric")

    n_c = int((df[spec.variant] == spec.control).sum())
    n_t = int((df[spec.variant] == spec.treatment).sum())
    srm = stats.srm_check(n_c, n_t, spec.expected_share)
    add("sample ratio (SRM)", f"{n_c:,} vs {n_t:,} ({srm['treatment_share']:.2%} treatment), p={srm['p_value']:.4f}",
        "ok" if srm["p_value"] >= 0.01 else "warn",
        "p < 0.01: the split differs from plan more than chance allows; check assignment and logging")

    if spec.engagement:
        e = df[spec.engagement]
        q999 = e.quantile(0.999)
        extreme = df.loc[e > 10 * q999, [spec.unit, spec.variant, spec.engagement]]
        # "note", not "warn": it only affects the engagement metric, never the retention decision.
        add("extreme engagement values", len(extreme), "ok" if extreme.empty else "note",
            f"> 10x the 99.9th percentile ({q999:,.0f}); max = {e.max():,.0f}")
        zero = float((e == 0).mean())
        add("users with zero engagement", f"{zero:.1%}", "ok",
            "never played, so never reached the changed gate; kept in (intent-to-treat)")
    return pd.DataFrame(rows)


def analyze_metrics(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    rows = []
    for m in spec.metrics:
        c = df.loc[df[spec.variant] == spec.control, m].dropna()
        t = df.loc[df[spec.variant] == spec.treatment, m].dropna()
        x1, n1, x2, n2 = int(c.sum()), len(c), int(t.sum()), len(t)
        p1, p2 = x1 / n1, x2 / n2
        z = stats.two_prop_ztest(x1, n1, x2, n2)
        lo, hi = stats.diff_ci(x1, n1, x2, n2, spec.alpha)
        boot = stats.bootstrap_diff(x1, n1, x2, n2, seed=spec.seed)
        rows.append({
            "metric": m, "n_control": n1, "n_treatment": n2,
            "rate_control": p1, "rate_treatment": p2,
            "diff_pp": (p2 - p1) * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "rel_lift_pct": (p2 / p1 - 1) * 100,
            "z": z["z"], "p_value": z["p_value"],
            "cohen_h": stats.cohen_h(p1, p2),
            "prob_treatment_worse": boot["prob_treatment_worse"],
            "mde_pp_80pct_power": stats.mde_two_prop(p1, n1, n2, spec.alpha) * 100,
            "power_at_observed": stats.power_two_prop(p1, p2, n1, n2, spec.alpha),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = stats.holm(out["p_value"].tolist())
    out["significant"] = out["p_holm"] < spec.alpha
    return out


def engagement_check(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Secondary metric: rounds played. Heavily skewed, so compare medians (Mann-Whitney U)."""
    if not spec.engagement:
        return pd.DataFrame()
    c = df.loc[df[spec.variant] == spec.control, spec.engagement].dropna()
    t = df.loc[df[spec.variant] == spec.treatment, spec.engagement].dropna()
    u = sps.mannwhitneyu(t, c, alternative="two-sided")
    return pd.DataFrame([{
        "metric": spec.engagement, "median_control": c.median(), "median_treatment": t.median(),
        "mean_control": c.mean(), "mean_treatment": t.mean(),
        "mann_whitney_p": u.pvalue,
    }])


def decision(results: pd.DataFrame, checks: pd.DataFrame, spec: Spec) -> str:
    worse = results[results["significant"] & (results["diff_pp"] < 0)]
    better = results[results["significant"] & (results["diff_pp"] > 0)]
    warn = checks[checks["status"] == "warn"]["check"].tolist()
    if not worse.empty:
        r = worse.iloc[0]
        text = (f"Keep {spec.control}. {spec.treatment} lowered {r['metric']} by {abs(r['diff_pp']):.2f} pp "
                f"(95% CI {r['ci_low_pp']:.2f} to {r['ci_high_pp']:.2f}; Holm-adjusted p = {r['p_holm']:.4f}).")
    elif not better.empty:
        r = better.iloc[0]
        text = (f"{spec.treatment} improved {r['metric']} by {r['diff_pp']:.2f} pp "
                f"(95% CI {r['ci_low_pp']:.2f} to {r['ci_high_pp']:.2f}); consider rollout.")
    else:
        text = "No significant difference on any metric; the test cannot tell the variants apart at this sample size."
    if warn:
        text += f" Caveat: {', '.join(warn)} flagged; confirm before acting."
    return text
