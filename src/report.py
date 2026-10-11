"""Stakeholder outputs: an Excel readout and two charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"retention_1": "Day-1 retention", "retention_7": "Day-7 retention"}


def _autosize(ws, df: pd.DataFrame) -> None:
    for j, col in enumerate(df.columns):
        width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
        ws.set_column(j, j, min(max(width + 2, 10), 90))


def export_excel(path: Path, decision: str, results, checks, engagement, sql_check) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readout = pd.DataFrame({"item": ["Decision", "How to read", "Data checks"],
                            "detail": [decision,
                                       "diff_pp = treatment minus control, in percentage points. "
                                       "p_holm adjusts for testing two metrics. prob_treatment_worse is from 20,000 bootstrap draws.",
                                       "See Checks: SRM and outliers are flagged before any result is trusted."]})
    sheets = {"Readout": readout, "Results": results.round(5), "Checks": checks,
              "Engagement": engagement.round(4), "SQL_crosscheck": sql_check}
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if df is None or df.empty:
                df = pd.DataFrame({"note": ["not run"]})
            df.to_excel(writer, index=False, sheet_name=name)
            _autosize(writer.sheets[name], df)


def plot_effects(results: pd.DataFrame, path: Path) -> None:
    """Each metric's difference with its 95% CI; the dashed line is 'no effect'."""
    r = results.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.9 * len(r)))
    y = range(len(r))
    colors = ["#c0392b" if s and d < 0 else "#27ae60" if s else "#7f8c8d"
              for s, d in zip(r["significant"], r["diff_pp"])]
    for yi, (d, lo, hi, c) in enumerate(zip(r["diff_pp"], r["ci_low_pp"], r["ci_high_pp"], colors)):
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ms=8, elinewidth=3, capsize=6)
    ax.axvline(0, color="#333", ls="--", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([LABELS.get(m, m) for m in r["metric"]])
    for yi, (d, p) in enumerate(zip(r["diff_pp"], r["p_holm"])):
        ax.annotate(f"{d:+.2f} pp  (p={p:.3f})", (d, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9)
    ax.set_xlabel("Treatment minus control (percentage points), 95% CI")
    ax.set_title("Moving the gate to level 40: effect on retention")
    ax.set_ylim(-0.6, len(r) - 0.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_bootstrap(draws, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(draws * 100, bins=60, color="#2b6cb0", alpha=.85)
    ax.axvline(0, color="#333", ls="--", lw=1)
    share = (draws < 0).mean()
    ax.set_title(f"{LABELS.get(metric, metric)}: 20,000 bootstrap draws, {share:.1%} below zero")
    ax.set_xlabel("Treatment minus control (percentage points)")
    ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
