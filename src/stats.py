"""Statistics for a two-arm A/B test on binary outcomes.

Small, dependency-light functions with textbook formulas, each checked
against statsmodels/scipy in tests/test_stats.py.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def two_prop_ztest(x1: int, n1: int, x2: int, n2: int) -> Dict[str, float]:
    """Two-sided z-test for p2 - p1, pooled standard error under H0 (p1 = p2)."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return {"z": z, "p_value": 2 * stats.norm.sf(abs(z))}


def diff_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> tuple:
    """Confidence interval for p2 - p1 (unpooled Wald; fine at these sample sizes)."""
    p1, p2 = x1 / n1, x2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return (p2 - p1 - z * se, p2 - p1 + z * se)


def cohen_h(p1: float, p2: float) -> float:
    """Effect size for two proportions: 2*asin(sqrt(p2)) - 2*asin(sqrt(p1))."""
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def power_two_prop(p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """Power of the two-sided test to detect the gap between p1 and p2 at these sample sizes."""
    h = abs(cohen_h(p1, p2))
    se = math.sqrt(1 / n1 + 1 / n2)
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(z_a - h / se) + stats.norm.cdf(-z_a - h / se))


def mde_two_prop(p_base: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute drop or lift (in proportion points) detectable with this power.

    Solves Cohen's h = (z_alpha/2 + z_power) * sqrt(1/n1 + 1/n2) for p2, on the
    side of the baseline that gives the smaller (more conservative) change.
    """
    h = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * math.sqrt(1 / n1 + 1 / n2)
    phi = math.asin(math.sqrt(p_base))
    up = math.sin(min(phi + h / 2, math.pi / 2)) ** 2 - p_base
    down = p_base - math.sin(max(phi - h / 2, 0.0)) ** 2
    return min(up, down)


def srm_check(n_control: int, n_treatment: int, expected_share: float = 0.5) -> Dict[str, float]:
    """Sample ratio mismatch: did randomization deliver the planned split?

    Chi-square goodness of fit against the planned share. A small p-value means
    the split itself is off, which can signal a bug in assignment or logging.
    """
    n = n_control + n_treatment
    expected = [n * expected_share, n * (1 - expected_share)]
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {"chi2": float(chi2), "p_value": float(p), "treatment_share": n_treatment / n}


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error across metrics)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_diff(x1: int, n1: int, x2: int, n2: int, n_boot: int = 20000, seed: int = 42) -> Dict[str, object]:
    """Bootstrap distribution of p2 - p1.

    For a 0/1 outcome, resampling n users with replacement and taking the mean
    is exactly a Binomial(n, p_hat)/n draw, so this is fast and exact.
    """
    rng = np.random.default_rng(seed)
    d = rng.binomial(n2, x2 / n2, n_boot) / n2 - rng.binomial(n1, x1 / n1, n_boot) / n1
    return {"draws": d, "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "prob_treatment_worse": float((d < 0).mean())}
