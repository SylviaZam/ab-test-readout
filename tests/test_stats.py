"""Each formula checked against statsmodels/scipy or a known answer."""
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from src import stats

# Cookie Cats Day-7 retention: control gate_30, treatment gate_40
X1, N1, X2, N2 = 8502, 44700, 8279, 45489


def test_ztest_matches_statsmodels():
    ours = stats.two_prop_ztest(X1, N1, X2, N2)
    z, p = proportions_ztest([X2, X1], [N2, N1])
    assert ours["z"] == pytest.approx(z)
    assert ours["p_value"] == pytest.approx(p)


def test_ci_contains_point_estimate_and_excludes_zero_here():
    lo, hi = stats.diff_ci(X1, N1, X2, N2)
    diff = X2 / N2 - X1 / N1
    assert lo < diff < hi < 0


def test_cohen_h_matches_statsmodels():
    p1, p2 = X1 / N1, X2 / N2
    assert stats.cohen_h(p1, p2) == pytest.approx(proportion_effectsize(p2, p1))


def test_power_matches_statsmodels():
    p1, p2 = X1 / N1, X2 / N2
    expected = NormalIndPower().power(proportion_effectsize(p2, p1), nobs1=N1, ratio=N2 / N1, alpha=0.05)
    assert stats.power_two_prop(p1, p2, N1, N2) == pytest.approx(expected, rel=1e-6)


def test_mde_is_the_effect_with_80_percent_power():
    p1 = X1 / N1
    mde = stats.mde_two_prop(p1, N1, N2)
    assert stats.power_two_prop(p1, p1 - mde, N1, N2) == pytest.approx(0.8, abs=0.005)


def test_srm_flags_the_cookie_cats_split():
    out = stats.srm_check(44700, 45489)
    assert out["p_value"] == pytest.approx(0.0086, abs=1e-4)
    assert stats.srm_check(50000, 50010)["p_value"] > 0.9


def test_holm_adjustment():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.0016, 0.0744]) == pytest.approx([0.0032, 0.0744])


def test_bootstrap_is_reproducible_and_agrees_with_the_ci():
    a = stats.bootstrap_diff(X1, N1, X2, N2, seed=7)
    b = stats.bootstrap_diff(X1, N1, X2, N2, seed=7)
    assert a["prob_treatment_worse"] == b["prob_treatment_worse"]
    lo, hi = stats.diff_ci(X1, N1, X2, N2)
    assert a["ci_low"] == pytest.approx(lo, abs=6e-4)
    assert a["ci_high"] == pytest.approx(hi, abs=6e-4)
