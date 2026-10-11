"""Data checks, decisions, SQL cross-check, and the CLI end to end."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import ab_cli, analysis
from src.analysis import Spec

DATA = Path(__file__).resolve().parents[1] / "data" / "cookie_cats.csv"


def synthetic(n=20000, p_c=0.20, p_t=0.20, seed=0, split=0.5):
    rng = np.random.default_rng(seed)
    version = np.where(rng.random(n) < split, "gate_40", "gate_30")
    p = np.where(version == "gate_40", p_t, p_c)
    return pd.DataFrame({
        "userid": np.arange(n), "version": version,
        "sum_gamerounds": rng.poisson(30, n),
        "retention_1": rng.random(n) < p * 2, "retention_7": rng.random(n) < p,
    })


def test_checks_catch_duplicates_and_unknown_variants(tmp_path):
    df = synthetic(2000)
    df.loc[0, "userid"] = df.loc[1, "userid"]
    df.loc[2, "version"] = "gate_50"
    df.to_csv(tmp_path / "bad.csv", index=False)
    checks = analysis.data_checks(analysis.load(tmp_path / "bad.csv", Spec()), Spec()).set_index("check")
    assert checks.loc["duplicate users", "status"] == "fail"
    assert checks.loc["variants", "status"] == "fail"


def test_srm_is_flagged_for_a_skewed_split(tmp_path):
    synthetic(40000, split=0.53).to_csv(tmp_path / "srm.csv", index=False)
    checks = analysis.data_checks(analysis.load(tmp_path / "srm.csv", Spec()), Spec()).set_index("check")
    assert checks.loc["sample ratio (SRM)", "status"] == "warn"


def test_no_effect_gives_a_no_difference_decision(tmp_path):
    synthetic(20000).to_csv(tmp_path / "null.csv", index=False)
    df = analysis.load(tmp_path / "null.csv", Spec())
    res = analysis.analyze_metrics(df, Spec())
    assert not res["significant"].any()
    assert analysis.decision(res, analysis.data_checks(df, Spec()), Spec()).startswith("No significant")


def test_real_effect_is_detected(tmp_path):
    synthetic(60000, p_c=0.20, p_t=0.17).to_csv(tmp_path / "eff.csv", index=False)
    df = analysis.load(tmp_path / "eff.csv", Spec())
    res = analysis.analyze_metrics(df, Spec()).set_index("metric")
    assert res.loc["retention_7", "significant"] and res.loc["retention_7", "diff_pp"] < 0


def test_cookie_cats_results_and_sql_agree():
    from src.sql_check import cross_check, run_sql
    spec = Spec()
    res = analysis.analyze_metrics(analysis.load(DATA, spec), spec).set_index("metric")
    assert res.loc["retention_1", "p_value"] == pytest.approx(0.0744, abs=1e-4)
    assert res.loc["retention_7", "p_value"] == pytest.approx(0.0016, abs=1e-4)
    assert cross_check(run_sql(DATA), res.reset_index(), spec.control, spec.treatment)["match"].all()


def test_cli_writes_readout_and_charts(tmp_path, capsys):
    assert ab_cli.main(["--root", str(tmp_path)]) == 0
    for rel in ["reports/ab_readout.xlsx", "assets/effect_ci.png", "assets/bootstrap.png"]:
        assert (tmp_path / rel).stat().st_size > 0
    out = capsys.readouterr().out
    assert "DECISION: Keep gate_30" in out and "4/4 values match" in out
    sheets = pd.ExcelFile(tmp_path / "reports/ab_readout.xlsx").sheet_names
    assert sheets == ["Readout", "Results", "Checks", "Engagement", "SQL_crosscheck"]


def test_cli_works_on_any_schema(tmp_path):
    df = synthetic(5000).rename(columns={"version": "arm", "retention_7": "converted"})
    df["arm"] = df["arm"].map({"gate_30": "A", "gate_40": "B"})
    df[["userid", "arm", "converted"]].to_csv(tmp_path / "other.csv", index=False)
    code = ab_cli.main(["--data", str(tmp_path / "other.csv"), "--variant", "arm", "--control", "A",
                        "--treatment", "B", "--metrics", "converted", "--engagement", "none",
                        "--root", str(tmp_path)])
    assert code == 0


def test_cli_stops_on_failed_check(tmp_path):
    df = synthetic(1000)
    df.loc[0, "userid"] = df.loc[1, "userid"]
    df.to_csv(tmp_path / "dup.csv", index=False)
    assert ab_cli.main(["--data", str(tmp_path / "dup.csv"), "--root", str(tmp_path)]) == 1
