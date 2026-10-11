"""Recompute per-variant metrics in SQL (DuckDB) and confirm they match pandas."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "variant_metrics.sql"


def run_sql(csv_path: Path) -> pd.DataFrame:
    import duckdb

    with duckdb.connect() as con:
        return con.execute(SQL_FILE.read_text(), [str(csv_path)]).df()


def cross_check(sql_df: pd.DataFrame, results: pd.DataFrame, control: str, treatment: str) -> pd.DataFrame:
    """Compare SQL rates with the pandas results table; returns one row per value checked."""
    sql = sql_df.set_index("variant")
    rows = []
    for metric, col in [("retention_1", "rate_d1"), ("retention_7", "rate_d7")]:
        r = results.set_index("metric").loc[metric]
        for variant, py_value in [(control, r["rate_control"]), (treatment, r["rate_treatment"])]:
            sql_value = float(sql.loc[variant, col])
            rows.append({"metric": metric, "variant": variant, "pandas": py_value, "sql": sql_value,
                         "match": abs(py_value - sql_value) < 1e-12})
    return pd.DataFrame(rows)
