-- Per-variant counts and retention rates, straight from the raw CSV.
-- Used as an independent cross-check of the pandas numbers (src/sql_check.py).
SELECT
    version                                         AS variant,
    COUNT(*)                                        AS users,
    COUNT(DISTINCT userid)                          AS distinct_users,
    SUM(CASE WHEN retention_1 THEN 1 ELSE 0 END)    AS retained_d1,
    SUM(CASE WHEN retention_7 THEN 1 ELSE 0 END)    AS retained_d7,
    AVG(CASE WHEN retention_1 THEN 1.0 ELSE 0 END)  AS rate_d1,
    AVG(CASE WHEN retention_7 THEN 1.0 ELSE 0 END)  AS rate_d7,
    MEDIAN(sum_gamerounds)                          AS median_rounds
FROM read_csv_auto(?)
GROUP BY version
ORDER BY version;
