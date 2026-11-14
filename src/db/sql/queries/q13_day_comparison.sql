-- name: q13_day_comparison
-- title: Day 1 versus day 2 fraud rate for the same hour of the cycle
-- question: Does the hourly fraud pattern repeat from the first 24 hours to the second 24 hours?
-- uses: CTE joined to itself (self-join), JOIN, GROUP BY
-- scope: all rows (descriptive)
-- note: the data has no absolute clock time; hour_in_cycle = hours since the first transaction modulo 24
WITH cycle AS (
    SELECT
        ts.day_index,
        ts.hour_in_cycle,
        COUNT(*)        AS n_transactions,
        SUM(t.class_id) AS n_fraud
    FROM transactions AS t
    JOIN time_slots   AS ts ON ts.hour_index = t.hour_index
    GROUP BY ts.day_index, ts.hour_in_cycle
)
SELECT
    d1.hour_in_cycle,
    d1.n_transactions                                  AS day1_transactions,
    d1.n_fraud                                         AS day1_fraud,
    ROUND(100.0 * d1.n_fraud / d1.n_transactions, 4)   AS day1_rate_pct,
    d2.n_transactions                                  AS day2_transactions,
    d2.n_fraud                                         AS day2_fraud,
    ROUND(100.0 * d2.n_fraud / d2.n_transactions, 4)   AS day2_rate_pct
FROM cycle AS d1
JOIN cycle AS d2 ON d2.hour_in_cycle = d1.hour_in_cycle
                AND d2.day_index = d1.day_index + 1
WHERE d1.day_index = 0
ORDER BY d1.hour_in_cycle;
