-- name: q05_rolling_fraud_rate
-- title: Three-hour rolling fraud rate with hour-to-hour change
-- question: How does fraud intensity move over time once single noisy hours are smoothed?
-- uses: CTE, window functions (SUM OVER with ROWS frame, LAG)
-- scope: all rows (descriptive)
WITH hourly AS (
    SELECT
        hour_index,
        COUNT(*)        AS n_transactions,
        SUM(class_id)   AS n_fraud
    FROM transactions
    GROUP BY hour_index
)
SELECT
    hour_index,
    n_transactions,
    n_fraud,
    ROUND(100.0 * n_fraud / n_transactions, 4)                                  AS fraud_rate_pct,
    ROUND(100.0 * SUM(n_fraud)        OVER w3 / SUM(n_transactions) OVER w3, 4) AS rolling_3h_rate_pct,
    ROUND(100.0 * n_fraud / n_transactions
          - 100.0 * LAG(n_fraud) OVER (ORDER BY hour_index)
                  / LAG(n_transactions) OVER (ORDER BY hour_index), 4)          AS change_vs_previous_hour_pp
FROM hourly
WINDOW w3 AS (ORDER BY hour_index ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
ORDER BY hour_index;
