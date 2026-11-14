-- name: q06_fraud_pareto_by_hour
-- title: Concentration of fraud: hours ranked by fraud count with cumulative share
-- question: How many hours account for half of all frauds?
-- uses: CTE, window functions (RANK, running SUM OVER, SUM OVER ())
-- scope: all rows (descriptive)
WITH hourly AS (
    SELECT hour_index, COUNT(*) AS n_transactions, SUM(class_id) AS n_fraud
    FROM transactions
    GROUP BY hour_index
)
SELECT
    RANK() OVER (ORDER BY n_fraud DESC)                      AS fraud_rank,
    hour_index,
    n_transactions,
    n_fraud,
    ROUND(100.0 * SUM(n_fraud) OVER (
              ORDER BY n_fraud DESC, hour_index
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / SUM(n_fraud) OVER (), 2)                         AS cumulative_share_of_frauds_pct
FROM hourly
ORDER BY n_fraud DESC, hour_index;
