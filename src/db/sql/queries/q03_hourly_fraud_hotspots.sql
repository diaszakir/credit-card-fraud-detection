-- name: q03_hourly_fraud_hotspots
-- title: Hours with the highest fraud rate (enough traffic only)
-- question: Which hours since the first transaction have the highest fraud rate, ignoring hours with little traffic?
-- uses: JOIN, GROUP BY, HAVING
-- scope: all rows (descriptive time profile)
-- param: min_transactions=1000
-- param: top_n=10
SELECT
    ts.hour_index,
    ts.day_index,
    ts.hour_in_cycle,
    COUNT(*)                                     AS n_transactions,
    SUM(t.class_id)                              AS n_fraud,
    ROUND(100.0 * SUM(t.class_id) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions AS t
JOIN time_slots   AS ts ON ts.hour_index = t.hour_index
GROUP BY ts.hour_index, ts.day_index, ts.hour_in_cycle
HAVING COUNT(*) >= :min_transactions
   AND SUM(t.class_id) > 0
ORDER BY fraud_rate_pct DESC, ts.hour_index
LIMIT :top_n;
