-- name: q04_hours_above_average_fraud_rate
-- title: Hours whose fraud rate exceeds the overall rate
-- question: In how many hours is the fraud rate above the dataset-wide average, and by what factor?
-- uses: JOIN, GROUP BY, HAVING, scalar subquery
-- scope: all rows (descriptive)
SELECT
    ts.hour_index,
    ts.hour_in_cycle,
    COUNT(*)                                                       AS n_transactions,
    SUM(t.class_id)                                                AS n_fraud,
    ROUND(100.0 * SUM(t.class_id) / COUNT(*), 4)                   AS fraud_rate_pct,
    ROUND((1.0 * SUM(t.class_id) / COUNT(*))
          / (SELECT 1.0 * SUM(class_id) / COUNT(*) FROM transactions), 2) AS times_overall_rate
FROM transactions AS t
JOIN time_slots   AS ts ON ts.hour_index = t.hour_index
GROUP BY ts.hour_index, ts.hour_in_cycle
HAVING 1.0 * SUM(t.class_id) / COUNT(*)
       > (SELECT 1.0 * SUM(class_id) / COUNT(*) FROM transactions)
ORDER BY times_overall_rate DESC;
