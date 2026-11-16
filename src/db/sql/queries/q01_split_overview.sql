-- name: q01_split_overview
-- title: Class balance and amounts per chronological split
-- question: How many transactions and frauds does each split hold, and how much money is involved?
-- uses: JOIN, GROUP BY, conditional aggregation
-- scope: all rows (descriptive)
SELECT
    s.split_name,
    COUNT(*)                                                       AS n_transactions,
    SUM(t.class_id)                                                AS n_fraud,
    ROUND(100.0 * SUM(t.class_id) / COUNT(*), 4)                   AS fraud_rate_pct,
    MIN(t.time_sec)                                                AS time_min_sec,
    MAX(t.time_sec)                                                AS time_max_sec,
    ROUND(SUM(t.amount), 2)                                        AS total_amount,
    ROUND(SUM(CASE WHEN t.class_id = 1 THEN t.amount ELSE 0 END), 2) AS fraud_amount
FROM transactions AS t
JOIN splits       AS s ON s.split_id = t.split_id
GROUP BY s.split_id, s.split_name
ORDER BY s.split_id;
