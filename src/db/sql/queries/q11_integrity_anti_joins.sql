-- name: q11_integrity_anti_joins
-- title: Referential integrity checks with anti-joins
-- question: Are there transactions without features, features without transactions, or unused dimension rows?
-- uses: LEFT JOIN (anti-join), UNION ALL, subquery
-- scope: whole database; every n_violations should be 0 except the informational unused_* rows
SELECT 'transactions_without_features' AS check_name, COUNT(*) AS n_violations
FROM transactions AS t
LEFT JOIN transaction_features AS f ON f.transaction_id = t.transaction_id
WHERE f.transaction_id IS NULL
UNION ALL
SELECT 'features_without_transactions', COUNT(*)
FROM transaction_features AS f
LEFT JOIN transactions AS t ON t.transaction_id = f.transaction_id
WHERE t.transaction_id IS NULL
UNION ALL
SELECT 'transactions_with_wrong_hour', COUNT(*)
FROM transactions
WHERE hour_index <> time_sec / 3600
UNION ALL
SELECT 'transactions_without_amount_band', COUNT(*)
FROM transactions AS t
WHERE NOT EXISTS (
    SELECT 1 FROM amount_bands AS b
    WHERE t.amount >= b.lower_bound AND (b.upper_bound IS NULL OR t.amount < b.upper_bound))
UNION ALL
SELECT 'unused_time_slots (informational)', COUNT(*)
FROM time_slots AS ts
LEFT JOIN transactions AS t ON t.hour_index = ts.hour_index
WHERE t.transaction_id IS NULL;
