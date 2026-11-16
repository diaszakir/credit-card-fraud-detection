-- name: q07_fraud_rate_by_amount_decile
-- title: Fraud rate by amount decile
-- question: Does fraud concentrate in the low or high end of the amount distribution?
-- uses: CTE, JOIN, window function (NTILE), GROUP BY
-- scope: split given by :split_name (default train, because this relates a feature to the target)
-- note: amounts contain many ties, so equal amounts can fall into two neighbouring deciles
-- param: split_name=train
WITH scoped AS (
    SELECT t.transaction_id, t.amount, t.class_id
    FROM transactions AS t
    JOIN splits       AS s ON s.split_id = t.split_id
    WHERE s.split_name = :split_name
),
ranked AS (
    SELECT transaction_id, amount, class_id,
           NTILE(10) OVER (ORDER BY amount, transaction_id) AS amount_decile
    FROM scoped
)
SELECT
    amount_decile,
    COUNT(*)                                     AS n_transactions,
    MIN(amount)                                  AS min_amount,
    MAX(amount)                                  AS max_amount,
    SUM(class_id)                                AS n_fraud,
    ROUND(100.0 * SUM(class_id) / COUNT(*), 4)   AS fraud_rate_pct
FROM ranked
GROUP BY amount_decile
ORDER BY amount_decile;
