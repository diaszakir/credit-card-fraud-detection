-- name: e04_label_check
-- title: Сверка меток прогнозов с базой
-- question: Совпадают ли y_true в прогнозах с классом транзакции в SQLite и все ли id есть в базе?
-- uses: CTE, JOIN, subquery, GROUP BY
WITH checked AS (
    SELECT p.transaction_id, p.y_true, t.class_id
    FROM eval_predictions AS p
    LEFT JOIN transactions AS t ON t.transaction_id = p.transaction_id
)
SELECT COUNT(*)                                          AS n_predictions,
       SUM(class_id IS NULL)                             AS missing_in_db,
       SUM(class_id IS NOT NULL AND class_id <> y_true)  AS label_mismatch,
       (SELECT COUNT(*) FROM eval_predictions WHERE y_true = 1) AS n_fraud_in_predictions
FROM checked;
