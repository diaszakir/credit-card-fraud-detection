-- name: e02_errors_by_hour
-- title: Ошибки модели по часу суток
-- question: В какие часы суток сосредоточены пропуски fraud и ложные тревоги?
-- uses: CTE, JOIN, GROUP BY, HAVING
WITH labelled AS (
    SELECT p.y_true, p.y_pred, ts.hour_in_cycle
    FROM eval_predictions AS p
    JOIN transactions     AS t  ON t.transaction_id = p.transaction_id
    JOIN time_slots       AS ts ON ts.hour_index    = t.hour_index
)
SELECT hour_in_cycle,
       COUNT(*)                                   AS n_transactions,
       SUM(y_true)                                AS n_fraud,
       SUM(y_true = 1 AND y_pred = 0)             AS fn,
       SUM(y_true = 0 AND y_pred = 1)             AS fp
FROM labelled
GROUP BY hour_in_cycle
HAVING SUM(y_true) > 0 OR SUM(y_true = 0 AND y_pred = 1) > 0
ORDER BY hour_in_cycle;
