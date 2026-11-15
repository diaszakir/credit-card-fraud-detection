-- name: e01_errors_by_amount_band
-- title: Ошибки модели по диапазонам суммы
-- question: В каких диапазонах суммы модель пропускает fraud (FN) и даёт ложные тревоги (FP)?
-- uses: CTE, JOIN, JOIN (range / non-equi join), GROUP BY, HAVING
WITH labelled AS (
    SELECT p.transaction_id, p.y_true, p.y_pred, p.score, t.amount,
           CASE WHEN p.y_true = 1 AND p.y_pred = 1 THEN 'TP'
                WHEN p.y_true = 0 AND p.y_pred = 1 THEN 'FP'
                WHEN p.y_true = 1 AND p.y_pred = 0 THEN 'FN'
                ELSE 'TN' END AS outcome
    FROM eval_predictions AS p
    JOIN transactions     AS t ON t.transaction_id = p.transaction_id
)
SELECT b.label                                          AS amount_band,
       COUNT(*)                                         AS n_transactions,
       SUM(l.y_true)                                    AS n_fraud,
       SUM(l.outcome = 'TP')                            AS tp,
       SUM(l.outcome = 'FN')                            AS fn,
       SUM(l.outcome = 'FP')                            AS fp,
       ROUND(1.0 * SUM(l.outcome = 'TP') / NULLIF(SUM(l.y_true), 0), 3) AS recall
FROM labelled     AS l
JOIN amount_bands AS b ON l.amount >= b.lower_bound
                      AND (b.upper_bound IS NULL OR l.amount < b.upper_bound)
GROUP BY b.band_id, b.label
HAVING SUM(l.y_true) > 0 OR SUM(l.outcome = 'FP') > 0
ORDER BY b.band_id;
