-- name: e03_error_ranking
-- title: Ошибки модели с рангом уверенности
-- question: Насколько уверенно модель ошибается: какие FN получили самый низкий score, а FP — самый высокий?
-- uses: CTE, JOIN, window function (OVER), subquery
WITH errors AS (
    SELECT p.transaction_id, p.score, t.amount, ts.hour_in_cycle,
           CASE WHEN p.y_true = 1 THEN 'FN' ELSE 'FP' END AS outcome
    FROM eval_predictions AS p
    JOIN transactions     AS t  ON t.transaction_id = p.transaction_id
    JOIN time_slots       AS ts ON ts.hour_index    = t.hour_index
    WHERE p.y_true <> p.y_pred
)
SELECT outcome, transaction_id, ROUND(score, 6) AS score, amount, hour_in_cycle,
       ROW_NUMBER() OVER (PARTITION BY outcome
                          ORDER BY CASE WHEN outcome = 'FN' THEN score ELSE -score END) AS error_rank,
       ROUND(AVG(score) OVER (PARTITION BY outcome), 6)                                 AS avg_score_in_group,
       ROUND(amount / (SELECT AVG(amount) FROM transactions AS a
                       WHERE a.transaction_id IN (SELECT transaction_id FROM eval_predictions)), 2)
                                                                                         AS amount_vs_split_avg
FROM errors
ORDER BY outcome, error_rank;
