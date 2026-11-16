-- name: q09_top_fraud_amounts_per_split
-- title: Largest fraudulent transactions in each split
-- question: Which are the three biggest frauds per split and when did they happen?
-- uses: CTE, JOIN, window function (ROW_NUMBER with PARTITION BY)
-- scope: all rows (descriptive)
-- param: top_n=3
WITH ranked AS (
    SELECT
        s.split_name,
        t.transaction_id,
        t.time_sec,
        t.hour_index,
        t.amount,
        ROW_NUMBER() OVER (PARTITION BY t.split_id
                           ORDER BY t.amount DESC, t.transaction_id) AS rank_in_split
    FROM transactions AS t
    JOIN splits       AS s ON s.split_id = t.split_id
    WHERE t.class_id = 1
)
SELECT split_name, rank_in_split, transaction_id, time_sec, hour_index, amount
FROM ranked
WHERE rank_in_split <= :top_n
ORDER BY CASE split_name WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END, rank_in_split;
