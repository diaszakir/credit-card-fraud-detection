-- name: q08_time_between_frauds
-- title: Time gaps between consecutive frauds inside each split
-- question: How bursty is fraud: how long is the typical wait from one fraud to the next?
-- uses: CTE, JOIN, window function (LAG with PARTITION BY), GROUP BY
-- scope: all rows; gaps are computed inside a split so they never cross a split boundary
WITH fraud AS (
    SELECT
        s.split_name,
        t.transaction_id,
        t.time_sec,
        t.time_sec - LAG(t.time_sec) OVER (
            PARTITION BY t.split_id ORDER BY t.time_sec, t.transaction_id) AS gap_sec
    FROM transactions AS t
    JOIN splits       AS s ON s.split_id = t.split_id
    WHERE t.class_id = 1
)
SELECT
    split_name,
    COUNT(*)                   AS n_fraud,
    COUNT(gap_sec)             AS n_gaps,
    ROUND(AVG(gap_sec), 1)     AS avg_gap_sec,
    MIN(gap_sec)               AS min_gap_sec,
    MAX(gap_sec)               AS max_gap_sec,
    SUM(CASE WHEN gap_sec = 0 THEN 1 ELSE 0 END) AS gaps_within_same_second
FROM fraud
GROUP BY split_name
ORDER BY CASE split_name WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END;
