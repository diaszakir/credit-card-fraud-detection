-- name: q12_duplicate_transactions
-- title: Repeated processed transactions (all 31 stored data columns identical)
-- question: How many rows repeat an earlier row, in which classes, and in groups of what size?
-- uses: CTE, JOIN, GROUP BY, HAVING, second GROUP BY over the CTE
-- scope: processed rows; raw duplicates were removed before float32 casts (see docs/CONTRACTS.md)
-- note: n_extra_rows must match Pandas on the processed data, not the raw audit before deduplication
WITH duplicate_groups AS (
    SELECT
        t.time_sec, t.amount, t.class_id,
        f.v1, f.v2, f.v3, f.v4, f.v5, f.v6, f.v7, f.v8, f.v9, f.v10, f.v11, f.v12, f.v13, f.v14, f.v15, f.v16, f.v17, f.v18, f.v19, f.v20, f.v21, f.v22, f.v23, f.v24, f.v25, f.v26, f.v27, f.v28,
        COUNT(*)              AS copies,
        MIN(t.transaction_id) AS first_transaction_id
    FROM transactions         AS t
    JOIN transaction_features AS f ON f.transaction_id = t.transaction_id
    GROUP BY t.time_sec, t.amount, t.class_id, f.v1, f.v2, f.v3, f.v4, f.v5, f.v6, f.v7, f.v8, f.v9, f.v10, f.v11, f.v12, f.v13, f.v14, f.v15, f.v16, f.v17, f.v18, f.v19, f.v20, f.v21, f.v22, f.v23, f.v24, f.v25, f.v26, f.v27, f.v28
    HAVING COUNT(*) > 1
)
SELECT
    c.label                 AS class_label,
    g.copies,
    COUNT(*)                AS n_groups,
    SUM(g.copies - 1)       AS n_extra_rows
FROM duplicate_groups AS g
JOIN classes          AS c ON c.class_id = g.class_id
GROUP BY c.class_id, c.label, g.copies
ORDER BY c.class_id, g.copies;
