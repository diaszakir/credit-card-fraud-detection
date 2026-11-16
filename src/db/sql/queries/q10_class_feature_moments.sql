-- name: q10_class_feature_moments
-- title: Mean and variance of every V feature, per class
-- question: Which anonymised features separate fraud from legitimate transactions?
-- uses: JOIN, JOIN, GROUP BY, aggregate expressions (population variance = E[x^2] - E[x]^2)
-- scope: split given by :split_name (default train, because this relates features to the target)
-- note: the result has one row per class; Pandas turns it into a ranked table of standardised mean differences
-- param: split_name=train
SELECT
    c.label AS class_label,
    COUNT(*) AS n_transactions,
    ROUND(AVG(f.v1), 6) AS mean_v1,
    ROUND(AVG(f.v1 * f.v1) - AVG(f.v1) * AVG(f.v1), 6) AS var_v1,
    ROUND(AVG(f.v2), 6) AS mean_v2,
    ROUND(AVG(f.v2 * f.v2) - AVG(f.v2) * AVG(f.v2), 6) AS var_v2,
    ROUND(AVG(f.v3), 6) AS mean_v3,
    ROUND(AVG(f.v3 * f.v3) - AVG(f.v3) * AVG(f.v3), 6) AS var_v3,
    ROUND(AVG(f.v4), 6) AS mean_v4,
    ROUND(AVG(f.v4 * f.v4) - AVG(f.v4) * AVG(f.v4), 6) AS var_v4,
    ROUND(AVG(f.v5), 6) AS mean_v5,
    ROUND(AVG(f.v5 * f.v5) - AVG(f.v5) * AVG(f.v5), 6) AS var_v5,
    ROUND(AVG(f.v6), 6) AS mean_v6,
    ROUND(AVG(f.v6 * f.v6) - AVG(f.v6) * AVG(f.v6), 6) AS var_v6,
    ROUND(AVG(f.v7), 6) AS mean_v7,
    ROUND(AVG(f.v7 * f.v7) - AVG(f.v7) * AVG(f.v7), 6) AS var_v7,
    ROUND(AVG(f.v8), 6) AS mean_v8,
    ROUND(AVG(f.v8 * f.v8) - AVG(f.v8) * AVG(f.v8), 6) AS var_v8,
    ROUND(AVG(f.v9), 6) AS mean_v9,
    ROUND(AVG(f.v9 * f.v9) - AVG(f.v9) * AVG(f.v9), 6) AS var_v9,
    ROUND(AVG(f.v10), 6) AS mean_v10,
    ROUND(AVG(f.v10 * f.v10) - AVG(f.v10) * AVG(f.v10), 6) AS var_v10,
    ROUND(AVG(f.v11), 6) AS mean_v11,
    ROUND(AVG(f.v11 * f.v11) - AVG(f.v11) * AVG(f.v11), 6) AS var_v11,
    ROUND(AVG(f.v12), 6) AS mean_v12,
    ROUND(AVG(f.v12 * f.v12) - AVG(f.v12) * AVG(f.v12), 6) AS var_v12,
    ROUND(AVG(f.v13), 6) AS mean_v13,
    ROUND(AVG(f.v13 * f.v13) - AVG(f.v13) * AVG(f.v13), 6) AS var_v13,
    ROUND(AVG(f.v14), 6) AS mean_v14,
    ROUND(AVG(f.v14 * f.v14) - AVG(f.v14) * AVG(f.v14), 6) AS var_v14,
    ROUND(AVG(f.v15), 6) AS mean_v15,
    ROUND(AVG(f.v15 * f.v15) - AVG(f.v15) * AVG(f.v15), 6) AS var_v15,
    ROUND(AVG(f.v16), 6) AS mean_v16,
    ROUND(AVG(f.v16 * f.v16) - AVG(f.v16) * AVG(f.v16), 6) AS var_v16,
    ROUND(AVG(f.v17), 6) AS mean_v17,
    ROUND(AVG(f.v17 * f.v17) - AVG(f.v17) * AVG(f.v17), 6) AS var_v17,
    ROUND(AVG(f.v18), 6) AS mean_v18,
    ROUND(AVG(f.v18 * f.v18) - AVG(f.v18) * AVG(f.v18), 6) AS var_v18,
    ROUND(AVG(f.v19), 6) AS mean_v19,
    ROUND(AVG(f.v19 * f.v19) - AVG(f.v19) * AVG(f.v19), 6) AS var_v19,
    ROUND(AVG(f.v20), 6) AS mean_v20,
    ROUND(AVG(f.v20 * f.v20) - AVG(f.v20) * AVG(f.v20), 6) AS var_v20,
    ROUND(AVG(f.v21), 6) AS mean_v21,
    ROUND(AVG(f.v21 * f.v21) - AVG(f.v21) * AVG(f.v21), 6) AS var_v21,
    ROUND(AVG(f.v22), 6) AS mean_v22,
    ROUND(AVG(f.v22 * f.v22) - AVG(f.v22) * AVG(f.v22), 6) AS var_v22,
    ROUND(AVG(f.v23), 6) AS mean_v23,
    ROUND(AVG(f.v23 * f.v23) - AVG(f.v23) * AVG(f.v23), 6) AS var_v23,
    ROUND(AVG(f.v24), 6) AS mean_v24,
    ROUND(AVG(f.v24 * f.v24) - AVG(f.v24) * AVG(f.v24), 6) AS var_v24,
    ROUND(AVG(f.v25), 6) AS mean_v25,
    ROUND(AVG(f.v25 * f.v25) - AVG(f.v25) * AVG(f.v25), 6) AS var_v25,
    ROUND(AVG(f.v26), 6) AS mean_v26,
    ROUND(AVG(f.v26 * f.v26) - AVG(f.v26) * AVG(f.v26), 6) AS var_v26,
    ROUND(AVG(f.v27), 6) AS mean_v27,
    ROUND(AVG(f.v27 * f.v27) - AVG(f.v27) * AVG(f.v27), 6) AS var_v27,
    ROUND(AVG(f.v28), 6) AS mean_v28,
    ROUND(AVG(f.v28 * f.v28) - AVG(f.v28) * AVG(f.v28), 6) AS var_v28
FROM transactions         AS t
JOIN transaction_features AS f ON f.transaction_id = t.transaction_id
JOIN splits               AS s ON s.split_id = t.split_id
JOIN classes              AS c ON c.class_id = t.class_id
WHERE s.split_name = :split_name
GROUP BY c.class_id, c.label
ORDER BY c.class_id;
