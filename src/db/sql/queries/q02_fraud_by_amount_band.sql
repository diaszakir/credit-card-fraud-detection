-- name: q02_fraud_by_amount_band
-- title: Fraud rate by business amount band
-- question: In which amount bands is fraud over- or under-represented?
-- uses: JOIN (range / non-equi join), JOIN, GROUP BY
-- scope: split given by :split_name (default train, because this relates a feature to the target)
-- param: split_name=train
SELECT
    b.band_id,
    b.label                                                  AS amount_band,
    COUNT(*)                                                 AS n_transactions,
    SUM(t.class_id)                                          AS n_fraud,
    ROUND(100.0 * SUM(t.class_id) / COUNT(*), 4)             AS fraud_rate_pct,
    ROUND(AVG(t.amount), 2)                                  AS avg_amount
FROM transactions  AS t
JOIN splits        AS s ON s.split_id = t.split_id
JOIN amount_bands  AS b ON t.amount >= b.lower_bound
                       AND (b.upper_bound IS NULL OR t.amount < b.upper_bound)
WHERE s.split_name = :split_name
GROUP BY b.band_id, b.label
ORDER BY b.band_id;
