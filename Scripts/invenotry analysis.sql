--LEAKAGE RATIO
--Cacses that remain oepn after 60 days, but were open 30 days ago

WITH CaseSnapshots AS (
    SELECT 
        case_id,
        created_date,
        -- How old was the case 30 days ago?
        DATEDIFF(day, created_date, DATEADD(day, -30, GETDATE())) AS age_30_days_ago,
        -- How old is it today?
        DATEDIFF(day, created_date, GETDATE()) AS current_age,
        [status]
    FROM cases_table
    -- We only care about cases that existed at least 30 days ago
    WHERE created_date <= DATEADD(day, -30, GETDATE())
)
SELECT 
    COUNT(case_id) AS total_cohort_size,
    -- Cases that migrated from <60 to >60 because they weren't closed
    SUM(CASE WHEN age_30_days_ago <= 60 AND current_age > 60 AND [status] = 'Open' THEN 1 ELSE 0 END) AS leaked_cases,
    -- Leakage Rate: What % of our workable backlog from last month did we fail to resolve?
    CAST(SUM(CASE WHEN age_30_days_ago <= 60 AND current_age > 60 AND [status] = 'Open' THEN 1 ELSE 0 END) AS FLOAT) / 
    NULLIF(SUM(CASE WHEN age_30_days_ago <= 60 THEN 1 ELSE 0 END), 0) * 100.0 AS leakage_rate_pct
FROM CaseSnapshots
;

--------------------------------------------------------------------------
--Median Outstanding Age vs Media Close Age Trend Over time

-------------------------------
--Cohort Survival Analysis
WITH cohorts AS (
    SELECT
        item_id,
        DATEADD(month, DATEDIFF(month, 0, open_date), 0)          AS cohort_month,
        DATEDIFF(day, open_date, COALESCE(close_date, GETDATE()))  AS age_days
    FROM inventory
)
SELECT
    cohort_month,
    COUNT(*)                                                        AS total_opened,
    SUM(CASE WHEN age_days > 90 THEN 1 ELSE 0 END)                 AS exceeded_90,
    ROUND(
        100.0 * SUM(CASE WHEN age_days > 90 THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                               AS pct_exceeded_90
FROM cohorts
GROUP BY cohort_month
ORDER BY cohort_month
;
-------------------------------
--near threshold risk
SELECT
    item_id,
    open_date,
    DATEDIFF(day, open_date, GETDATE())         AS age_days,
    90 - DATEDIFF(day, open_date, GETDATE())    AS days_until_breach,
    owner,
    category
FROM inventory
WHERE status = 'open'
  AND DATEDIFF(day, open_date, GETDATE()) BETWEEN 61 AND 90
ORDER BY age_days DESC
;

-------------------------------
--resoultion by age
SELECT
    CASE
        WHEN DATEDIFF(day, open_date, close_date) <= 30 THEN '0-30 days'
        WHEN DATEDIFF(day, open_date, close_date) <= 60 THEN '31-60 days'
        WHEN DATEDIFF(day, open_date, close_date) <= 90 THEN '61-90 days'
        ELSE '90+ days'
    END                                                       AS resolved_in,
    COUNT(*)                                                  AS items_closed,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2
    )                                                         AS pct_of_total
FROM inventory
WHERE status = 'closed'
  AND close_date >= DATEADD(month, -12, GETDATE())
GROUP BY
    CASE
        WHEN DATEDIFF(day, open_date, close_date) <= 30 THEN '0-30 days'
        WHEN DATEDIFF(day, open_date, close_date) <= 60 THEN '31-60 days'
        WHEN DATEDIFF(day, open_date, close_date) <= 90 THEN '61-90 days'
        ELSE '90+ days'
    END
;
-------------------------------
--dimension breakdown
SELECT
    category,
    owner,
    SUM(CASE WHEN age_days > 90 THEN 1 ELSE 0 END)   AS over_90,
    COUNT(*)                                           AS total_open,
    ROUND(
        100.0 * SUM(CASE WHEN age_days > 90 THEN 1 ELSE 0 END) / COUNT(*), 2
    )                                                  AS pct_over_90
FROM inventory_snapshots
WHERE status = 'open'
  AND snapshot_date = CAST(GETDATE() AS DATE)
GROUP BY category, owner
ORDER BY pct_over_90 DESC
