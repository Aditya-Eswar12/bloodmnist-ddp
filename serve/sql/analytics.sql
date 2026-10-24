-- Analytics over the prediction log. Each query is separated by a line "-- name: <name>"
-- so db.py can load them by name. Parameters use :named placeholders.

-- name: differential
-- White-cell differential for one slide: % of accepted white cells per type.
-- Platelets are excluded; erythroblasts are reported per 100 white cells.
-- Flags compare the ROUNDED percentage with the range, so a displayed "1.0%" is never flagged above 1.0.
WITH accepted AS (                                   -- CTE 1: only cells that passed the gate
    SELECT pred_class
    FROM cells
    WHERE slide_id = :slide_id AND accepted = 1 AND pred_class <> 'platelet'
),
counts AS (                                          -- CTE 2: one row per cell type
    SELECT pred_class AS cell_type, COUNT(*) AS n
    FROM accepted
    GROUP BY pred_class
),
wbc_total AS (                                       -- CTE 3: white cells only
    SELECT SUM(n) AS total FROM counts WHERE cell_type <> 'erythroblast'
),
shares AS (                                          -- CTE 4: rounded percentages
    SELECT c.cell_type, c.n, ROUND(100.0 * c.n / NULLIF(w.total, 0), 1) AS pct, w.total AS wbc_counted
    FROM counts c CROSS JOIN wbc_total w
    UNION ALL                                        -- types never seen on this slide still get a 0 row
    SELECT r.cell_type, 0, 0.0, (SELECT total FROM wbc_total)
    FROM reference_ranges r
    WHERE r.cell_type NOT IN (SELECT cell_type FROM counts)
)
SELECT s.cell_type, s.n, s.pct, r.low_pct, r.high_pct,
       CASE WHEN s.pct > r.high_pct THEN 'high'
            WHEN s.pct < r.low_pct  THEN 'low'
            ELSE 'normal' END AS flag,
       s.wbc_counted
FROM shares s
LEFT JOIN reference_ranges r ON r.cell_type = s.cell_type
ORDER BY s.cell_type;

-- name: slide_qc
-- Per-slide quality numbers used by the QC gate.
SELECT slide_id,
       COUNT(*)                                        AS cells_total,
       SUM(accepted)                                   AS cells_accepted,
       ROUND(1.0 - AVG(accepted), 3)                   AS rejected_fraction,
       ROUND(AVG(confidence), 3)                       AS mean_confidence,
       SUM(CASE WHEN accepted = 1 AND pred_class NOT IN ('platelet', 'erythroblast')
                THEN 1 ELSE 0 END)                     AS wbc_accepted
FROM cells
WHERE slide_id = :slide_id
GROUP BY slide_id;

-- name: abnormal_cohort
-- Cohort extraction: every slide with at least one cell type outside its reference
-- range, ranked by how far outside it is (window function RANK over the cohort).
WITH per_type AS (
    SELECT slide_id, pred_class AS cell_type, COUNT(*) AS n
    FROM cells
    WHERE accepted = 1 AND pred_class <> 'platelet'
    GROUP BY slide_id, pred_class
),
pct AS (
    SELECT p.slide_id, p.cell_type,
           100.0 * p.n / SUM(CASE WHEN p.cell_type <> 'erythroblast' THEN p.n ELSE 0 END)
                         OVER (PARTITION BY p.slide_id)          AS pct      -- window: slide total
    FROM per_type p
),
deviation AS (
    SELECT q.slide_id, q.cell_type, ROUND(q.pct, 1) AS pct, r.low_pct, r.high_pct,
           MAX(ROUND(q.pct, 1) - r.high_pct, r.low_pct - ROUND(q.pct, 1), 0) AS excess
    FROM pct q JOIN reference_ranges r USING (cell_type)
)
SELECT slide_id, cell_type, pct, low_pct, high_pct, ROUND(excess, 1) AS excess,
       RANK() OVER (ORDER BY excess DESC)                         AS severity_rank,
       COUNT(*) OVER (PARTITION BY slide_id)                      AS abnormal_types_on_slide
FROM deviation
WHERE excess > 0
ORDER BY severity_rank, slide_id;

-- name: confidence_trend
-- Model monitoring: confidence of each prediction with a rolling mean over the
-- previous 50 cells, and the change since the previous slide (LAG).
WITH per_slide AS (
    SELECT slide_id, MIN(cell_id) AS first_cell, AVG(confidence) AS mean_conf,
           AVG(1 - accepted) AS reject_rate
    FROM cells GROUP BY slide_id
)
SELECT slide_id,
       ROUND(mean_conf, 3)                                             AS mean_confidence,
       ROUND(reject_rate, 3)                                           AS reject_rate,
       ROUND(mean_conf - LAG(mean_conf) OVER (ORDER BY first_cell), 3) AS change_vs_previous_slide,
       ROUND(AVG(mean_conf) OVER (ORDER BY first_cell
                                  ROWS BETWEEN 4 PRECEDING AND CURRENT ROW), 3) AS rolling_mean_5_slides
FROM per_slide
ORDER BY first_cell;

-- name: per_class_accuracy
-- Only for slides where the true class is known (simulated data or expert review):
-- accuracy per predicted class and its share of all errors.
WITH labelled AS (
    SELECT true_class, pred_class, accepted FROM cells WHERE true_class IS NOT NULL
),
per_class AS (
    SELECT true_class,
           COUNT(*)                                          AS n,
           SUM(pred_class = true_class)                      AS correct,
           SUM(pred_class <> true_class)                     AS wrong
    FROM labelled GROUP BY true_class
)
SELECT true_class, n, correct,
       ROUND(1.0 * correct / n, 3)                           AS recall,
       ROUND(1.0 * wrong / NULLIF(SUM(wrong) OVER (), 0), 3) AS share_of_all_errors
FROM per_class
ORDER BY recall;
