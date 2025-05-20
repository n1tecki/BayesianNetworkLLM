
DROP TABLE IF EXISTS _icustays_sepsis_filtered;
CREATE TABLE _icustays_sepsis_filtered AS
WITH 
-- 1) keep only reasonably long stays
filtered AS (
  SELECT *
  FROM _icustays_sepsis
  WHERE los >= 0.04
),

-- 2) merge any multiple ICU stays per (subject_id, hadm_id)
merged AS (
  SELECT
    subject_id,
    hadm_id,
    MIN(intime)    AS intime,
    MAX(outtime)   AS outtime,
    -- recalc length of stay in days
    (julianday(MAX(outtime)) - julianday(MIN(intime))) AS los,
    sepsis
  FROM filtered
  GROUP BY subject_id, hadm_id
),

-- 3) pick only the first ICU stay per subject
numbered AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY subject_id
      ORDER BY intime ASC
    ) AS rn
  FROM merged
)

-- 4) output only the “rn = 1” rows
SELECT
  subject_id,
  hadm_id,
  intime,
  outtime,
  los,
  sepsis
FROM numbered
WHERE rn = 1
;

CREATE INDEX idx_icustays_sepsis_filtered_hadm_id
  ON _icustays_sepsis_filtered(hadm_id);