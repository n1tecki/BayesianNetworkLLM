-- Merging sepsis diagnoses stays with all admission information and additional non sepsis stays

-- CREATE INDEX idx_admissions_hadm_id ON admissions(hadm_id);
DROP TABLE IF EXISTS _icustays_sepsis;

CREATE TABLE _icustays_sepsis AS
WITH sepsis_cohort AS (
  SELECT DISTINCT
    d.subject_id,
    d.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    i.los
  FROM _diagnoses_icd_sepsis d
  JOIN icustays i 
    ON d.hadm_id = i.hadm_id
),
random_controls AS (
  SELECT DISTINCT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    i.los
  FROM icustays i
  WHERE i.hadm_id NOT IN (SELECT hadm_id FROM sepsis_cohort)
  ORDER BY random()
  LIMIT 10000
)
SELECT
  subject_id,
  hadm_id,
  stay_id,
  intime,
  outtime,
  los,
  1 AS sepsis
FROM sepsis_cohort

UNION ALL

SELECT
  subject_id,
  hadm_id,
  stay_id,
  intime,
  outtime,
  los,
  0 AS sepsis
FROM random_controls
;

CREATE INDEX idx_icustays_sepsis_hadm_id 
  ON _icustays_sepsis(hadm_id);

CREATE INDEX idx_icustays_sepsis_stay_id
  ON _icustays_sepsis(stay_id);
