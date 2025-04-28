------------------------------------------------------------------------
-- 1) Drop any existing table to start fresh
------------------------------------------------------------------------
DROP TABLE IF EXISTS _icustays_sepsis;

------------------------------------------------------------------------
-- 2) Build two cohorts: sepsis cases and random non-sepsis controls
------------------------------------------------------------------------
CREATE TABLE _icustays_sepsis AS
WITH 

  /*  
   sepsis_cohort: all ICU stays where the patient had a sepsis diagnosis 
   and a minimum length of stay (los) threshold 
  */
  sepsis_cohort AS (
    SELECT DISTINCT
      d.subject_id,
      d.hadm_id,
      i.stay_id,
      i.intime,
      i.outtime,
      i.los,
      1 AS sepsis                   -- flag these as cases
    FROM _diagnoses_icd_sepsis d
    JOIN icustays i 
      ON d.hadm_id = i.hadm_id
    WHERE i.los >= 0.04             -- filter out extremely short stays
  ),

  /*
   random_controls: a random sample of one ICU stay per subject WITHOUT sepsis
   diagnoses, enforcing 10,000 distinct subjects, same los filter.
   We use ROW_NUMBER() to pick one random stay per subject, then LIMIT to 10,000.
  */
  random_controls AS (
    SELECT
      subject_id,
      hadm_id,
      stay_id,
      intime,
      outtime,
      los,
      0 AS sepsis                   -- flag these as controls
    FROM (
      SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        i.los,
        ROW_NUMBER() OVER (
          PARTITION BY i.subject_id
          ORDER BY random()
        ) AS rn
      FROM icustays i
      WHERE i.los >= 0.04
        AND NOT EXISTS (
          SELECT 1 FROM sepsis_cohort sc WHERE sc.hadm_id = i.hadm_id
        )
    ) t
    WHERE rn = 1
    LIMIT 10000                         -- cap at 10,000 distinct subjects
  ),

  /*
   all_data: union both cohorts into a single set for further filtering
  */
  all_data AS (
    SELECT * FROM sepsis_cohort
    UNION ALL
    SELECT * FROM random_controls
  ),

  /*
   first_intimes: for each subject, find their earliest ICU admission time
  */
  first_intimes AS (
    SELECT
      subject_id,
      MIN(intime) AS first_intime
    FROM all_data
    GROUP BY subject_id
  )

------------------------------------------------------------------------
-- 3) Final selection: keep only stays whose intime equals each subject’s
--    earliest intime.  This will:
--      • Drop any additional stays beyond the first per subject.
--      • Still retain duplicates if they share the same (subject_id, hadm_id)
--        and identical earliest intime.
------------------------------------------------------------------------
SELECT
  a.subject_id,
  a.hadm_id,
  a.stay_id,
  a.intime,
  a.outtime,
  a.los,
  a.sepsis
FROM all_data a
JOIN first_intimes f
  ON a.subject_id = f.subject_id
  AND a.intime    = f.first_intime;

------------------------------------------------------------------------
-- 4) Re-create indexes to support fast lookups by hadm_id and stay_id
------------------------------------------------------------------------
CREATE INDEX idx_icustays_sepsis_hadm_id 
  ON _icustays_sepsis(hadm_id);

CREATE INDEX idx_icustays_sepsis_stay_id
  ON _icustays_sepsis(stay_id);




SELECT DISTINCT subject_id FROM _icustays_sepsis
ORDER BY los ASC

SELECT * FROM _icustays_sepsis_filtered
WHERE sepsis = 1