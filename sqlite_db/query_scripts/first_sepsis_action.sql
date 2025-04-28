DROP TABLE IF EXISTS _first_sepsis_action;

-- drug catalogue
CREATE TABLE _first_sepsis_action AS
WITH sepsis_items AS (          
    SELECT itemid, label
    FROM   d_items
    WHERE LOWER(label) LIKE '%antibiotic%'
        OR LOWER(label) LIKE '%vasopressor%'
        OR LOWER(label) LIKE '%norepinephrine%'
        OR LOWER(label) LIKE '%epinephrine%'
        OR LOWER(label) LIKE '%vasopressin%'
        OR LOWER(label) LIKE '%dopamine%'
        OR LOWER(label) LIKE '%hydrocortisone%'
        OR LOWER(label) LIKE '%vancomycin%'
        OR LOWER(label) LIKE '%meropenem%'
        OR LOWER(label) LIKE '%piperacillin%'
        OR LOWER(label) LIKE '%cefepime%'
        OR LOWER(label) LIKE '%ceftriaxone%'
        OR LOWER(label) LIKE '%linezolid%'
        OR LOWER(label) LIKE '%levofloxacin%'
        OR LOWER(label) LIKE '%ciprofloxacin%'
        OR LOWER(label) LIKE '%fluconazole%'
        OR LOWER(label) LIKE '%caspofungin%'
),

-- actual administrations
med_admin AS (                  
    /* ICU drips & boluses */
    SELECT ie.hadm_id,
           ie.itemid,
           si.label AS item_label,
           ie.starttime AS charttime
    FROM   inputevents    AS ie
    JOIN   sepsis_items   AS si USING (itemid)

    UNION ALL

    SELECT char.hadm_id,
           char.itemid,
           si.label    AS item_label,
           char.charttime AS charttime
    FROM   chartevents  AS char
    JOIN   sepsis_items AS si USING (itemid)
),
cohort AS (
    SELECT DISTINCT hadm_id
    FROM   _icustays_sepsis_filtered
),
ranked AS (
    SELECT m.*,
           ROW_NUMBER() OVER (
               PARTITION BY m.hadm_id
               ORDER BY      m.charttime
           ) AS rn
    FROM   med_admin AS m
    JOIN   cohort    AS c USING (hadm_id)
)
SELECT hadm_id,
       charttime      AS first_action_time,
       itemid,
       item_label
FROM   ranked
WHERE  rn = 1 -- earliest only
ORDER  BY hadm_id;

CREATE UNIQUE INDEX idx_first_sepsis_action_hadm
    ON _first_sepsis_action (hadm_id);

CREATE INDEX idx_first_sepsis_action_time
    ON _first_sepsis_action (first_action_time);