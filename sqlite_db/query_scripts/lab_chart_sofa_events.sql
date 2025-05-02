/*CREATE INDEX idx_labevents_itemid_hadm_charttime
    ON labevents (itemid, hadm_id, charttime);

CREATE INDEX idx_chartevents_itemid_hadm_charttime
    ON chartevents (itemid, hadm_id, charttime);*/


DROP TABLE IF EXISTS _lab_chart_sofa_events;

CREATE TABLE _lab_chart_sofa_events AS
WITH
/* -------- LABEVENTS -------- */
pao2 AS (
    SELECT hadm_id, storetime, value, valuenum, valueuom
    FROM (
        SELECT le.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id
                                  ORDER BY charttime ASC) AS rn
        FROM   labevents le
        WHERE  itemid = 50821
    )
    WHERE rn = 1
),
platelet AS (
    SELECT hadm_id, storetime, value, valuenum, valueuom
    FROM (
        SELECT le.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM labevents le
        WHERE itemid = 51265
    )
    WHERE rn = 1
),
bilirubin AS (
    SELECT hadm_id, storetime, value, valuenum, valueuom
    FROM (
        SELECT le.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM labevents le
        WHERE itemid = 50885
    )
    WHERE rn = 1
),
creatinine AS (
    SELECT hadm_id, storetime, value, valuenum, valueuom
    FROM (
        SELECT le.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM labevents le
        WHERE itemid = 50912
    )
    WHERE rn = 1
),
/* -------- CHARTEVENTS -------- */
vaso AS (
    SELECT hadm_id, charttime, storetime, value, valuenum, valueuom
    FROM (
        SELECT ce.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM chartevents ce
        WHERE itemid = 221906
    )
    WHERE rn = 1
),
fio2 AS (
    SELECT hadm_id, charttime, storetime, value, valuenum, valueuom
    FROM (
        SELECT ce.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM chartevents ce
        WHERE itemid = 223835
    )
    WHERE rn = 1
),
map AS (
    SELECT hadm_id, charttime, storetime, value, valuenum, valueuom
    FROM (
        SELECT ce.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM chartevents ce
        WHERE itemid = 220045
    )
    WHERE rn = 1
),
gcs_eye AS (
    SELECT hadm_id, charttime, storetime, value, valuenum, valueuom
    FROM (
        SELECT ce.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM chartevents ce
        WHERE itemid IN (184,220739, 223902)
    )
    WHERE rn = 1
),
gcs_verbal AS (
    SELECT hadm_id, charttime, storetime, value, valuenum, valueuom
    FROM (
        SELECT ce.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM chartevents ce
        WHERE itemid IN (723, 223900)
    )
    WHERE rn = 1
),
gcs_motor AS (
    SELECT hadm_id, charttime, storetime, value, valuenum, valueuom
    FROM (
        SELECT ce.*,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
        FROM chartevents ce
        WHERE itemid IN (454, 223901)
    )
    WHERE rn = 1
)

/* ------------ final SELECT pulled into the new table ------------ */
SELECT DISTINCT
    s.hadm_id,
    s.sepsis,

    /* labs */
    pao2.storetime            AS PaO2_storetime,
    pao2.value                AS PaO2_value,
    pao2.valuenum             AS PaO2_valuenum,
    pao2.valueuom             AS PaO2_valueuom,

    platelet.storetime        AS platelet_count_storetime,
    platelet.value            AS platelet_count_value,
    platelet.valuenum         AS platelet_count_valuenum,
    platelet.valueuom         AS platelet_count_valueuom,

    bilirubin.storetime       AS bilirubin_total_storetime,
    bilirubin.value           AS bilirubin_total_value,
    bilirubin.valuenum        AS bilirubin_total_valuenum,
    bilirubin.valueuom        AS bilirubin_total_valueuom,

    creatinine.storetime      AS creatinin_storetime,
    creatinine.value          AS creatinin_value,
    creatinine.valuenum       AS creatinin_valuenum,
    creatinine.valueuom       AS creatinin_valueuom,

    /* bedside signals */
    vaso.charttime            AS vasopressors_charttime,
    vaso.storetime            AS vasopressors_storetime,
    vaso.value                AS vasopressors_value,
    vaso.valuenum             AS vasopressors_valuenum,
    vaso.valueuom             AS vasopressors_valueuom,

    fio2.charttime            AS FiO2_charttime,
    fio2.storetime            AS FiO2_storetime,
    fio2.value                AS FiO2_value,
    fio2.valuenum             AS FiO2_valuenum,
    fio2.valueuom             AS FiO2_valueuom,

    map.charttime             AS mean_arterial_pressure_charttime,
    map.storetime             AS mean_arterial_pressure_storetime,
    map.value                 AS mean_arterial_pressure_value,
    map.valuenum              AS mean_arterial_pressure_valuenum,
    map.valueuom              AS mean_arterial_pressure_valueuom,

    gcs_eye.charttime         AS gcs_eye_charttime,
    gcs_eye.storetime         AS gcs_eye_storetime,
    gcs_eye.value             AS gcs_eye_value,
    gcs_eye.valuenum          AS gcs_eye_valuenum,
    gcs_eye.valueuom          AS gcs_eye_valueuom,

    gcs_verbal.charttime      AS gcs_verbal_charttime,
    gcs_verbal.storetime      AS gcs_verbal_storetime,
    gcs_verbal.value          AS gcs_verbal_value,
    gcs_verbal.valuenum       AS gcs_verbal_valuenum,
    gcs_verbal.valueuom       AS gcs_verbal_valueuom,

    gcs_motor.charttime       AS gcs_motor_charttime,
    gcs_motor.storetime       AS gcs_motor_storetime,
    gcs_motor.value           AS gcs_motor_value,
    gcs_motor.valuenum        AS gcs_motor_valuenum,
    gcs_motor.valueuom        AS gcs_motor_valueuom

FROM   _icustays_sepsis_filtered AS s
LEFT   JOIN pao2        USING (hadm_id)
LEFT   JOIN platelet    USING (hadm_id)
LEFT   JOIN bilirubin   USING (hadm_id)
LEFT   JOIN creatinine  USING (hadm_id)
LEFT   JOIN vaso        USING (hadm_id)
LEFT   JOIN fio2        USING (hadm_id)
LEFT   JOIN map         USING (hadm_id)
LEFT   JOIN gcs_eye     USING (hadm_id)
LEFT   JOIN gcs_verbal  USING (hadm_id)
LEFT   JOIN gcs_motor   USING (hadm_id)
ORDER  BY s.hadm_id;

/* Index for faster downstream joins */
CREATE INDEX idx__lab_chart_sofa_events_hadm_id
          ON _lab_chart_sofa_events (hadm_id);
