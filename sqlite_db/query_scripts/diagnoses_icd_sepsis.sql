-- selecting diagnoses of patients that match sepsis icd codes

DROP TABLE IF EXISTS _diagnoses_icd_sepsis;

CREATE TABLE _diagnoses_icd_sepsis AS 
SELECT DISTINCT
    subject_id,
    hadm_id,
    seq_num
FROM diagnoses_icd
WHERE icd_code IN (SELECT icd FROM _sepsis_icd_codes);

CREATE INDEX idx_diagnoses_icd_sepsis ON _diagnoses_icd_sepsis(subject_id);