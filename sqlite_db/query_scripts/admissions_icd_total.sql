-- Based on the icd diagnoses codes, tha patients are selected from the admissions table.
DROP TABLE IF EXISTS admissions_icd_total;

CREATE TABLE admissions_icd_total AS 
SELECT 
    a.subject_id, 
    a.hadm_id, 
    a.admittime, 
    a.dischtime, 
    d.icd_code, 
    icd.long_title,
    icd_total.diagnoses_category
FROM admissions a
JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
JOIN d_icd_diagnoses icd ON d.icd_code = icd.icd_code
JOIN icd_total ON d.icd_code = icd_total.icd_code;

CREATE INDEX idx_admissions_icd_total ON admissions_icd_total(subject_id);