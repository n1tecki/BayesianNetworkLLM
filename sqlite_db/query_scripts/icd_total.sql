-- All icd codes of the major categories are stored in a temporary table all_icd_codes.
DROP TABLE IF EXISTS icd_total;

CREATE TABLE icd_total AS 
SELECT icd_code, long_title, diagnoses_category FROM icd_sepsis
UNION
SELECT icd_code, long_title, diagnoses_category FROM icd_aci
UNION
SELECT icd_code, long_title, diagnoses_category FROM icd_pneumonia
UNION
SELECT icd_code, long_title, diagnoses_category FROM icd_chf
UNION
SELECT icd_code, long_title, diagnoses_category FROM icd_gi;

CREATE INDEX idx_icd_total ON icd_total(icd_code);