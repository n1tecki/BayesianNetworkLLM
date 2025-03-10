-- Query for icd codes of Congestive Heart Failure (CHF) and subcategories 
-- to be classified as - Congestive Heart Failure (CHF)
CREATE TEMP TABLE icd_chf AS
SELECT *, 'chf' AS diagnoses_category
FROM d_icd_diagnoses
WHERE long_title LIKE '%Rheumatic heart failure (congestive)%'
OR long_title LIKE '%Malignant hypertensive heart disease with heart failure%'
OR long_title LIKE '%Benign hypertensive heart disease with heart failure%'
OR long_title LIKE '%Unspecified hypertensive heart disease with heart failure%'
OR long_title LIKE '%Hypertensive heart and chronic kidney disease, malignant, with heart failure%'
OR long_title LIKE '%Hypertensive heart and chronic kidney disease, benign, with heart failure%'
OR long_title LIKE '%Hypertensive heart and chronic kidney disease, unspecified, with heart failure%'
OR long_title LIKE '%Congestive heart failure, unspecified%'
OR long_title LIKE '%Left heart failure%'
OR long_title LIKE '%Systolic heart failure, unspecified%'
OR long_title LIKE '%Acute systolic heart failure%'
OR long_title LIKE '%Chronic systolic heart failure%'
OR long_title LIKE '%Acute on chronic systolic heart failure%'
OR long_title LIKE '%Diastolic heart failure, unspecified%'
OR long_title LIKE '%Acute diastolic heart failure%'
OR long_title LIKE '%Chronic diastolic heart failure%'
OR long_title LIKE '%Acute on chronic diastolic heart failure%'
OR long_title LIKE '%Combined systolic and diastolic heart failure, unspecified%'
OR long_title LIKE '%Acute combined systolic and diastolic heart failure%'
OR long_title LIKE '%Chronic combined systolic and diastolic heart failure%'
OR long_title LIKE '%Acute on chronic combined systolic and diastolic heart failure%'
OR long_title LIKE '%Heart failure, unspecified%'
OR long_title LIKE '%Cardiogenic shock%'
OR long_title LIKE '%Postoperative shock, cardiogenic%'
OR long_title LIKE '%Rheumatic heart failure%'
OR long_title LIKE '%Hypertensive heart disease with heart failure%'
OR long_title LIKE '%Hypertensive heart and chronic kidney disease with heart failure and stage 1 through stage 4 chronic kidney disease%'
OR long_title LIKE '%Hypertensive heart and chronic kidney disease with heart failure and with stage 5 chronic kidney disease, or end stage renal disease%'
OR long_title LIKE '%Heart failure%' -- This will match broad ICD-10 I50 parent
OR long_title LIKE '%Systolic (congestive) heart failure%'
OR long_title LIKE '%Unspecified systolic (congestive) heart failure%'
OR long_title LIKE '%Acute systolic (congestive) heart failure%'
OR long_title LIKE '%Chronic systolic (congestive) heart failure%'
OR long_title LIKE '%Acute on chronic systolic (congestive) heart failure%'
OR long_title LIKE '%Diastolic (congestive) heart failure%'
OR long_title LIKE '%Unspecified diastolic (congestive) heart failure%'
OR long_title LIKE '%Acute diastolic (congestive) heart failure%'
OR long_title LIKE '%Chronic diastolic (congestive) heart failure%'
OR long_title LIKE '%Acute on chronic diastolic (congestive) heart failure%'
OR long_title LIKE '%Combined systolic (congestive) and diastolic (congestive) heart failure%'
OR long_title LIKE '%Unspecified combined systolic (congestive) and diastolic (congestive) heart failure%'
OR long_title LIKE '%Acute combined systolic (congestive) and diastolic (congestive) heart failure%'
OR long_title LIKE '%Chronic combined systolic (congestive) and diastolic (congestive) heart failure%'
OR long_title LIKE '%Acute on chronic combined systolic (congestive) and diastolic (congestive) heart failure%'
OR long_title LIKE '%Other heart failure%'
OR long_title LIKE '%Right heart failure%'
OR long_title LIKE '%Biventricular heart failure%'
OR long_title LIKE '%High output heart failure%'
OR long_title LIKE '%End stage heart failure%'
OR long_title LIKE '%Postprocedural heart failure%'
OR long_title LIKE '%Cardiogenic shock%'  -- Also included in R570, T8111, etc.
OR long_title LIKE '%Postprocedural cardiogenic shock%'
;


CREATE INDEX idx_icd_chf ON icd_chf(icd_code);