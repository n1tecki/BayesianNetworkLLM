-- Creating a table of matched admissions with icd diagnoses and lab events
-- Additionally matched by lab date and patient admission date for close timeframe
CREATE TABLE matched_admission_labevents AS 
SELECT *
FROM labevents_itemid_total l
JOIN admissions_icd_total a ON l.subject_id = a.subject_id
WHERE l.charttime BETWEEN DATE(a.admittime, '-1 month') 
                      AND DATE(a.dischtime, '+1 month')
LIMIT 1000000;

CREATE INDEX idx_matched_admission_labevents ON matched_admission_labevents(subject_id);
