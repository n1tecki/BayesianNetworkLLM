DROP TABLE IF EXISTS labevents_itemid_total;
DROP TABLE IF EXISTS itemid_total;
DROP TABLE IF EXISTS admissions_icd_total;
DROP TABLE IF EXISTS icd_total;
DROP TABLE IF EXISTS matched_admission_labevents;
DROP TABLE IF EXISTS balanced_matched_admission_labevents;



SELECT diagnoses_category, COUNT(diagnoses_category)
FROM balanced_matched_admission_labevents
GROUP BY diagnoses_category;

SELECT lab_category, COUNT(lab_category)
FROM balanced_matched_admission_labevents
GROUP BY lab_category;