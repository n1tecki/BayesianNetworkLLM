-- All icd codes of the major categories are stored in a temporary table all_icd_codes

CREATE TABLE itemid_total AS 
SELECT itemid, label, lab_category FROM itemid_bun
UNION
SELECT itemid, label, lab_category FROM itemid_creatinine
UNION
SELECT itemid, label, lab_category FROM itemid_hgb
UNION
SELECT itemid, label, lab_category FROM itemid_k
UNION
SELECT itemid, label, lab_category FROM itemid_lactate
UNION
SELECT itemid, label, lab_category FROM itemid_na
UNION
SELECT itemid, label, lab_category FROM itemid_platelets
UNION
SELECT itemid, label, lab_category FROM itemid_wbc;

CREATE INDEX idx_itemid_total ON itemid_total(itemid);