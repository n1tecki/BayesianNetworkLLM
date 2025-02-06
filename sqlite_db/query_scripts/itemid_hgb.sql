-- Query for itemid code of Hemoglobin (Hgb) Count and subcategories 
-- to be classified as - HGB
CREATE TEMP TABLE itemid_hgb AS
SELECT *, 'hgb' AS lab_category
FROM d_labitems
WHERE label LIKE '%Hematocrit%'
   OR label LIKE '%Hemoglobin%'
   OR label LIKE '%RBC%'
   OR label LIKE '%Red Blood Cells%'
   OR label LIKE '%Reticulocyte%'
   OR label LIKE '%NRBC%'
   OR label LIKE '%Glycated Hemoglobin%'
   OR label LIKE '%Hgb%'
;



CREATE INDEX idx_itemid_hgb ON itemid_hgb(itemid);