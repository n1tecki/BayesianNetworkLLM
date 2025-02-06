-- Query for itemid code of White Blood Cell (WBC) Count and subcategories 
-- to be classified as - WBC
CREATE TEMP TABLE itemid_wbc AS
SELECT *, 'wbc' AS lab_category
FROM d_labitems
WHERE label LIKE '%Leukocyte Alkaline Phosphatase%'
   OR label LIKE '%WBC Count%'
   OR label LIKE '%White Blood Cells%'
   OR label LIKE '%Leukocytes%'
   OR label LIKE '%WBC Casts%'
   OR label LIKE '%WBC Clumps%'
   OR label LIKE '%WBCScat%'
   OR label LIKE '%wbcp%'
   OR label = 'WBC'
   OR label = 'WBC '
;


CREATE INDEX idx_itemid_wbc ON itemid_wbc(itemid);