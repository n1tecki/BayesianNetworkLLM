-- Query for itemid code of Creatinine and subcategories 
-- to be classified as - Creatinine
CREATE TEMP TABLE itemid_creatinine AS
SELECT *, 'creatinine' AS lab_category
FROM d_labitems
WHERE label LIKE '%Creatinine%'
   OR label = 'Cr'
;



CREATE INDEX idx_itemid_creatinine ON itemid_creatinine(itemid);