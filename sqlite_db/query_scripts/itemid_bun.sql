-- Query for itemid code of Blood Urea Nitrogen (BUN) and subcategories 
-- to be classified as - BUN
CREATE TEMP TABLE itemid_bun AS
SELECT *, 'bun' AS lab_category
FROM d_labitems
WHERE label LIKE '%Urea Nitrogen%'
   OR label LIKE '%Bun%'
;




CREATE INDEX idx_itemid_bun ON itemid_bun(itemid);