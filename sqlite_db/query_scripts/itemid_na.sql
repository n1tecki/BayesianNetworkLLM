-- Query for itemid code of Sodium (Na⁺) and subcategories 
-- to be classified as - NA
CREATE TEMP TABLE itemid_na AS
SELECT *, 'na' AS lab_category
FROM d_labitems
WHERE label LIKE '%Sodium%'
;



CREATE INDEX idx_itemid_na ON itemid_na(itemid);