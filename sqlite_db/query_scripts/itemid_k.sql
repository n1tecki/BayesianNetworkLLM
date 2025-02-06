-- Query for itemid code of Potassium (K⁺) and subcategories 
-- to be classified as - K
CREATE TEMP TABLE itemid_k AS
SELECT *, 'k' AS lab_category
FROM d_labitems
WHERE label LIKE '%Potassium%'
   OR label = 'K'
   OR label LIKE 'K (GREEN)%'
;



CREATE INDEX idx_itemid_k ON itemid_k(itemid);
