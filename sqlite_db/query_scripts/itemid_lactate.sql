-- Query for itemid code of Lactate and subcategories 
-- to be classified as - Lactate
CREATE TEMP TABLE itemid_lactate AS
SELECT *, 'lactate' AS lab_category
FROM d_labitems
WHERE label = 'Lactate'
   OR label LIKE 'Lactate %'
;


CREATE INDEX idx_itemid_lactate ON itemid_lactate(itemid);
