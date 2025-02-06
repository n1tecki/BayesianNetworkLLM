-- Query for itemid code of Platelets Count and subcategories 
-- to be classified as - Platelets
CREATE TEMP TABLE itemid_platelets AS
SELECT *, 'platelets' AS lab_category
FROM d_labitems
WHERE label LIKE '%Platelet%'
   OR label LIKE '%Platelets%'
   OR label LIKE '%Platelet Count%'
   OR label LIKE '%Platelet Clumps%'
   OR label LIKE '%Platelet Smear%'
   OR label LIKE '%Antiplatelet%'
   OR label LIKE '%Platelet Aggregation%'
   OR label LIKE '%PltDist%'
   OR label LIKE '%PltScat%'
   OR label LIKE '%PltClmp%'
   OR label LIKE '%Mean Platelet Volume%'
;




CREATE INDEX idx_itemid_platelets ON itemid_platelets(itemid);