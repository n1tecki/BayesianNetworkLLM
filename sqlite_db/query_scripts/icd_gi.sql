-- Query for icd codes of Gastrointestinal (GI) Bleed and subcategories 
-- to be classified as - Gastrointestinal (GI) Bleed
CREATE TEMP TABLE icd_gi AS
SELECT *, 'gi' AS diagnoses_category
FROM d_icd_diagnoses
WHERE long_title LIKE '%Esophageal varices with bleeding%'
OR long_title LIKE '%Esophageal varices in diseases classified elsewhere, with bleeding%'
OR long_title LIKE '%Acute peptic ulcer of unspecified site with hemorrhage%'
OR long_title LIKE '%Acute peptic ulcer of unspecified site with hemorrhage and perforation%'
OR long_title LIKE '%Chronic or unspecified peptic ulcer of unspecified site with hemorrhage%'
OR long_title LIKE '%Chronic or unspecified peptic ulcer of unspecified site with hemorrhage and perforation%'
OR long_title LIKE '%Hematemesis%'
OR long_title LIKE '%Hemorrhage of gastrointestinal tract, unspecified%'
OR long_title LIKE '%Gastrointestinal hemorrhage of fetus or newborn%'
OR long_title LIKE '%Hematemesis and melena of newborn%'
OR long_title LIKE '%Acute peptic ulcer, site unspecified, with hemorrhage%'
OR long_title LIKE '%Acute peptic ulcer, site unspecified, with both hemorrhage and perforation%'
OR long_title LIKE '%Chronic or unspecified peptic ulcer, site unspecified, with hemorrhage%'
OR long_title LIKE '%Chronic or unspecified peptic ulcer, site unspecified, with both hemorrhage and perforation%'
OR long_title LIKE '%Vomiting following gastrointestinal surgery%'  -- Only if you consider “hematemesis” plus “vomiting after GI surgery” relevant to bleed.
OR long_title LIKE '%K920 Hematemesis%'  -- Or more specifically: '%Hematemesis%' above
OR long_title LIKE '%K921 Melena%'
OR long_title LIKE '%K922 Gastrointestinal hemorrhage, unspecified%'
OR long_title LIKE '%Neonatal hematemesis%'
OR long_title LIKE '%Neonatal melena%'
OR long_title LIKE '%Other neonatal gastrointestinal hemorrhage%'
OR long_title LIKE '%Neonatal hematemesis and melena due to swallowed maternal blood%'
OR long_title LIKE '%Esophageal varices%'
    AND long_title LIKE '%with bleeding%'  -- Some ICD-10 lines show “I8501” or “I8511”
;


CREATE INDEX idx_icd_gi ON icd_gi(icd_code);