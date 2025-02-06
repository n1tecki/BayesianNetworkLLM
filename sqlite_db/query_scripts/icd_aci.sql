-- Query for icd codes of ACI Acute Kidney Injury and subcategories 
-- to be classified as - ACI
CREATE TEMP TABLE icd_aci AS
SELECT *, 'aci' AS diagnoses_category
FROM d_icd_diagnoses
WHERE long_title LIKE '%Acute kidney failure with lesion of tubular necrosis%'
OR long_title LIKE '%Acute kidney failure with lesion of renal cortical necrosis%'
OR long_title LIKE '%Acute kidney failure with lesion of renal medullary [papillary] necrosis%'
OR long_title LIKE '%Acute kidney failure with other specified pathological lesion in kidney%'
OR long_title LIKE '%Acute kidney failure, unspecified%'
OR long_title LIKE '%Kidney failure following abortion and ectopic and molar pregnancies%'
OR long_title LIKE '%Acute kidney failure following labor and delivery, unspecified as to episode of care or not applicable%'
OR long_title LIKE '%Acute kidney failure following labor and delivery, delivered, with mention of postpartum complication%'
OR long_title LIKE '%Acute kidney failure following labor and delivery, postpartum condition or complication%'
OR long_title LIKE '%Acute kidney failure%'
OR long_title LIKE '%Acute kidney failure with tubular necrosis%'
OR long_title LIKE '%Acute kidney failure with acute cortical necrosis%'
OR long_title LIKE '%Acute kidney failure with medullary necrosis%'
OR long_title LIKE '%Other acute kidney failure%'
OR long_title LIKE '%Acute kidney failure, unspecified%'
OR long_title LIKE '%Unspecified kidney failure%'
OR long_title LIKE '%Postprocedural (acute) (chronic) kidney failure%'
OR long_title LIKE '%Postpartum acute kidney failure%'
;


CREATE INDEX idx_icd_aci ON icd_aci(icd_code);