-- SQLite


-- All patients diagnosis based on common diagnosis
SELECT di.subject_id, di.hadm_id, di.seq_num, di.icd_code, dd.long_title
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0200'
);




-- Number of patients with given illness
SELECT COUNT(DISTINCT di.subject_id)
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0030'
);



-- Number of total stays for given illness
SELECT COUNT(DISTINCT di.hadm_id)
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0030'
);



-- Count of all distinct diagnosis in the given search
SELECT COUNT(DISTINCT di.icd_code)
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0030'
);



-- Numbers of diagnoses per stay
SELECT di.hadm_id, COUNT(DISTINCT di.icd_code)
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0030'
)
GROUP BY di.hadm_id
ORDER BY COUNT(DISTINCT di.icd_code) DESC;



-- Number of stays per patient
SELECT di.subject_id, COUNT(DISTINCT di.hadm_id)
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0030'
)
GROUP BY di.subject_id
ORDER BY COUNT(DISTINCT di.hadm_id) DESC;



-- Count of unique diagnoses
SELECT di.icd_code, COUNT(di.icd_code), dd.long_title
FROM diagnoses_icd di
JOIN d_icd_diagnoses dd
  ON di.icd_code = dd.icd_code
WHERE di.hadm_id IN (
    SELECT hadm_id
    FROM diagnoses_icd
    WHERE icd_code = '0030'
)
GROUP BY di.icd_code
ORDER BY COUNT(di.icd_code) DESC;