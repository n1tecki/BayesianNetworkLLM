-- Balanse the table for all diagnoses categories to have the same occurence
CREATE TABLE balanced_matched_admission_labevents AS
WITH min_group_size AS (
    SELECT MIN(category_count) AS min_count
    FROM (
        SELECT diagnoses_category, COUNT(*) AS category_count
        FROM matched_admission_labevents
        GROUP BY diagnoses_category
    ) AS counts
),
numbered AS (
    SELECT 
        a.*,
        ROW_NUMBER() OVER (
            PARTITION BY a.diagnoses_category 
            ORDER BY a.subject_id
        ) AS rn
    FROM matched_admission_labevents a
)
SELECT n.*
FROM numbered n
JOIN min_group_size m ON 1=1
WHERE n.rn <= m.min_count;