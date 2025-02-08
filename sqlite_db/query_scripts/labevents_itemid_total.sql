-- Based on the itemid lab test codes, tha patients are selected from the labevents table.
DROP TABLE IF EXISTS labevents_itemid_total;

CREATE TABLE labevents_itemid_total AS 
SELECT labevent_id,
       subject_id,
       charttime,
       value,
       valuenum,
       valueuom,
       ref_range_lower,
       ref_range_upper,
       flag,
       priority,
       d.itemid,  
       label,
       lab_category
FROM labevents s
JOIN itemid_total d ON s.itemid = d.itemid;

CREATE INDEX idx_labevents_itemid_total ON labevents_itemid_total(subject_id);