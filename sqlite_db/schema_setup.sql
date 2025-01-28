-- Contains information about patient hospital admissions
CREATE TABLE IF NOT EXISTS admissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT,  -- Unique patient identifier
    hadm_id TEXT,  -- Unique hospital admission identifier
    admittime TEXT,  -- Admission timestamp
    dischtime TEXT,  -- Discharge timestamp
    admission_type TEXT  -- Type of admission (e.g., emergency, elective)
);

-- Contains ICD diagnosis codes and their descriptions
CREATE TABLE IF NOT EXISTS d_icd_diagnoses (
    icd_code TEXT PRIMARY KEY,  -- ICD diagnosis code
    icd_version INTEGER,  -- ICD version (9 or 10)
    long_title TEXT  -- Description of the diagnosis
);

-- Contains ICD procedure codes and their descriptions
CREATE TABLE IF NOT EXISTS d_icd_procedures (
    icd_code TEXT PRIMARY KEY,  -- ICD procedure code
    icd_version INTEGER,  -- ICD version (9 or 10)
    long_title TEXT  -- Description of the procedure
);

-- Contains lab test items with details about the fluid and category
CREATE TABLE IF NOT EXISTS d_labitems (
    itemid TEXT PRIMARY KEY,  -- Unique identifier for the lab test
    label TEXT,  -- Name of the lab test
    fluid TEXT,  -- Type of fluid tested (e.g., blood, urine)
    category TEXT  -- Lab test category (e.g., chemistry, hematology)
);

-- Stores ICD diagnoses assigned to patients during their hospital stay
CREATE TABLE IF NOT EXISTS diagnoses_icd (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT,  -- Unique patient identifier
    seq_num INTEGER,  -- Sequence number of the diagnosis for the admission
    icd_code TEXT,  -- ICD diagnosis code
    icd_version INTEGER  -- ICD version (9 or 10)
);

-- Contains information about medication administration records
CREATE TABLE IF NOT EXISTS emar (
    subject_id TEXT,  -- Unique patient identifier (links to patients table)
    hadm_id TEXT,  -- Unique hospital admission identifier (links to admissions table)
    emar_id TEXT PRIMARY KEY,  -- Unique identifier for the medication administration event
    charttime TEXT,  -- Timestamp when the medication was administered (or recorded)
    medication TEXT,  -- Name of the medication given
    medication_name_generic TEXT,  -- Standardized generic name of the medication
    charted_dose TEXT,  -- Administered dose
    charted_dose_units TEXT,  -- Units of the administered dose (e.g., mg, mL, mcg)
    status TEXT,  -- Status of medication administration (Given, Not Given, Held)
    reason TEXT,  -- Reason for administration (or why it was held/not given)
    comments TEXT  -- Additional comments about the administration
);

