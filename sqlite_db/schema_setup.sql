-- Patients table: Contains demographic details about patients
CREATE TABLE IF NOT EXISTS patients (
    subject_id INTEGER PRIMARY KEY,  -- Unique patient identifier
    gender TEXT NOT NULL,  -- Gender of the patient
    anchor_age INTEGER NOT NULL,  -- Age at the anchor time
    anchor_year INTEGER NOT NULL,  -- Year of anchor date
    anchor_year_group TEXT NOT NULL,  -- Time period grouping for the anchor year
    dod TEXT  -- Date of death (nullable)
);

-- Admissions table: Records each hospital admission for patients
CREATE TABLE IF NOT EXISTS admissions (
    hadm_id INTEGER PRIMARY KEY,  -- Unique hospital admission identifier
    subject_id INTEGER NOT NULL,  -- Links to patients
    admittime TIMESTAMP NOT NULL,  -- Admission timestamp
    dischtime TIMESTAMP,  -- Discharge timestamp
    deathtime TIMESTAMP,  -- Death timestamp if applicable
    admission_type TEXT NOT NULL,  -- Type of admission (e.g., emergency, elective)
    admission_location TEXT,  -- Source of admission
    discharge_location TEXT,  -- Discharge destination
    insurance TEXT,  -- Insurance type
    language TEXT,  -- Patient's preferred language
    marital_status TEXT,  -- Marital status
    ethnicity TEXT,  -- Ethnicity of the patient
    edregtime TIMESTAMP,  -- Emergency department registration time
    edouttime TIMESTAMP,  -- Emergency department discharge time
    hospital_expire_flag INTEGER NOT NULL,  -- 1 if patient died in hospital, else 0
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id)
);

-- Diagnoses table: Links patients with their diagnoses
CREATE TABLE IF NOT EXISTS diagnoses_icd (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique row identifier
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER NOT NULL,  -- Links to admissions
    seq_num INTEGER,  -- Order of diagnosis
    icd_code TEXT NOT NULL,  -- Diagnosis code
    icd_version INTEGER NOT NULL,  -- ICD version (9 or 10)
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY (icd_code) REFERENCES d_icd_diagnoses(icd_code)
);

-- ICD Diagnoses reference table: Contains descriptions of ICD diagnosis codes
CREATE TABLE IF NOT EXISTS d_icd_diagnoses (
    icd_code TEXT PRIMARY KEY,  -- ICD diagnosis code
    icd_version INTEGER NOT NULL,  -- ICD version (9 or 10)
    long_title TEXT NOT NULL  -- Description of the diagnosis
);

-- ICD Procedures reference table: Contains descriptions of ICD procedure codes
CREATE TABLE IF NOT EXISTS d_icd_procedures (
    icd_code TEXT PRIMARY KEY,  -- ICD procedure code
    icd_version INTEGER NOT NULL,  -- ICD version (9 or 10)
    long_title TEXT NOT NULL  -- Description of the procedure
);

-- Procedures table: Links patients with procedures performed
CREATE TABLE IF NOT EXISTS procedures_icd (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique row identifier
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER NOT NULL,  -- Links to admissions
    seq_num INTEGER,  -- Order of procedure
    icd_code TEXT NOT NULL,  -- Procedure code
    icd_version INTEGER NOT NULL,  -- ICD version (9 or 10)
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY (icd_code) REFERENCES d_icd_procedures(icd_code)
);

-- Lab Items reference table: Describes lab tests
CREATE TABLE IF NOT EXISTS d_labitems (
    itemid INTEGER PRIMARY KEY,  -- Unique identifier for the lab test
    label TEXT NOT NULL,  -- Name of the lab test
    fluid TEXT,  -- Type of fluid tested (e.g., blood, urine)
    category TEXT  -- Lab test category (e.g., chemistry, hematology)
);

-- Lab Events table: Stores lab test results for patients
CREATE TABLE IF NOT EXISTS labevents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique row identifier
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER,  -- Links to admissions (nullable for outpatient labs)
    itemid INTEGER NOT NULL,  -- Links to d_labitems
    charttime TIMESTAMP NOT NULL,  -- Time lab was recorded
    value TEXT,  -- Lab result value
    valuenum REAL,  -- Numeric lab result (if applicable)
    valueuom TEXT,  -- Units of measurement
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id),
    FOREIGN KEY (itemid) REFERENCES d_labitems(itemid)
);

-- EMAR table: Stores medication administration records
CREATE TABLE IF NOT EXISTS emar (
    emar_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier for each administration event
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER,  -- Links to admissions (nullable for outpatient medications)
    charttime TIMESTAMP,  -- Time medication was administered
    medication TEXT NOT NULL,  -- Name of the medication
    medication_name_generic TEXT NOT NULL,  -- Standardized generic name of the medication
    charted_dose REAL,  -- Administered dose
    charted_dose_units TEXT,  -- Units of the administered dose
    status TEXT NOT NULL,  -- Status of administration (Given, Not Given, Held)
    reason TEXT,  -- Reason for administration (or why it was held/not given)
    comments TEXT,  -- Additional comments
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
);

-- EMAR Detail table: Additional details for medication administration
CREATE TABLE IF NOT EXISTS emar_detail (
    emar_detail_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier
    emar_id INTEGER NOT NULL,  -- Links to emar
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER,  -- Links to admissions
    route TEXT,  -- Route of administration (e.g., oral, IV)
    dose REAL,  -- Administered dose
    dose_units TEXT,  -- Units of dose
    infusion_rate TEXT,  -- Infusion rate for IV medications
    FOREIGN KEY (emar_id) REFERENCES emar(emar_id),
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
);

-- Prescriptions table: Stores medication orders
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER,  -- Links to admissions
    pharmacy_id INTEGER,  -- Links to pharmacy (nullable for external meds)
    startdate TIMESTAMP,  -- Start date of prescription
    enddate TIMESTAMP,  -- End date of prescription
    drug TEXT NOT NULL,  -- Medication name
    drug_name_generic TEXT,  -- Generic medication name
    formulary_drug_cd TEXT,  -- Formulary drug code
    dose_val_rx REAL,  -- Prescribed dose
    dose_unit_rx TEXT,  -- Dose unit
    route TEXT,  -- Route of administration
    status TEXT,  -- Prescription status
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
);

-- Transfers table: Stores patient movements within the hospital
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Unique identifier for transfer event
    subject_id INTEGER NOT NULL,  -- Links to patients
    hadm_id INTEGER NOT NULL,  -- Links to admissions
    eventtype TEXT,  -- Type of transfer event
    careunit TEXT,  -- Care unit transferred to
    intime TIMESTAMP,  -- Time of transfer in
    outtime TIMESTAMP,  -- Time of transfer out
    FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
    FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
);
