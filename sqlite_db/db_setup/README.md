# Building the MIMIC database with SQLite

Either `db_setup.sh` or `db_setup.py` can be used to generate a [SQLite](https://sqlite.org/index.html) database file from the MIMIC-IV demo or full dataset.

`db_setup.sh` is a shell script that will work with any POSIX compliant shell.
It is memory efficient and does not require loading entire data files
into memory. It only needs three things to run:

1. A POSIX compliant shell (e.g., dash, bash, zsh, ksh, etc.)
2. [SQLite]([https://sqlite.org/index.html)
3. gzip (which is installed by default on any Linux/BSD/Mac variant)

**Note:** The `db_setup.sh` script will set all data fields to *text*.

`db_setup.py` is a python script. It requires the following to run:

1. Python 3 installed
2. [pandas](https://pandas.pydata.org/)

## Step 1: Download the CSV or CSV.GZ files.

- Download the MIMIC-IV dataset from: https://physionet.org/content/mimiciv/
- Place `db_setup.sh` or `db_setup.py` into the same folder as the `csv` or `csv.gz` files

i.e. your folder structure should resemble:

```
path/to/mimic-iv/
├── db_setup.sh
├── db_setup.py
├── hosp
│   ├── admissions.csv.gz
│   ├── ...
│   └── transfers.csv.gz
└── icu
    ├── chartevents.csv.gz
    ├── ...
    └── procedureevents.csv.gz
```

## Step 2: Generate the SQLite file

To generate the SQLite file:

If you are using `db_setup.sh`, run on the command-line:

```
$ ./db_setup.sh
```

If you are using `db_setup.py`, run on the command-line:

```
$ python db_setup.py
```

If loading the full dataset, this will take some time,
particularly the `CHARTEVENTS` table.

The scripts will ultimately generate an SQLite database file called `mimic4.db`.