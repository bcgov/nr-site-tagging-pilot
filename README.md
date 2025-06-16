# BC Environment Capstone: Document Classification Pipeline

This project automates the classification, renaming, and metadata extraction of scanned site remediation documents (PDFs) provided by the BC Ministry of Environment.

---

For each PDF in the input folder:

1. Extract the **Site ID** from the filename using regex, or fall back to **LLM (Mistral via Ollama)**.
2. Use a **document classifier** to assign a type (e.g., REPORT, PSI, CORR):
   - Default: regex-based keyword matcher
   - Optional: ML-based BERT model (Hugging Face)
3. Generate a standardized filename:  
   `YYYY-MM-DD – SITE_ID – TYPE.pdf`
4. Copy the renamed file into `data/output/{TYPE}/`
5. Log extracted metadata to `logs/metadata_log.csv`
6. Optionally compare results to a gold-standard metadata CSV for validation

---

## Setup

- Clone nr-site-tagging-pilot
```git clone https://github.com/bcgov/nr-site-tagging-pilot.git```

- Download project_files.zip file from teams channel.

- Unzip the file.

After unzipping, the folder would look like this:

```
project_files/
├── data/    ← Copy this folder to CAPSTONE/
├── models/  ← Copy this folder to CAPSTONE/nr-site-tagging-pilot/
```

## Folder Setup

Your folder should look like this:

### **Directory Structure**

```plaintext
CAPSTONE/
├── data/
│   ├── input/                                      ← Raw PDFs go here
│   ├── output/                                     ← Renamed + organized PDFs will go here
│   ├── evaluation/
│   │   ├── evaluation_log.csv                      ← Log of all processed files
│   │   └── output/                                 ← Renamed + organized gold PDFs will go here
│   ├── logs/                                       ← CSV log will go here
│   ├── gold_files/                                 ← Gold standard PDFs for validation
│   └── lookups/                                    ← Required lookup/reference files
│       └── site_registry_mapping.xlsx              ← REQUIRED: Document type to release eligibility mapping
│       └── site_ids.csv                            ← REQUIRED: Site ID to Address mapping
│       └── clean_metadata.csv                      ← REQUIRED: Gold standard metadata for validation
├── nr-site-tagging-pilot/
│   ├── main.py
│   ├── evaluate.py
│   ├── config.py
│   ├── requirements.txt
│   ├── README.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── LICENSE.txt
│   ├── prompts/
│   │   └── metadata_prompt.txt                     ← LLM prompt template
│   │   └── address_reprompt.txt
│   │   └── site_id_reprompt.txt
│   ├── models/ ← Save model here, we will share this with team
│   │   └── document_classification_model ← ensure same name
│   └── utils/                   ← Helper scripts (see internal README)

```

---

REQUIRED LOOKUP FILES

The following file must be placed inside the data/lookups/ directory **before running the pipeline**:

1. site_registry_mapping.xlsx
   - Purpose: Determines whether a document is publicly releasable based on its type. Avaliable on the Teams channel.
   - Columns:
     • Document_Type  
     • Site_Registry_Releaseable (values must be "yes" or "no")

2. site_ids.csv  
   - Purpose: Contains site id to address mapping.  

This file is mandatory. The pipeline will not run if it is missing. Avaliable on the Teams channel.

3. clean_metadata.csv  
   - Purpose: Contains the gold standard metadata for validation. Avaliable on the Teams channel.

---

## How to Run

1. **Install dependencies**  
`pip install -r requirements.txt`

2. **Download ollama and install the .exe file** from following <https://ollama.com/download/>

3. **Install Models**

`ollama pull mistral`

4. **Run Mistral model**  
`ollama run mistral`

5. **Run the pipeline**  
`python main.py`

---

## Evaulation Mode

To evaluate the accuracy of the pipeline:

1. **Place gold test PDFs in data/gold_files/**

2. **Ensure gold metadata file exists at data/lookups/clean_metadata.csv**

3. **Run:**
`python evaluate.py`

This will:

- Run the pipeline using gold input files
- Save processed files in data/evaluation/output/
- Log output metadata in data/evaluation/evaluation_log.csv
- Compute and print F1, precision and recall scores for:
  - Duplicate detection (yes vs no)
  - Site Registry Releasable (yes vs no)

Evaluation output is self-contained and does not modify your main output/logs folders.

## Output

Each processed PDF will appear in `outputs/Site_ID/YYYY-DOC_TYPE/` with a new standardized filename. A CSV log is created at `logs/metadata_log.csv` during normal runs, and at `evaluation/evaluation_log.csv` during evaluation mode. The log contains the following fields:

- Original_Filename            → Name of the original input PDF
- New_Filename                 → Standardized renamed file
- Site_id                      → 4-digit Site ID (from filename or LLM)
- Document_Type                → Classified type of the document
- Site_Registry_Releaseable    → "yes", "no" or "no (duplicate)", if ready for public release
- Title                        → Full document or report title
- Receiver                     → Person or organization the document is addressed to
- Sender                       → Person or organization that authored/submitted the document
- Address                      → Site or project location
- Duplicate                    → "yes" if flagged as a duplicate, otherwise "no"
- Duplicate_File               → duplicate filename if flagged as a duplicate, otherwise ""
- Similarity_Score             → duplicate similarity score if flagged as a duplicate, otherwise ""
- Readable                     → "yes" if the document was mostly readable, otherwise "no"
- Output_Path                  → Full path to the renamed file in the output directory

---

## Notes

- Raw PDFs are never modified
- The LLM is only used when metadata is incomplete or ambiguous
- Document types are inferred using keyword matching
- To use the ML classifier, set USE_ML_CLASSIFIER = True in main.py

This is an early prototype to demonstrate pipeline automation and reproducibility.
