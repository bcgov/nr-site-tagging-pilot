# utils/

This folder contains all modular helper scripts used by the main document classification pipeline.

---

## Module Descriptions

### `__init__.py`
- Marks this folder as a Python package. Can be left empty.

---

### `classifier.py`
- Classifies a document’s type (e.g., REPORT, PSI, CORR).
- Uses regex-based keyword matching on filenames and/or extracted text.
- May be extended in future versions to use trained models.

---

### `checks.py`
- Performs startup verification for required files and directories.
- `verify_required_files()`:
  - Ensures all required lookup files (e.g., Excel configs) exist before pipeline runs.
  - Fails fast with clear error messages if any file is missing.
- `verify_required_dirs()`:
  - Ensures necessary directories (e.g., `data/input/`, `data/output/`) are present.
  - Fails cleanly if folders are missing and provides actionable messaging.

---

### `file_organizer.py`
- Copies a file into `data/output/{SITE_ID}/{YEAR}-{DOC_TYPE}/` using its new standardized filename. Also supports writing to `data/evaluation/output/...` when running in evaluation mode.
- Automatically creates output directories if they don’t exist.
- Designed to keep output files separate from the input dataset and repository.

---

### `gold_data_extraction.py`
- Loads gold-standard metadata from `clean_metadata.csv`.
- Matches records using the filename (typically mapped to folder names).
- Used in the pipeline to compare predicted metadata to the official ground truth.

---

### `llm_interface.py`
- Interfaces with Ollama to send prompts to a local quantized LLaMA 2 or Mistral model.
- Used to extract metadata fields like site ID, title, sender, and address.
- Supports both structured (JSON) and single-field prompting.
- Includes re-prompt logic for fallback cases when key fields are missing.

---

### `loader.py`
- Loads `.pdf` files from the input directory (`data/input/`).
- Uses PyMuPDF (`fitz`) to extract and clean text content.
- Also includes OCR cleanup utilities to improve prompt readability.

---

### `logger.py`
- Logs structured metadata for each processed file to a target .csv log file.
- In normal mode, logs to `data/logs/metadata_log.csv`.
- In evaluation mode, logs to `data/evaluation/evaluation_log.csv`.
- Automatically initializes headers if the file doesn’t exist.
- Supports logging fields including:
  - Original and new filename
  - Site ID
  - Document type
  - Title, sender, receiver, address
  - Readability flag
  - Output file path
- Creates the log file and headers if missing.

---

### `metadata_extractor.py`
- Extracts the site ID from a filename using regex.
- Returns a 3–5 digit number if found at the start of the filename.
- Used as the first-pass method before querying the LLM for site ID.
- Provides duplicate detection using a two-step strategy:
  1.	Page-window ROUGE score (rouge1, rouge2, or rougeL)
  2.	Fallback to RapidFuzz token-sort ratio if ROUGE fails
  The function returns 3 values:
  - `duplicate_status`: "contained", "likely_duplicate_ocr", or "no"
  - `matched_file`: Path of the matched file
  - `is_current_file_shorter`: True if current file has fewer pages (used to decide which file is tagged -DUP)
  Searches for duplicates within all subfolders of the same Site ID.
  
- Searches across all output subfolders under the same site ID to catch misclassified duplicates.
- Supports configurable thresholding and ROUGE metric selection.
- Extracts document release eligibility by matching document types to a preloaded Excel lookup (site_registry_mapping.xlsx).

---

### `metadata_rules.py`
- **[Deprecated]** Placeholder for future rule-based metadata extraction.
- Not currently used in the active pipeline.

---

### `metadata_rules.py`
- **[Deprecated]** Placeholder for future rule-based metadata extraction.
- Not currently used in the active pipeline.

---

### `rename.py`
- Generates standardized output filenames in the format:
  `YYYY-MM-DD – SITE_ID – TYPE[-DUP][_n].pdf`
    - Appends -DUP if the file is a confirmed duplicate
    - Adds _n to resolve filename collisions
    - Also returns the year string so the output folder can be named `{YEAR}-{DOC_TYPE}`
- Site ID and document type are injected based on earlier steps.
- Does **not** modify or move the file itself — just returns the new name string.

---

### `site_id_to_address.py`
- Cleans and formats site addresses based on specific rules.
- Uses fuzzy matching to compare two address fields and retains the most informative one based on length and similarity.
- Extracts numbers from address fields to identify relevant details like street numbers or suite numbers.
- Handles cases where the second address field is redundant or contains information not found in the first field.
- Formats the address by combining the first address, second address (if needed), urban area, and postal code (if available).
- Supports configurable fuzzy matching threshold to control redundancy detection.