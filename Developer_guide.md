
BC Environment Capstone: Document Classification Pipeline Code Documentation
=========================================================

Overview:
---------
This project automates the extraction, classification, renaming, and logging of metadata from environmental PDF documents. It is tailored for scanned remediation documents from the BC Ministry of Environment. The system supports a hybrid rule-based + ML/LLM pipeline, involving both classical ML and LLM-based metadata extraction using Mistral through Ollama.

Core Capabilities:
------------------
1. Extract Site ID from filename or LLM fallback.
2. Extract document metadata (title, sender, receiver, address, readable).
3. Classify document type using:
   - Regex-based rule matching.
   - ML-based classifier (HuggingFace model).
4. Detect duplicate documents using ROUGE and RapidFuzz similarity.
5. Determine if document is releasable based on Excel mapping.
6. Rename document using standard format.
7. Copy file to appropriate folder and log metadata.
8. Support evaluation mode comparing predictions to gold metadata.

MAIN SCRIPT - main.py:
=======================

The main orchestration script. It contains the `main()` function and a core function `process_file()` that processes each PDF.

Key Functions:
--------------

main():
- Loads ML model if available.
- Validates required files and directories.
- Loads all PDFs in input folder.
- Initializes logs.
- Iterates through each file, calling `process_file()`.

process_file():
- Extracts site ID from filename or queries LLM.
- Extracts OCR-cleaned text (first 8 pages max).
- If text is unreadable (<50 words), flags document.
- Prompts LLM to extract metadata (title, sender, etc).
- Re-prompts fields using `validate_and_reprompt_field()` if malformed.
- Validates site ID, optionally queries again.
- Retrieves address from CSV mapping, falling back to LLM or cached value.
- Uses regex or ML model to classify document type.
- Detects duplicates using `check_duplicate_by_rouge()`.
- Determines if file is releasable using `get_site_registry_releasable()`.
- Generates standardized filename using `generate_new_filename()`.
- Copies file to destination using `organize_files()`.
- Logs metadata to CSV using `log_metadata()`.

UTILITIES & SUPPORT MODULES:
============================

utils/classifier.py:
--------------------
- classify_document(): delegates to regex or ML classifier.
- classify_with_regex(): uses keyword matching from filename.
- classify_with_ml(): uses HuggingFace transformer to classify document titles.
- load_huggingface_model(): loads model and tokenizer.

utils/llm_interface.py:
-----------------------
- query_llm(): uses Ollama to prompt Mistral and extract metadata.
- llm_single_field_query(): prompts for single metadata field.
- validate_and_reprompt_field(): re-prompts LLM to extract well-formed field.
- field_is_well_formed(), all_words_in_text(): validate hallucination by cross-checking against document content.
- load_prompt_template(): loads and formats LLM prompt file.

utils/metadata_extractor.py:
----------------------------
- extract_site_id_from_filename(): regex to extract site ID from filename.
- check_duplicate_by_rouge(): compares full document text to others in output folder using ROUGE and RapidFuzz.
- get_site_registry_releasable(): checks Excel mapping to determine public release eligibility.

utils/rename.py:
----------------
- generate_new_filename(): constructs standardized filename; adds `-DUP` suffix if needed; handles name collision.

utils/file_organizer.py:
------------------------
- organize_files(): copies files to structured folder path.

utils/logger.py:
----------------
- init_log(): initializes CSV metadata log.
- log_metadata(): appends single row to log.
- update_log_row(): updates specific row in log based on original filename.

utils/loader.py:
----------------
- load_pdfs(): retrieves all PDF paths.
- extract_text_from_pdf(): uses PyMuPDF to extract text.
- clean_ocr_text(): cleans and strips OCR noise for LLM usage.

utils/checks.py:
----------------
- verify_required_files(): ensures all required reference files exist.
- verify_required_dirs(): ensures folders exist before starting pipeline.

utils/gold_data_extraction.py:
------------------------------
- loading_gold_metadata_csv(): loads clean metadata with specific header offset.
- load_gold_data(): fetches gold metadata for a specific file.

utils/site_id_to_address.py:
----------------------------
- get_site_address(): returns clean address based on site ID.
- format_address(), clean_address(): deduplication, reformatting, and fuzzy logic.

EVALUATION SCRIPT - evaluate.py:
================================

evaluate.py evaluates the accuracy of the full pipeline using gold labels.

Key Functions:
--------------

files_preparation():
- Clears all contents of evaluation directory except `output/` subfolder.

load_evaluation_dataframe():
- Loads and merges predicted + gold CSVs using filenames.
- Normalizes column names, trims whitespace, converts labels to common format.
- Handles mismatched casing, missing values, prefix stripping.

compute_row_rouge_recalls():
- Applies ROUGE-1 recall for all text fields row-wise.

compute_scores():
- Calculates recall, precision, F1 for:
    - Duplicate detection (yes/no)
    - Site Registry Releasable (yes/no)
- Computes ROUGE-1 recall for:
    - Title, Sender, Receiver, Address
- Saves row-level and summary CSVs to evaluation folder.

Command-line arguments:
-----------------------
--use-test-metadata: uses alternate test CSV (test_metadata.csv).

RUNNING THE PIPELINE:
=====================

Normal Mode:
------------
> python main.py

Evaluation Mode:
----------------
> python evaluate.py

With test metadata:
> python evaluate.py --use-test-metadata

OUTPUT STRUCTURE:
=================

```plaintext
data/
├── input/                  ← Raw PDFs
├── output/                 ← Structured PDFs organized by site_id + type
├── logs/                   ← Log of run (metadata_log.csv)
├── lookups/                ← Required reference data
│   ├── site_ids.csv
│   ├── site_registry_mapping.xlsx
│   └── clean_metadata.csv
├── gold_files/             ← Files used for eval mode
├── evaluation/             ← Temp eval results
│   ├── output/             ← Structured PDFs organized by site_id + type of Test dataset
│   ├── evaluation_log.csv
│   ├── evaluation_merged_output.csv
│   └── evaluation_summary_metrics.csv
```

NOTES:
------
- LLM (Mistral via Ollama) is used only for fallback and metadata uncertainty.
- All metadata fields have fallback: site ID via regex or LLM; address via CSV or cached dict.
- Classification model (BERT) is optional and fallback is regex.
- Duplicate detection is conservative: relies on ROUGE and RapidFuzz.
- Files flagged for review have uncertain or unverifiable fields.
- Gold metadata must match filenames exactly for evaluation to work.
