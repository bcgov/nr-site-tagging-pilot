from pathlib import Path

# Base data directory
PDF_DATA_PATH = Path(__file__).resolve().parents[1] / "data"

# Standard pipeline paths
INPUT_DIR = PDF_DATA_PATH / "input"
OUTPUT_DIR = PDF_DATA_PATH / "output"
LOG_PATH = PDF_DATA_PATH / "logs" / "metadata_log.csv"
LOOKUPS_PATH = PDF_DATA_PATH / "lookups"

# Paths for evaluation
EVALUATION_DIR = PDF_DATA_PATH / "evaluation"
GOLD_FILES_DIR = PDF_DATA_PATH / "gold_files"
GOLD_METADATA_PATH = LOOKUPS_PATH / "clean_metadata.csv"