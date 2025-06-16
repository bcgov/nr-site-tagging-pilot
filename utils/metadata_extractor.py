import re
import os
import string
from pathlib import Path
from .loader import extract_text_from_pdf, clean_ocr_text
from rouge_score import rouge_scorer
import pandas as pd
import sys
import fitz
from rapidfuzz import fuzz

def extract_site_id_from_filename(filename):
    """
    Extracts a numeric site ID from the start of a filename using regex.
    
    Parameters:
        filename (str): The name of the file to extract the site ID from.
    
    Returns:
        str or None: A 3- to 5-digit site ID if matched; otherwise, None.
    """
    match = re.match(r"^(\d{3,5})\b", filename)
    if match:
        return match.group(1)
    return None


def extract_metadata(file_path):
    """
    Returns placeholder metadata for testing or debugging purposes.

    Parameters:
        file_path (Path): Path to the input file (used for display only).

    Returns:
        dict: A dictionary containing dummy values for site_id, address, sender, and receiver.
    """
    metadata = {
        "site_id": "12345",
        "address": "N/A",
        "sender": "Unknown",
        "receiver": "Unknown"
    }
    print(f"[Metadata] Extracted from {file_path.name}: {metadata}")
    return metadata

_punct = str.maketrans("", "", string.punctuation)
_space = re.compile(r"\s+")

def _clean(txt: str) -> str:
    return _space.sub(" ", txt.lower().translate(_punct)).strip()

# def _best_window_min_f1(shorter_pages, longer_pages, scorer) -> float:
#     best = 0.0
#     for i in range(len(longer_pages) - len(shorter_pages) + 1):
#         window = longer_pages[i:i+len(shorter_pages)]
#         worst = min(
#             scorer.score(a, b)["rouge1"].fmeasure
#             for a, b in zip(shorter_pages, window)
#         )
#         best = max(best, worst)
#     return best

def check_duplicate_by_rouge(
    current_file_path: Path,
    site_id: str,
    site_id_dir: Path,
    rouge_th: float = 0.75,
    rapid_th: float = 78.0,
    rouge_metric: str = "rouge1"
) -> tuple[str, Path | None, bool, float]:
    """
    Two-step duplicate detector comparing full document text using ROUGE and RapidFuzz.

    Returns:
        (duplicate_status, matched_file_path, is_current_file_shorter, similarity_score)
    """
    scorer = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)

    try:
        cur_doc = fitz.open(current_file_path)
        cur_text = " ".join([clean_ocr_text(page.get_text()) for page in cur_doc])
        cur_doc.close()
    except Exception:
        return "no", None, False, 0.0

    if not site_id_dir.exists():
        return "no", None, False, 0.0

    for root, _, files in os.walk(site_id_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
            if site_id and str(site_id) not in file:
                continue

            cand_path = Path(root) / file
            if cand_path.resolve() == current_file_path.resolve():
                continue

            try:
                cand_doc = fitz.open(cand_path)
                cand_text = " ".join([clean_ocr_text(p.get_text()) for p in cand_doc])
                cand_doc.close()
            except Exception as e:
                print(f"[WARN] {cand_path.name}: {e}")
                continue

            is_current_file_shorter = len(cur_text) <= len(cand_text)

            # ROUGE Recall
            if is_current_file_shorter:
                recall_score = scorer.score(cand_text, cur_text)[rouge_metric].recall
            else:
                recall_score = scorer.score(cur_text, cand_text)[rouge_metric].recall

            if recall_score >= rouge_th:
                print(f"[CONTAINED by ROUGE] {file}")
                return "contained", cand_path, is_current_file_shorter, recall_score

            # RapidFuzz fallback
            rapid_score = fuzz.token_sort_ratio(cur_text, cand_text)
            if rapid_score >= rapid_th:
                print(f"[LIKELY DUPLICATE (OCR)] {file}")
                return "likely_duplicate_ocr", cand_path, is_current_file_shorter, rapid_score / 100.0

    return "no", None, False, 0.0


_release_df = None  # cache for loaded Excel

def get_site_registry_releasable(doc_type: str, lookup_file_path: str) -> str:
    """
    Looks up the Site Registry Releasable status for a given document type.
    If the Excel file is missing or no match is found, the program exits with a message.

    Parameters:
        doc_type (str): Document type (e.g., 'report')
        lookup_file_path (str): Path to Excel lookup file

    Returns:
        str: 'yes' or 'no' based on the lookup file
    """
    global _release_df

    try:
        # Load and cache Excel file if not already
        if _release_df is None:

            _release_df = pd.read_excel(lookup_file_path)
            _release_df['Document_Type'] = _release_df['Document_Type'].astype(str).str.lower()
            _release_df['Site_Registry_Releaseable'] = _release_df['Site_Registry_Releaseable'].astype(str).str.lower().str.strip()

        doc_type = doc_type.lower().strip()

        # Partial (contains) match
        for _, row in _release_df.iterrows():
            if doc_type == row['Document_Type']:
                return row['Site_Registry_Releaseable']

        # No match found — exit safely
        print(f"[ERROR] Document type '{doc_type}' not found in site registry mapping.")
        sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed to check site registry releasable: {e}")
        sys.exit(1)


