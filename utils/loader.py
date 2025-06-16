from pathlib import Path
import os
import fitz  # PyMuPDF
import re

def load_pdfs(pdf_dir: Path):
    """
    Returns a sorted list of all PDF files in the specified directory.

    Parameters:
        pdf_dir (Path): Directory path where PDF files are located.

    Returns:
        list[Path]: List of PDF file paths sorted alphabetically.
    """
    return sorted([file for file in pdf_dir.glob("*.pdf") if file.is_file()])


def extract_text_from_pdf(pdf_path, max_pages=5):
    """
    Extracts text from the first few pages of a PDF file using PyMuPDF.

    Parameters:
        pdf_path (Path): Path to the PDF file.
        max_pages (int): Maximum number of pages to extract text from (default: 5).

    Returns:
        str: Concatenated text from the specified number of pages.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc[:max_pages]:
        text += page.get_text()
    doc.close()
    return text


def clean_ocr_text(text):
    """
    Cleans up raw OCR-extracted text by removing unwanted characters and excess whitespace.

    Parameters:
        text (str): Raw text extracted from a PDF document.

    Returns:
        str: Cleaned and normalized text suitable for LLM prompting.
    """
    text = text.replace("\n", " ")
    text = re.sub(r'[^a-zA-Z0-9\s:,\-./]', '', text)  
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()
