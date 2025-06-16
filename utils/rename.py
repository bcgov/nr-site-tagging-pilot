import re
from pathlib import Path

def generate_new_filename(file_path: Path, site_id: str = "UNKNOWN", doc_type: str = "REPORT", duplicate: bool = False, output_dir: Path = None) -> tuple[str, str]:
    """
    Generates a standardized filename in the format:
    'YYYY-MM-DD – SITE_ID – DOC_TYPE[-DUP][_n].pdf'

    Adds -DUP if the file is a confirmed duplicate.
    Adds a numbered suffix if a file with the same name already exists.

    Parameters:
        file_path (Path): Path to the original PDF file.
        site_id (str): Extracted or assigned site ID.
        doc_type (str): Classified document type.
        duplicate (bool): Whether the file is a confirmed duplicate.
        output_dir (Path, optional): If provided, checks for filename collisions in this directory.

    Returns:
        tuple[str, str]: (Unique filename, year string for subfolder construction)
    """
    filename = file_path.name

    date_match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", filename)
    if date_match:
        date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        year_str = date_match.group(1)
    else:
        date_str = "0000-00-00"
        year_str = "0000"

    suffix = "-DUP" if duplicate else ""
    base_name = f"{date_str} - {site_id} - {doc_type.upper()}{suffix}"
    ext = file_path.suffix
    final_name = f"{base_name}{ext}"

    if output_dir:
        counter = 1
        while (output_dir / final_name).exists():
            final_name = f"{base_name}_{counter}{ext}"
            counter += 1

    return final_name, year_str