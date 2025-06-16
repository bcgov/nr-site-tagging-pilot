import csv
from pathlib import Path

_log_headers = []


def init_log(filepath: Path, headers: list):
    """
    Initializes a CSV log file with specified column headers.
    Creates parent directories if they do not exist.

    Parameters:
    ----------
    filepath : pathlib.Path
        Path to the CSV log file to be created or initialized.
    headers : list of str
        List of column headers for the log file.

    Returns:
    -------
    None
    """
    global _log_headers
    _log_headers = headers
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def log_metadata(filepath: Path, row: dict):
    """
    Appends a row of metadata to the existing CSV log file using the original header order.
    Missing keys are filled with an empty string.

    Parameters:
    ----------
    filepath : pathlib.Path
        Path to the existing CSV log file.
    row : dict
        Dictionary of values to be written as a row. Keys should correspond to column headers.

    Returns:
    -------
    None
    """
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col, "") for col in _log_headers])


def update_log_row(log_path: Path, original_filename: str, updated_values: dict):
    """
    Overwrites a row in the CSV log where 'Original_Filename' matches the given name.
    Only replaces the values provided in `updated_values`.

    Parameters:
    ----------
    log_path : pathlib.Path
        Path to the existing CSV log file.
    original_filename : str
        The name of the original file whose log entry should be updated.
    updated_values : dict
        Dictionary of column names and new values to update in the matched row.

    Returns:
    -------
    None
    """
    temp_rows = []
    found = False

    with open(log_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["Original_Filename"] == original_filename:
                row.update(updated_values)
                found = True
            temp_rows.append(row)

    if not found:
        print(
            f"[WARNING] Could not find {original_filename} in metadata log to update.")

    with open(log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(temp_rows)
