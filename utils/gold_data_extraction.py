import pandas as pd


def loading_gold_metadata_csv(csv_path):
    """
    Loads the gold metadata CSV file containing annotated document information.

    Parameters:
    ----------
    csv_path : str or pathlib.Path
        Path to the gold metadata CSV file.

    Returns:
    -------
    pandas.DataFrame
        DataFrame containing the loaded gold metadata.
    """

    df = pd.read_csv(csv_path, header=3, encoding='ISO-8859-1')

    return df


def load_gold_data(file_path, csv_path):
    """
    Loads gold-standard metadata for a given file based on its name.

    This function reads a metadata CSV (typically annotated manually),
    filters it to find a row matching the input file name, and returns
    a dictionary of ground truth metadata fields.

    Parameters:
    ----------
    file_path : str
        The name of the file to look up in the gold data.
    csv_path : str or Path
        Path to the CSV file containing annotated gold metadata.

    Returns:
    -------
    dict or str
        A dictionary containing gold metadata fields if found, otherwise
        a string message indicating that no match was found.
    """
    # df = pd.read_csv(csv_path, encoding='windows-1252', header=3)
    df = pd.read_csv(csv_path, header=3, encoding='ISO-8859-1')
    match = df[df['Current BC Mail title'] == file_path]

    if match.empty:
        return f"No matching entry for '{file_path}' in gold data."

    row = match.iloc[0]

    metadata = {
        'title': str(row.get('Title/Subject', '')),
        'receiver': str(row.get('Receiver', '')),
        'sender': str(row.get('Sender/Author', '')),
        'address': str(row.get('Address', '')),
        'site_id': str(row.get('Site ID', ''))
    }

    return metadata
