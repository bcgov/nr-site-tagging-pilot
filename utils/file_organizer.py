import shutil
from pathlib import Path


def organize_files(original_path: Path, output_path: Path):
    """
    Copies a file from its original location to a structured output path.
    Automatically creates the output directory if it doesn't exist.

    Parameters:
        original_path (Path): Full path to the source file.
        output_path (Path): Full destination path including new filename.

    Side Effects:
        Creates output directory if needed and copies the file.
        Prints a confirmation message to the console.

    Returns:
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(original_path, output_path)
    print(f"[Organize] Copied to {output_path}")
