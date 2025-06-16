import sys
from pathlib import Path


def verify_required_files(required_paths: list):
    """
    Checks for the existence of all required files.

    Parameters:
    ----------
    required_paths : list of pathlib.Path
        List of file paths that must exist for the pipeline to proceed.

    Returns:
    -------
    None
    """
    missing = [str(p) for p in required_paths if not p.exists()]

    if missing:
        print("\n[ERROR] The following required lookup file(s) are missing:\n")
        for path in missing:
            print(f"  ✗ {path}")
        print(
            "\n[ABORTING] Please ensure all required files exist before running the pipeline.\n")
        sys.exit(1)
    else:
        print(f"[OK] All {len(required_paths)} required file(s) found.\n")


def verify_required_dirs(required_dirs: list):
    """
    Checks for the existence of all required directories.

    Parameters:
    ----------
    required_dirs : list of pathlib.Path
        List of directory paths that must exist for the pipeline to proceed.

    Returns:
    -------
    None
    """
    missing = [str(d)
               for d in required_dirs if not d.exists() or not d.is_dir()]

    if missing:
        print("\n[ERROR] The following required folder(s) are missing:\n")
        for path in missing:
            print(f"  ✗ {path}")
        print(
            "\n[ABORTING] Please create the missing folders before running the pipeline.\n")
        sys.exit(1)
    else:
        print(f"[OK] All {len(required_dirs)} required folder(s) found.\n")
