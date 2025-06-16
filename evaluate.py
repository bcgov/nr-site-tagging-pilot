import shutil
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
from config import INPUT_DIR, OUTPUT_DIR, LOG_PATH, GOLD_FILES_DIR, GOLD_METADATA_PATH, EVALUATION_DIR
from main import main
from rouge_score import rouge_scorer
import os
from utils.checks import verify_required_dirs, verify_required_files
from utils.gold_data_extraction import loading_gold_metadata_csv
import config
import argparse


config.INPUT_DIR = config.GOLD_FILES_DIR
config.OUTPUT_DIR = config.EVALUATION_DIR / "output"
OUTPUT_DIR = config.OUTPUT_DIR
config.LOG_PATH = config.EVALUATION_DIR / "evaluation_log.csv"
LOG_PATH = config.LOG_PATH


def files_preparation():
    """
    Clears all contents in EVALUATION_DIR except the 'output' folder.
    Recreates necessary log directory structure if missing.

    Returns:
    -------
    None
    """

    # 🔐 Safety guard to avoid dangerous deletion
    if "evaluation" not in str(config.EVALUATION_DIR).lower():
        raise RuntimeError(
            f"Aborting: EVALUATION_DIR '{config.EVALUATION_DIR}' does not appear safe to wipe.")

    # Delete everything inside EVALUATION_DIR except 'output'
    if config.EVALUATION_DIR.exists():
        for item in config.EVALUATION_DIR.iterdir():
            if item.name == "output":
                continue  # Skip the output folder
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def normalize_columns(df, columns):
    """
    Lowercases and strips whitespace from gold and predicted versions of specified columns.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the columns to normalize.
    columns : list of str
        List of base column names. For each base name, the function processes 
        the '<column>_gold' and '<column>_pred' columns.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with normalized '_gold' and '_pred' columns.

    """
    for column in columns:
        df[column + "_gold"] = df[column + "_gold"].str.lower().str.strip()
        df[column + "_pred"] = df[column + "_pred"].str.lower().str.strip()

    return df


def remove_prefix_labels(df, label_columns):
    """
    Removes leading labels (e.g., 'receiver:', 'sender:') with optional whitespace and colon from both gold and predicted columns.
    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the labeled columns.
    label_columns : list of str
        List of base label names to process (e.g., ['receiver', 'sender']).

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with cleaned '_gold' and '_pred' label columns.
    """
    for column in label_columns:
        pattern = rf"^{column.lower()}\s*:\s*"  # e.g., "receiver\s*:\s*"
        for suffix in ["_gold", "_pred"]:
            full_col = column + suffix
            df[full_col] = df[full_col].str.replace(pattern, "", regex=True)

    return df


def load_evaluation_dataframe(gold_metadata_path=config.GOLD_METADATA_PATH):
    """
    Loads and merges gold and predicted metadata, normalizing relevant columns for evaluation.

    Returns:
        pd.DataFrame: Cleaned and merged evaluation DataFrame.
    """

    pred_df = pd.read_csv(LOG_PATH)
    gold_df = loading_gold_metadata_csv(gold_metadata_path)

    # Normalize filename column for alignment
    pred_df["Original_Filename"] = pred_df["Original_Filename"].str.strip()
    gold_df["Current BC Mail title"] = gold_df["Current BC Mail title"].str.strip()

    # rename so that column names are same for both pred and gold
    rename_dict = {
        "Duplicate  (Y/N)": "Duplicate",
        'Site Registry releaseable': 'Site_Registry_Releaseable',
        'Title/Subject': 'Title',
        'Sender/Author': 'Sender',
        'Document Type': 'Document_Type'
    }

    gold_df = gold_df.rename(columns=rename_dict)

    # Merge on original filename
    merged_df = pd.merge(
        gold_df,
        pred_df,
        left_on="Current BC Mail title",
        right_on="Original_Filename",
        suffixes=("_gold", "_pred"),
        how="inner"
    )

    # Normalize Duplicate_gold from 'Y'/'N' to 'yes'/'no'
    merged_df["Duplicate_gold"] = merged_df["Duplicate_gold"].astype(str).str.strip().str.upper().map({
        "Y": "yes",
        "N": "no"
    }).fillna("no")

    # Normalize Duplicate_pred from prediction strings to 'yes'/'no'
    merged_df["Duplicate_pred"] = merged_df["Duplicate_pred"].astype(str).str.strip().str.lower().map(
        lambda x: "yes" if x in ["contained",
                                 "likely_duplicate_ocr", "yes"] else "no"
    )

    # Normalize Site_Registry_Releaseable columns
    merged_df["Site_Registry_Releaseable_gold"] = merged_df["Site_Registry_Releaseable_gold"].astype(str).str.strip().str.upper().map({
        "Y": "yes",
        "N": "no",
        "N (DUPLICATE)": "no"
    }).fillna("no")

    merged_df["Site_Registry_Releaseable_pred"] = merged_df["Site_Registry_Releaseable_pred"].astype(str).str.strip().str.lower().map(
        lambda x: "yes" if x in ["yes", "y"] else "no"
    )

    # Normalize other string columns for ROUGE comparison
    merged_df = normalize_columns(
        merged_df, ["Title", "Sender", "Receiver", "Document_Type", "Address"]
    )

    # Removing prefix lables like "Receiver: " etc from gold and pred text
    merged_df = remove_prefix_labels(
        merged_df, ["Title", "Sender", "Receiver"]
    )

    return merged_df


def compute_row_rouge_recalls(row, col_pairs, scorer):
    """
    Calculates ROUGE-1 recall for each (gold, pred) text column pair in a row.

    Parameters:
    ----------
    row : pandas.Series
        A single row from a DataFrame, containing gold and predicted text columns.
    col_pairs : list of tuple(str, str)
        List of tuples, where each tuple is (gold_column_name, pred_column_name).
    scorer : rouge_score.rouge_scorer.RougeScorer
        A ROUGE scorer instance, typically initialized with `use_stemmer=True`.

    Returns:
    -------
    pandas.Series
        A Series containing recall scores for each attribute in the format '{attribute}_recall'.
    """
    results = {}
    for gold_col, pred_col in col_pairs:
        gold, pred = row[gold_col], row[pred_col]
        if isinstance(gold, str) and isinstance(pred, str):
            score = scorer.score(gold, pred)['rouge1'].recall
        else:
            score = 0.0
        attr_name = gold_col.replace('_gold', '')  # safer and clearer
        results[f"{attr_name}_recall"] = score
    return pd.Series(results)


def compute_scores(merged_df):
    """
    Compute ROUGE-1 recall for text fields and classification metrics (F1, precision, recall)
    for discrete labels. Saves two output files: a detailed row-level output and a summary.

    Parameters:
    ----------
    merged_df : pandas.DataFrame
        DataFrame containing side-by-side gold and predicted columns, e.g., 
        'Title_gold', 'Title_pred', 'Duplicate_gold', 'Duplicate_pred', etc.

    Returns:
    -------
    None
        Writes results to disk. Outputs are saved in `config.EVALUATION_DIR`.
    """
    # -----------------------
    # Compute ROUGE-1 recalls
    # -----------------------
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_col_pairs = [
        ('Title_gold', 'Title_pred'),
        ('Receiver_gold', 'Receiver_pred'),
        ('Sender_gold', 'Sender_pred'),
        ('Address_gold', 'Address_pred')
    ]

    recall_df = merged_df.apply(lambda row: compute_row_rouge_recalls(
        row, rouge_col_pairs, scorer), axis=1)
    merged_df = pd.concat([merged_df, recall_df], axis=1)

    # ----------------------------------
    # Classification metric calculations
    # ----------------------------------
    class_metrics = {}
    for attr in ['Duplicate', 'Site_Registry_Releaseable']:
        y_true = merged_df[f"{attr}_gold"]
        y_pred = merged_df[f"{attr}_pred"]
        f1 = f1_score(y_true, y_pred, pos_label="yes", zero_division=1)
        prec = precision_score(
            y_true, y_pred, pos_label="yes", zero_division=1)
        rec = recall_score(y_true, y_pred, pos_label="yes", zero_division=1)

        class_metrics[attr] = {
            'F1': f1,
            'Precision': prec,
            'Recall': rec
        }

        # Log to console
        print(f"\n===== {attr.upper()} =====")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    # ----------------------------
    # Save row-level evaluation
    # ----------------------------
    merged_df.to_csv(config.EVALUATION_DIR /
                     "evaluation_merged_output.csv", index=False)

    # ------------------------------------
    # Save summary metrics (aggregated)
    # ------------------------------------
    summary_data = []

    # Classification metrics
    for attr, scores in class_metrics.items():
        summary_data.append({
            'Attribute': attr,
            'F1': scores['F1'],
            'Recall': scores['Recall'],
            'Precision': scores['Precision']
        })

    # ROUGE attributes – recall only
    for gold_col, _ in rouge_col_pairs:
        attr_name = gold_col.replace('_gold', '')
        avg_recall = merged_df[f"{attr_name}_recall"].mean()
        summary_data.append({
            'Attribute': attr_name,
            'F1': None,
            'Recall': avg_recall,
            'Precision': None
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(config.EVALUATION_DIR /
                      "evaluation_summary_metrics.csv", index=False)


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Evaluation script with optional test metadata switch.")
    parser.add_argument('--use-test-metadata', action='store_true',
                        help="Use 'test_metadata.csv' instead of 'clean_metadata.csv'")
    args = parser.parse_args()

    # Lookup File checks, if they do not exist program shuts down gracefully
    lookups_path = config.LOOKUPS_PATH

    metadata_filename = "test_metadata.csv" if args.use_test_metadata else "clean_metadata.csv"
    required_files = [lookups_path / metadata_filename]
    gold_metadata_path = lookups_path / metadata_filename

    verify_required_files(required_files)

    # Step 0: ensure some filenames match between Gold CSV and input directory. If not, notify user and exit early.

    # Extract all filenames (with extensions) from INPUT_DIR.
    input_filenames = {f.name.strip() for f in config.INPUT_DIR.iterdir() if f.is_file()}
    # Load gold metadata and extract gold titles.
    gold_df = loading_gold_metadata_csv(gold_metadata_path)
    gold_titles = set(gold_df["Current BC Mail title"].astype(str).str.strip())
    # Check for matches and exit if there are none.
    matching_filenames = input_filenames & gold_titles
    if not matching_filenames:
        print("❌ No matching filenames found between input directory files and Gold CSV.")
        print(f"Input directory:\n{config.INPUT_DIR}\nGold CSV:\n{gold_metadata_path}.\nExiting...")
        exit(0)

    # Step 1: Prepare files
    files_preparation()

    # Step 2: Run the pipeline
    if args.use_test_metadata:
        main(gold_metadata_path=gold_metadata_path)
        merged_df = load_evaluation_dataframe(
            gold_metadata_path=gold_metadata_path)
    else:
        main()
        merged_df = load_evaluation_dataframe()

    # Step 3: Load evaluation dataframe

    # Step 4: Compute F1 scores
    compute_scores(merged_df)
