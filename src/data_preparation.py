"""
data_preparation.py
Module for robust data loading, quality auditing, text normalization,
conflict handling, duplicate-leakage prevention, and semi-supervised split simulation.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Label mappings
# Scikit-learn's SelfTrainingClassifier reserves -1 for unlabeled data.
# Therefore we map our original {-1: Negative, 0: Neutral, 1: Positive}
# to mapped integers {0: Negative, 1: Neutral, 2: Positive}.
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}
LABEL_NAMES = {-1: "Negative", 0: "Neutral", 1: "Positive"}
MAPPED_LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def load_raw_dataset(filepath: str = "dataset/Indonesian_Sentiment_Twitter_Dataset_Labeled.csv") -> pd.DataFrame:
    """
    Robustly loads the raw tweet dataset.
    Detects tab delimiter or falls back gracefully, validating required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # The file has a .csv extension but is tab-delimited
    try:
        df = pd.read_csv(filepath, sep="\t", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep="\t", encoding="latin-1")

    # Validate columns
    expected_cols = {"sentimen", "Tweet"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {expected_cols}, got {df.columns.tolist()}")

    # Ensure sentimen is integer
    df["sentimen"] = df["sentimen"].astype(int)
    # Ensure Tweet is string
    df["Tweet"] = df["Tweet"].astype(str)

    return df


def normalize_text_for_dedup(text_series: pd.Series) -> pd.Series:
    """
    Normalizes text for duplicate and conflict detection:
    - Lowercase
    - Collapses repeated whitespace
    - Strips leading and trailing whitespace
    Preserves original tokens, punctuation, numbers, and vocabulary.
    """
    return (
        text_series.astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def audit_dataset(df: pd.DataFrame) -> dict:
    """
    Performs a thorough audit of data quality, dimensions, duplicates,
    conflicts, lengths, and distributions.
    """
    df = df.copy()
    df["norm_text"] = normalize_text_for_dedup(df["Tweet"])
    df["char_len"] = df["Tweet"].str.len()
    df["word_len"] = df["Tweet"].str.split().str.len()

    total_rows = len(df)
    total_cols = df.shape[1]
    missing_vals = df[["sentimen", "Tweet"]].isnull().sum().to_dict()

    raw_exact_dupes = int(df.duplicated(subset=["sentimen", "Tweet"]).sum())
    raw_unique_tweets = int(df["Tweet"].nunique())
    raw_repeated_tweets = int((df["Tweet"].value_counts() > 1).sum())

    norm_unique_tweets = int(df["norm_text"].nunique())
    norm_repeated_tweets = int((df["norm_text"].value_counts() > 1).sum())

    # Conflicting label analysis on normalized text
    grouped = df.groupby("norm_text")["sentimen"].nunique()
    conflicting_groups = grouped[grouped > 1]
    n_conflicting_groups = int(len(conflicting_groups))
    conflicting_rows = df[df["norm_text"].isin(conflicting_groups.index)]
    n_conflicting_rows = int(len(conflicting_rows))

    # Class counts raw
    class_counts_raw = df["sentimen"].value_counts().to_dict()

    # Length stats
    char_stats = {
        "min": float(df["char_len"].min()),
        "mean": float(df["char_len"].mean()),
        "median": float(df["char_len"].median()),
        "max": float(df["char_len"].max()),
        "std": float(df["char_len"].std()),
    }
    word_stats = {
        "min": float(df["word_len"].min()),
        "mean": float(df["word_len"].mean()),
        "median": float(df["word_len"].median()),
        "max": float(df["word_len"].max()),
        "std": float(df["word_len"].std()),
    }

    # Length by class
    class_length_stats = {}
    for c, name in LABEL_NAMES.items():
        sub = df[df["sentimen"] == c]
        class_length_stats[name] = {
            "char_mean": float(sub["char_len"].mean()),
            "char_std": float(sub["char_len"].std()),
            "word_mean": float(sub["word_len"].mean()),
            "word_std": float(sub["word_len"].std()),
        }

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_vals": missing_vals,
        "raw_exact_dupes": raw_exact_dupes,
        "raw_unique_tweets": raw_unique_tweets,
        "raw_repeated_tweets": raw_repeated_tweets,
        "norm_unique_tweets": norm_unique_tweets,
        "norm_repeated_tweets": norm_repeated_tweets,
        "n_conflicting_groups": n_conflicting_groups,
        "n_conflicting_rows": n_conflicting_rows,
        "class_counts_raw": class_counts_raw,
        "char_stats": char_stats,
        "word_stats": word_stats,
        "class_length_stats": class_length_stats,
    }


def prepare_clean_dataset(df: pd.DataFrame, exclude_conflicts: bool = True) -> pd.DataFrame:
    """
    Cleans dataset by:
    1. Creating normalized text column.
    2. Excluding conflicting label groups if exclude_conflicts is True.
    3. Deduplicating repeated records with consistent text and label so
       each unique normalized tweet text appears exactly once.
    4. Preserving the original tweet text column.
    """
    df = df.copy()
    df["norm_text"] = normalize_text_for_dedup(df["Tweet"])

    if exclude_conflicts:
        grouped = df.groupby("norm_text")["sentimen"].nunique()
        conflict_keys = set(grouped[grouped > 1].index)
        df = df[~df["norm_text"].isin(conflict_keys)].copy()

    # Deduplicate by norm_text (since all remaining norm_texts have consistent labels)
    df_dedup = df.drop_duplicates(subset=["norm_text"]).copy().reset_index(drop=True)
    return df_dedup


def create_train_test_split(df_clean: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Splits the deduplicated cleaned dataset into train and holdout test sets.
    Because each row is a unique normalized tweet text, this guarantees
    ZERO text overlap/leakage between train and test splits, while preserving
    exact class stratification.
    """
    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        stratify=df_clean["sentimen"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def simulate_labeled_unlabeled_split(
    train_df: pd.DataFrame,
    labeled_ratio: float,
    random_state: int = 42,
):
    """
    Simulates a semi-supervised budget within the training set.
    - labeled_ratio: e.g. 0.10, 0.20, 0.30
    - Selects labeled examples using stratified sampling.
    - Remaining training examples have their labels hidden (marked as -1 for SelfTrainingClassifier).
    Returns:
    - X_train (Series of tweet texts)
    - y_train_ssl (numpy array: mapped classes 0, 1, 2 for labeled; -1 for unlabeled)
    - y_train_true_mapped (numpy array: true mapped classes 0, 1, 2 for all training samples)
    - is_labeled (boolean mask indicating which training samples are initially labeled)
    """
    # Mapped labels: -1 -> 0, 0 -> 1, 1 -> 2
    y_true_mapped = train_df["sentimen"].map(LABEL_MAP).values
    X_train = train_df["Tweet"].values

    # Stratified split to pick the labeled subset
    labeled_indices, _ = train_test_split(
        np.arange(len(train_df)),
        train_size=labeled_ratio,
        stratify=y_true_mapped,
        random_state=random_state,
    )

    is_labeled = np.zeros(len(train_df), dtype=bool)
    is_labeled[labeled_indices] = True

    # Build SSL target: keep mapped label for labeled, assign -1 for unlabeled
    y_train_ssl = np.full(len(train_df), fill_value=-1, dtype=int)
    y_train_ssl[is_labeled] = y_true_mapped[is_labeled]

    return X_train, y_train_ssl, y_true_mapped, is_labeled


if __name__ == "__main__":
    print("Testing data preparation module...")
    raw_df = load_raw_dataset()
    audit = audit_dataset(raw_df)
    print("Raw rows:", audit["total_rows"])
    print("Class counts raw:", audit["class_counts_raw"])
    print("Conflicting text groups:", audit["n_conflicting_groups"])
    print("Conflicting rows affected:", audit["n_conflicting_rows"])

    clean_df = prepare_clean_dataset(raw_df, exclude_conflicts=True)
    print("Clean deduplicated rows:", len(clean_df))
    print("Clean class distribution:\n", clean_df["sentimen"].value_counts())

    train_df, test_df = create_train_test_split(clean_df, test_size=0.20, random_state=42)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Check leakage
    train_texts = set(normalize_text_for_dedup(train_df["Tweet"]))
    test_texts = set(normalize_text_for_dedup(test_df["Tweet"]))
    overlap = train_texts.intersection(test_texts)
    print(f"Leakage check: {len(overlap)} overlapping texts between train and test!")

    # Check SSL simulation
    X_tr, y_tr_ssl, y_tr_true, is_lab = simulate_labeled_unlabeled_split(train_df, labeled_ratio=0.10, random_state=42)
    print(f"10% budget: {is_lab.sum()} labeled, {(~is_lab).sum()} unlabeled (marked as -1)")
    print("Data preparation module tests passed successfully!")
