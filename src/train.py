"""
train.py
Feature engineering, supervised baseline training, semi-supervised self-training,
and comprehensive experimental matrix execution.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from src.data_preparation import (
    load_raw_dataset,
    prepare_clean_dataset,
    create_train_test_split,
    simulate_labeled_unlabeled_split,
    LABEL_MAP,
    INV_LABEL_MAP,
    MAPPED_LABEL_NAMES,
)


def build_feature_extractor(
    word_ngram_range=(1, 2),
    char_ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
):
    """
    Constructs a FeatureUnion combining:
    1. Word-level TF-IDF (unigrams and bigrams) to capture phrases, negations, and sentiment words.
    2. Character-level TF-IDF (3-5 ngrams) to capture Indonesian slang, morphology (affixes),
       abbreviations, and informal social media spelling variations.
    """
    feature_union = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=word_ngram_range,
                min_df=min_df,
                sublinear_tf=sublinear_tf,
                token_pattern=r"(?u)\b\w+\b",
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=char_ngram_range,
                min_df=min_df,
                sublinear_tf=sublinear_tf,
            ),
        ),
    ])
    return feature_union


def evaluate_predictions(y_true, y_pred):
    """
    Computes standard classification evaluation metrics.
    Classes are mapped integers (0, 1, 2).
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=[MAPPED_LABEL_NAMES[i] for i in range(3)],
        output_dict=True,
    )
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "f1_negative": float(f1_per_class[0]),
        "f1_neutral": float(f1_per_class[1]),
        "f1_positive": float(f1_per_class[2]),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


def train_and_evaluate_supervised_baseline(
    X_train_vec,
    y_train_ssl,
    is_labeled,
    X_test_vec,
    y_test,
    C=1.0,
    class_weight="balanced",
    random_state=42,
):
    """
    Trains a supervised Logistic Regression baseline strictly on the labeled subset.
    """
    start_time = time.time()
    X_lab = X_train_vec[is_labeled]
    y_lab = y_train_ssl[is_labeled]

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X_lab, y_lab)
    train_time = time.time() - start_time

    y_pred = clf.predict(X_test_vec)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["train_time_sec"] = train_time
    metrics["n_labeled"] = int(is_labeled.sum())
    metrics["model_type"] = "Supervised_Baseline"
    return clf, metrics, y_pred


def train_and_evaluate_selftraining(
    X_train_vec,
    y_train_ssl,
    is_labeled,
    y_train_true,
    X_test_vec,
    y_test,
    threshold=0.80,
    max_iter=10,
    C=1.0,
    class_weight="balanced",
    random_state=42,
):
    """
    Trains a semi-supervised SelfTrainingClassifier using Logistic Regression as base estimator.
    Collects rich metadata on pseudo-label acceptance and distribution.
    """
    start_time = time.time()
    base_clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )

    st_clf = SelfTrainingClassifier(
        estimator=base_clf,
        threshold=threshold,
        max_iter=max_iter,
        verbose=False,
    )
    st_clf.fit(X_train_vec, y_train_ssl)
    train_time = time.time() - start_time

    y_pred = st_clf.predict(X_test_vec)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["train_time_sec"] = train_time
    metrics["threshold"] = threshold
    metrics["max_iter"] = max_iter
    metrics["actual_iterations"] = int(st_clf.n_iter_)
    metrics["model_type"] = "SemiSupervised_SelfTraining"

    # Analyze pseudo-labels
    # labeled_iter_ > 0 indicates samples labeled during self-training
    pseudo_mask = st_clf.labeled_iter_ > 0
    n_pseudo = int(pseudo_mask.sum())
    metrics["n_pseudo_labeled"] = n_pseudo
    metrics["n_unlabeled_remaining"] = int((st_clf.labeled_iter_ == -1).sum())

    # Pseudo-label class breakdown
    pseudo_labels = st_clf.transduction_[pseudo_mask]
    pseudo_dist = {
        "pseudo_neg": int((pseudo_labels == 0).sum()),
        "pseudo_neu": int((pseudo_labels == 1).sum()),
        "pseudo_pos": int((pseudo_labels == 2).sum()),
    }
    metrics.update(pseudo_dist)

    # Compute ground-truth accuracy of accepted pseudo-labels
    if n_pseudo > 0:
        pseudo_true = y_train_true[pseudo_mask]
        pseudo_acc = float((pseudo_labels == pseudo_true).mean())
        metrics["pseudo_label_accuracy"] = pseudo_acc
    else:
        metrics["pseudo_label_accuracy"] = 0.0

    return st_clf, metrics, y_pred


def run_full_benchmark(
    budgets=(0.10, 0.20, 0.30),
    seeds=(42, 123, 456),
    default_threshold=0.80,
    output_dir="outputs",
):
    """
    Executes the complete experimental design across budgets and random seeds:
    1. Loads and cleans raw data.
    2. Performs stratified group-aware train/test split.
    3. Fits FeatureUnion strictly on X_train.
    4. Evaluates Supervised Baseline vs Self-Training across budgets and seeds.
    5. Evaluates Oracle (100% labeled training set).
    6. Runs confidence threshold sensitivity analysis (0.70 to 0.90) at 20% budget.
    7. Evaluates Secondary Experiment: Majority-label resolution on conflicting tweets.
    8. Saves all artifacts, tables, and models.
    """
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)

    print("Step 1: Loading raw dataset and performing data audit...")
    raw_df = load_raw_dataset()
    clean_df = prepare_clean_dataset(raw_df, exclude_conflicts=True)
    print(f"Cleaned dataset: {len(clean_df)} unique tweets.")

    print("Step 2: Splitting into 80% train and 20% holdout test set...")
    train_df, test_df = create_train_test_split(clean_df, test_size=0.20, random_state=42)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Save test set for auditability
    test_df.to_csv(f"{output_dir}/predictions/test_set.csv", index=False)
    train_df.to_csv(f"{output_dir}/predictions/train_set.csv", index=False)

    print("Step 3: Building and fitting text feature pipeline (Word + Char TF-IDF)...")
    feature_extractor = build_feature_extractor()
    X_train_vec = feature_extractor.fit_transform(train_df["Tweet"].values)
    X_test_vec = feature_extractor.transform(test_df["Tweet"].values)
    y_test_mapped = test_df["sentimen"].map(LABEL_MAP).values
    print(f"Feature matrix dimensions: Train {X_train_vec.shape}, Test {X_test_vec.shape}")

    # Primary Experiments: Budgets x Seeds
    results = []
    best_macro_f1 = -1.0
    best_model_bundle = None

    print("\nStep 4: Executing Primary SSL Benchmark across budgets and seeds...")
    for budget in budgets:
        for seed in seeds:
            print(f"\n--- Running Budget: {int(budget*100)}% | Seed: {seed} ---")
            X_train, y_train_ssl, y_train_true, is_labeled = simulate_labeled_unlabeled_split(
                train_df, labeled_ratio=budget, random_state=seed
            )

            # 1. Supervised Baseline
            clf_sup, metrics_sup, y_pred_sup = train_and_evaluate_supervised_baseline(
                X_train_vec, y_train_ssl, is_labeled, X_test_vec, y_test_mapped, random_state=seed
            )
            metrics_sup["budget"] = budget
            metrics_sup["seed"] = seed
            results.append(metrics_sup)
            print(f"  [Supervised] Macro F1: {metrics_sup['macro_f1']:.4f} | Acc: {metrics_sup['accuracy']:.4f}")

            # 2. Self-Training Classifier
            clf_ssl, metrics_ssl, y_pred_ssl = train_and_evaluate_selftraining(
                X_train_vec,
                y_train_ssl,
                is_labeled,
                y_train_true,
                X_test_vec,
                y_test_mapped,
                threshold=default_threshold,
                max_iter=10,
                random_state=seed,
            )
            metrics_ssl["budget"] = budget
            metrics_ssl["seed"] = seed
            results.append(metrics_ssl)
            print(
                f"  [Self-Training] Macro F1: {metrics_ssl['macro_f1']:.4f} | Acc: {metrics_ssl['accuracy']:.4f} "
                f"| Pseudo added: {metrics_ssl['n_pseudo_labeled']} (Acc: {metrics_ssl['pseudo_label_accuracy']:.2%})"
            )

            # Check for best model to persist
            if metrics_ssl["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics_ssl["macro_f1"]
                best_model_bundle = {
                    "model": clf_ssl,
                    "feature_extractor": feature_extractor,
                    "budget": budget,
                    "seed": seed,
                    "metrics": metrics_ssl,
                    "label_map": LABEL_MAP,
                    "inv_label_map": INV_LABEL_MAP,
                }

    # Oracle: 100% Labeled Training Data (Upper Bound)
    print("\nStep 5: Training Supervised Oracle (100% labeled training data)...")
    y_train_full = train_df["sentimen"].map(LABEL_MAP).values
    clf_oracle, metrics_oracle, y_pred_oracle = train_and_evaluate_supervised_baseline(
        X_train_vec,
        y_train_full,
        np.ones(len(train_df), dtype=bool),
        X_test_vec,
        y_test_mapped,
        random_state=42,
    )
    metrics_oracle["budget"] = 1.0
    metrics_oracle["seed"] = 42
    metrics_oracle["model_type"] = "Supervised_Oracle_100pct"
    results.append(metrics_oracle)
    print(f"  [Oracle 100%] Macro F1: {metrics_oracle['macro_f1']:.4f} | Acc: {metrics_oracle['accuracy']:.4f}")

    # Threshold Sensitivity Analysis at 20% budget, seed 42
    print("\nStep 6: Running Confidence Threshold Sensitivity Analysis (20% budget, seed 42)...")
    _, y_train_ssl_20, y_train_true_20, is_labeled_20 = simulate_labeled_unlabeled_split(
        train_df, labeled_ratio=0.20, random_state=42
    )
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    thresh_results = []
    for thresh in thresholds:
        _, m_thresh, _ = train_and_evaluate_selftraining(
            X_train_vec,
            y_train_ssl_20,
            is_labeled_20,
            y_train_true_20,
            X_test_vec,
            y_test_mapped,
            threshold=thresh,
            max_iter=10,
            random_state=42,
        )
        thresh_results.append({
            "threshold": thresh,
            "macro_f1": m_thresh["macro_f1"],
            "accuracy": m_thresh["accuracy"],
            "weighted_f1": m_thresh["weighted_f1"],
            "n_pseudo_labeled": m_thresh["n_pseudo_labeled"],
            "pseudo_label_accuracy": m_thresh["pseudo_label_accuracy"],
            "actual_iterations": m_thresh["actual_iterations"],
            "pseudo_neg": m_thresh["pseudo_neg"],
            "pseudo_neu": m_thresh["pseudo_neu"],
            "pseudo_pos": m_thresh["pseudo_pos"],
        })
        print(f"  Threshold {thresh:.2f}: Macro F1 = {m_thresh['macro_f1']:.4f}, Pseudo = {m_thresh['n_pseudo_labeled']}, Pseudo Acc = {m_thresh['pseudo_label_accuracy']:.2%}")

    # Save threshold sensitivity table
    df_thresh = pd.DataFrame(thresh_results)
    df_thresh.to_csv(f"{output_dir}/tables/threshold_sensitivity.csv", index=False)

    # Step 7: Secondary Sensitivity Experiment: Majority-vote handling of conflicting tweets
    print("\nStep 7: Executing Secondary Experiment (Majority vote for conflicting tweets)...")
    sec_results = run_secondary_conflict_experiment(raw_df, feature_extractor, output_dir)

    # Step 8: Save best model bundle
    print(f"\nStep 8: Saving best model (Macro F1 = {best_macro_f1:.4f}) to {output_dir}/models/best_model.joblib...")
    joblib.dump(best_model_bundle, f"{output_dir}/models/best_model.joblib")

    # Step 9: Save primary benchmark results table
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{output_dir}/tables/detailed_seed_results.csv", index=False)

    # Also compute aggregate summary across seeds
    summary_rows = []
    for budget in budgets:
        for m_type in ["Supervised_Baseline", "SemiSupervised_SelfTraining"]:
            sub = df_results[(df_results["budget"] == budget) & (df_results["model_type"] == m_type)]
            summary_rows.append({
                "budget": budget,
                "model_type": m_type,
                "macro_f1_mean": float(sub["macro_f1"].mean()),
                "macro_f1_std": float(sub["macro_f1"].std()),
                "accuracy_mean": float(sub["accuracy"].mean()),
                "accuracy_std": float(sub["accuracy"].std()),
                "weighted_f1_mean": float(sub["weighted_f1"].mean()),
                "weighted_f1_std": float(sub["weighted_f1"].std()),
                "f1_neg_mean": float(sub["f1_negative"].mean()),
                "f1_neu_mean": float(sub["f1_neutral"].mean()),
                "f1_pos_mean": float(sub["f1_positive"].mean()),
                "train_time_mean": float(sub["train_time_sec"].mean()),
                "n_pseudo_mean": float(sub["n_pseudo_labeled"].mean()) if "n_pseudo_labeled" in sub else 0.0,
                "pseudo_acc_mean": float(sub["pseudo_label_accuracy"].mean()) if "pseudo_label_accuracy" in sub else 0.0,
            })

    # Add Oracle to summary
    oracle_sub = df_results[df_results["model_type"] == "Supervised_Oracle_100pct"]
    summary_rows.append({
        "budget": 1.0,
        "model_type": "Supervised_Oracle_100pct",
        "macro_f1_mean": float(oracle_sub["macro_f1"].values[0]),
        "macro_f1_std": 0.0,
        "accuracy_mean": float(oracle_sub["accuracy"].values[0]),
        "accuracy_std": 0.0,
        "weighted_f1_mean": float(oracle_sub["weighted_f1"].values[0]),
        "weighted_f1_std": 0.0,
        "f1_neg_mean": float(oracle_sub["f1_negative"].values[0]),
        "f1_neu_mean": float(oracle_sub["f1_neutral"].values[0]),
        "f1_pos_mean": float(oracle_sub["f1_positive"].values[0]),
        "train_time_mean": float(oracle_sub["train_time_sec"].values[0]),
        "n_pseudo_mean": 0.0,
        "pseudo_acc_mean": 0.0,
    })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{output_dir}/tables/budget_experiments_summary.csv", index=False)
    print("\nBenchmark completed successfully!")
    return df_results, df_summary, df_thresh, sec_results


def run_secondary_conflict_experiment(raw_df, feature_extractor, output_dir):
    """
    Sensitivity experiment: Instead of excluding conflicting-label tweets,
    resolves conflicts by assigning the majority label (ties broken towards neutral).
    Evaluates whether retaining conflicting tweets impacts performance.
    """
    df = raw_df.copy()
    from src.data_preparation import normalize_text_for_dedup
    df["norm_text"] = normalize_text_for_dedup(df["Tweet"])

    # Group by norm_text and determine majority label
    def get_majority_label(series):
        mode = series.mode()
        if len(mode) == 1:
            return mode.iloc[0]
        # Tie break: if 0 is among modes, choose 0 (Neutral), else first
        if 0 in mode.values:
            return 0
        return mode.iloc[0]

    majority_df = df.groupby("norm_text").agg({
        "sentimen": get_majority_label,
        "Tweet": "first"
    }).reset_index()

    train_maj, test_maj = create_train_test_split(majority_df, test_size=0.20, random_state=42)
    X_train_maj_vec = feature_extractor.transform(train_maj["Tweet"].values)
    X_test_maj_vec = feature_extractor.transform(test_maj["Tweet"].values)
    y_test_maj = test_maj["sentimen"].map(LABEL_MAP).values

    # Evaluate 20% budget with seed 42
    _, y_train_ssl_maj, y_train_true_maj, is_lab_maj = simulate_labeled_unlabeled_split(
        train_maj, labeled_ratio=0.20, random_state=42
    )

    _, m_sup, _ = train_and_evaluate_supervised_baseline(
        X_train_maj_vec, y_train_ssl_maj, is_lab_maj, X_test_maj_vec, y_test_maj, random_state=42
    )
    _, m_ssl, _ = train_and_evaluate_selftraining(
        X_train_maj_vec, y_train_ssl_maj, is_lab_maj, y_train_true_maj, X_test_maj_vec, y_test_maj,
        threshold=0.80, random_state=42
    )

    sec_results = [
        {"treatment": "Exclude_Conflicts (Primary)", "model": "Supervised_20%", "macro_f1": None}, # filled later
        {"treatment": "Majority_Vote (Secondary)", "model": "Supervised_20%", "macro_f1": m_sup["macro_f1"], "accuracy": m_sup["accuracy"]},
        {"treatment": "Majority_Vote (Secondary)", "model": "SelfTraining_20%", "macro_f1": m_ssl["macro_f1"], "accuracy": m_ssl["accuracy"]},
    ]
    df_sec = pd.DataFrame(sec_results)
    df_sec.to_csv(f"{output_dir}/tables/secondary_conflict_experiment.csv", index=False)
    print(f"  Majority-vote results: Supervised F1 = {m_sup['macro_f1']:.4f}, SSL F1 = {m_ssl['macro_f1']:.4f}")
    return df_sec


if __name__ == "__main__":
    run_full_benchmark()
