"""
evaluate.py
Module for generating publication-quality figures, comprehensive evaluation tables,
error analysis, and reporting artifacts.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.data_preparation import (
    load_raw_dataset,
    prepare_clean_dataset,
    audit_dataset,
    LABEL_MAP,
    INV_LABEL_MAP,
    LABEL_NAMES,
    MAPPED_LABEL_NAMES,
)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#eeeeee"
plt.rcParams["grid.linestyle"] = "--"

CLASS_COLORS = {
    "Negative": "#e74c3c",  # Coral red
    "Neutral": "#7f8c8d",   # Slate gray
    "Positive": "#2ecc71",  # Emerald green
}
MODEL_COLORS = {
    "Supervised Baseline": "#3498db",  # Blue
    "Semi-Supervised (Self-Training)": "#e67e22",  # Orange
    "Oracle (100% Labeled)": "#2c3e50",  # Dark Slate
}


def generate_eda_figures(raw_df, clean_df, output_dir="outputs/figures"):
    """
    Generates Figures 1 to 4:
    - fig1: Class distribution (Raw, Conflicting, Clean)
    - fig2: Length distribution (characters & words)
    - fig3: Length by sentiment class
    - fig4: Top unigrams & bigrams by sentiment class
    """
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Figure 1: Class Distribution Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Raw counts
    raw_counts = raw_df["sentimen"].map(LABEL_NAMES).value_counts().reindex(["Negative", "Neutral", "Positive"])
    colors = [CLASS_COLORS[c] for c in raw_counts.index]
    bars1 = axes[0].bar(raw_counts.index, raw_counts.values, color=colors, alpha=0.85, edgecolor="#333333", linewidth=1.2)
    axes[0].set_title("Raw Dataset Class Distribution (N=10,806)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Number of Tweets", fontsize=11)
    axes[0].set_ylim(0, 6000)
    for bar in bars1:
        yval = bar.get_height()
        pct = yval / len(raw_df) * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 100, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Clean deduplicated counts
    clean_counts = clean_df["sentimen"].map(LABEL_NAMES).value_counts().reindex(["Negative", "Neutral", "Positive"])
    bars2 = axes[1].bar(clean_counts.index, clean_counts.values, color=colors, alpha=0.85, edgecolor="#333333", linewidth=1.2)
    axes[1].set_title("Clean Deduplicated Dataset (N=9,926)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Number of Tweets", fontsize=11)
    axes[1].set_ylim(0, 6000)
    for bar in bars2:
        yval = bar.get_height()
        pct = yval / len(clean_df) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 100, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_class_distribution.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Tweet Length Distribution (Chars & Words)
    # -------------------------------------------------------------
    clean_df = clean_df.copy()
    clean_df["char_len"] = clean_df["Tweet"].str.len()
    clean_df["word_len"] = clean_df["Tweet"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Character length
    sns.histplot(clean_df["char_len"], bins=35, kde=True, ax=axes[0], color="#2980b9", edgecolor="#1c5980", alpha=0.7)
    axes[0].axvline(clean_df["char_len"].mean(), color="#c0392b", linestyle="--", linewidth=1.8, label=f"Mean: {clean_df['char_len'].mean():.1f}")
    axes[0].axvline(clean_df["char_len"].median(), color="#27ae60", linestyle=":", linewidth=1.8, label=f"Median: {clean_df['char_len'].median():.1f}")
    axes[0].set_title("Distribution of Character Length", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Number of Characters per Tweet", fontsize=11)
    axes[0].set_ylabel("Tweet Count", fontsize=11)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    # Word length
    sns.histplot(clean_df["word_len"], bins=28, kde=True, ax=axes[1], color="#8e44ad", edgecolor="#5e2d73", alpha=0.7)
    axes[1].axvline(clean_df["word_len"].mean(), color="#c0392b", linestyle="--", linewidth=1.8, label=f"Mean: {clean_df['word_len'].mean():.1f}")
    axes[1].axvline(clean_df["word_len"].median(), color="#27ae60", linestyle=":", linewidth=1.8, label=f"Median: {clean_df['word_len'].median():.1f}")
    axes[1].set_title("Distribution of Word Count", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Number of Words per Tweet", fontsize=11)
    axes[1].set_ylabel("Tweet Count", fontsize=11)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "fig2_tweet_length_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Tweet Length by Sentiment Class
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    clean_df["Sentiment"] = clean_df["sentimen"].map(LABEL_NAMES)
    order = ["Negative", "Neutral", "Positive"]
    palette = [CLASS_COLORS[c] for c in order]

    sns.boxplot(x="Sentiment", y="char_len", data=clean_df, order=order, palette=palette, ax=axes[0], width=0.5, boxprops=dict(alpha=0.85))
    axes[0].set_title("Character Length across Sentiment Classes", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Sentiment Class", fontsize=11)
    axes[0].set_ylabel("Character Length", fontsize=11)

    sns.boxplot(x="Sentiment", y="word_len", data=clean_df, order=order, palette=palette, ax=axes[1], width=0.5, boxprops=dict(alpha=0.85))
    axes[1].set_title("Word Count across Sentiment Classes", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Sentiment Class", fontsize=11)
    axes[1].set_ylabel("Word Count", fontsize=11)

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_length_by_sentiment.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------
    # Figure 4: Top Terms / N-grams by Sentiment Class
    # -------------------------------------------------------------
    # Minimal stopword list excluding negation words (tidak, tak, bukan, jangan kept!)
    custom_stops = {
        "yg", "dan", "di", "yang", "ini", "itu", "ke", "dari", "untuk",
        "ada", "aku", "saya", "kamu", "dia", "mereka", "kita", "nya",
        "kalo", "kalau", "bisa", "udah", "sudah", "aja", "saja", "lagi",
        "jadi", "mau", "buat", "apa", "dengan", "sama", "pada", "dgn"
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)
    for idx, sentiment in enumerate(order):
        s_val = [k for k, v in LABEL_NAMES.items() if v == sentiment][0]
        subset_texts = clean_df[clean_df["sentimen"] == s_val]["Tweet"]

        cv = CountVectorizer(ngram_range=(1, 2), min_df=3, stop_words=list(custom_stops), token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        dtm = cv.fit_transform(subset_texts)
        term_freq = np.asarray(dtm.sum(axis=0)).ravel()
        vocab = np.array(cv.get_feature_names_out())

        top_indices = term_freq.argsort()[::-1][:10]
        top_terms = vocab[top_indices]
        top_counts = term_freq[top_indices]

        y_pos = np.arange(len(top_terms))
        axes[idx].barh(y_pos, top_counts, color=CLASS_COLORS[sentiment], alpha=0.85, edgecolor="#333333")
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(top_terms, fontsize=10)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f"Top Terms: {sentiment}", fontsize=12, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Term Frequency", fontsize=10)

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_top_ngrams_by_sentiment.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig4_path}")


def generate_benchmark_figures(
    df_summary,
    df_details,
    df_thresh,
    output_dir="outputs/figures",
):
    """
    Generates Figures 5 to 10:
    - fig5: Macro F1 comparison across budgets (error bars for std, oracle line)
    - fig6: Confusion matrix Supervised Baseline (20% budget, seed 42)
    - fig7: Confusion matrix Self-Training (20% budget, seed 42)
    - fig8: Pseudo-label analysis (classes, count, pseudo accuracy)
    - fig9: Threshold sensitivity (Macro F1 & accepted volume vs threshold)
    - fig10: Per-class F1 comparison across budgets
    """
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Figure 5: Macro F1 Comparison across Labeled Budgets
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    budgets = [0.10, 0.20, 0.30]
    budget_labels = ["10% Labeled\n(N=794)", "20% Labeled\n(N=1,588)", "30% Labeled\n(N=2,382)"]

    sup_means = []
    sup_stds = []
    ssl_means = []
    ssl_stds = []

    for b in budgets:
        sup_row = df_summary[(df_summary["budget"] == b) & (df_summary["model_type"] == "Supervised_Baseline")].iloc[0]
        ssl_row = df_summary[(df_summary["budget"] == b) & (df_summary["model_type"] == "SemiSupervised_SelfTraining")].iloc[0]
        sup_means.append(sup_row["macro_f1_mean"])
        sup_stds.append(sup_row["macro_f1_std"])
        ssl_means.append(ssl_row["macro_f1_mean"])
        ssl_stds.append(ssl_row["macro_f1_std"])

    x = np.arange(len(budgets))
    width = 0.35

    rects1 = ax.bar(x - width/2, sup_means, width, yerr=sup_stds, capsize=5, label="Supervised Baseline", color="#3498db", alpha=0.9, edgecolor="#1f618d")
    rects2 = ax.bar(x + width/2, ssl_means, width, yerr=ssl_stds, capsize=5, label="Semi-Supervised (Self-Training)", color="#e67e22", alpha=0.9, edgecolor="#b96213")

    # Oracle line
    oracle_row = df_summary[df_summary["model_type"] == "Supervised_Oracle_100pct"].iloc[0]
    oracle_val = oracle_row["macro_f1_mean"]
    ax.axhline(oracle_val, color="#2c3e50", linestyle="--", linewidth=2.0, label=f"Oracle 100% Labeled (F1={oracle_val:.4f})")

    ax.set_ylabel("Macro F1-Score (Test Set)", fontsize=12, fontweight="bold")
    ax.set_title("Supervised vs Semi-Supervised Performance by Labeled Budget", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels, fontsize=11)
    ax.set_ylim(0.40, 0.70)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "fig5_macro_f1_comparison.png")
    fig.savefig(fig5_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig5_path}")

    # -------------------------------------------------------------
    # Figure 6 & 7: Confusion Matrices for 20% Budget (Seed 42)
    # -------------------------------------------------------------
    sup_20 = df_details[(df_details["budget"] == 0.20) & (df_details["seed"] == 42) & (df_details["model_type"] == "Supervised_Baseline")].iloc[0]
    ssl_20 = df_details[(df_details["budget"] == 0.20) & (df_details["seed"] == 42) & (df_details["model_type"] == "SemiSupervised_SelfTraining")].iloc[0]

    # Confusion matrix supervised
    cm_sup = np.array(eval(sup_20["confusion_matrix"]) if isinstance(sup_20["confusion_matrix"], str) else sup_20["confusion_matrix"])
    cm_sup_norm = cm_sup.astype("float") / cm_sup.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    labels = ["Negative", "Neutral", "Positive"]
    sns.heatmap(cm_sup_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=True, annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax.set_title("Supervised Baseline Confusion Matrix\n(20% Labeled Data Budget, Seed 42)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "fig6_confusion_matrix_supervised.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig6_path}")

    # Confusion matrix self-training
    cm_ssl = np.array(eval(ssl_20["confusion_matrix"]) if isinstance(ssl_20["confusion_matrix"], str) else ssl_20["confusion_matrix"])
    cm_ssl_norm = cm_ssl.astype("float") / cm_ssl.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)
    sns.heatmap(cm_ssl_norm, annot=True, fmt=".2%", cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=ax, cbar=True, annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax.set_title("Self-Training Classifier Confusion Matrix\n(20% Labeled Data Budget, Seed 42, Thresh=0.80)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    fig7_path = os.path.join(output_dir, "fig7_confusion_matrix_selftraining.png")
    fig.savefig(fig7_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig7_path}")

    # -------------------------------------------------------------
    # Figure 8: Pseudo-label Dynamics Across Budgets
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Class distribution of pseudo-labels (mean across seeds)
    pseudo_class_data = []
    for b in budgets:
        sub = df_details[(df_details["budget"] == b) & (df_details["model_type"] == "SemiSupervised_SelfTraining")]
        pseudo_class_data.append({
            "budget": f"{int(b*100)}% Budget",
            "Negative": sub["pseudo_neg"].mean(),
            "Neutral": sub["pseudo_neu"].mean(),
            "Positive": sub["pseudo_pos"].mean(),
        })
    df_p_class = pd.DataFrame(pseudo_class_data).set_index("budget")
    df_p_class.plot(kind="bar", stacked=True, ax=axes[0], color=[CLASS_COLORS[c] for c in ["Negative", "Neutral", "Positive"]], edgecolor="#333333", alpha=0.85)
    axes[0].set_title("Accepted Pseudo-Labels Breakdown by Class", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Labeled Data Budget", fontsize=11)
    axes[0].set_ylabel("Average Count of Pseudo-Labels", fontsize=11)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Sentiment Class", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Pseudo-label accuracy vs total pool
    pseudo_accs = [df_details[(df_details["budget"] == b) & (df_details["model_type"] == "SemiSupervised_SelfTraining")]["pseudo_label_accuracy"].mean() * 100 for b in budgets]
    pseudo_counts = [df_details[(df_details["budget"] == b) & (df_details["model_type"] == "SemiSupervised_SelfTraining")]["n_pseudo_labeled"].mean() for b in budgets]

    ax2_twin = axes[1].twinx()
    p1 = axes[1].bar(x - 0.15, pseudo_counts, width=0.3, color="#16a085", alpha=0.85, label="Pseudo-labels Added (Count)", edgecolor="#0e6251")
    p2 = ax2_twin.plot(x + 0.15, pseudo_accs, color="#d35400", marker="o", markersize=8, linewidth=2.5, label="Pseudo-label Accuracy (%)")

    axes[1].set_title("Pseudo-Label Volume vs Ground-Truth Accuracy", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(budget_labels, fontsize=10)
    axes[1].set_ylabel("Pseudo-Labels Accepted (Count)", fontsize=11, color="#16a085")
    ax2_twin.set_ylabel("Accuracy on Hidden Labels (%)", fontsize=11, color="#d35400")
    ax2_twin.set_ylim(50, 100)

    # Combined legend
    lines, labels_leg = axes[1].get_legend_handles_labels()
    lines2, labels2_leg = ax2_twin.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels_leg + labels2_leg, loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    fig8_path = os.path.join(output_dir, "fig8_pseudolabel_analysis.png")
    fig.savefig(fig8_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig8_path}")

    # -------------------------------------------------------------
    # Figure 9: Confidence Threshold Sensitivity Curve
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.5, 5), dpi=300)
    ax2 = ax1.twinx()

    thresh_x = df_thresh["threshold"]
    color1 = "#2980b9"
    color2 = "#c0392b"

    line1 = ax1.plot(thresh_x, df_thresh["macro_f1"], color=color1, marker="s", markersize=8, linewidth=2.2, label="Macro F1-Score")
    line2 = ax2.plot(thresh_x, df_thresh["n_pseudo_labeled"], color=color2, marker="^", markersize=8, linewidth=2.2, linestyle="--", label="Pseudo-Labels Accepted")

    ax1.set_xlabel("Confidence Decision Threshold (Threshold parameter)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test Set Macro F1-Score", fontsize=11, color=color1, fontweight="bold")
    ax2.set_ylabel("Number of Pseudo-Labels Accepted", fontsize=11, color=color2, fontweight="bold")
    ax1.set_title("Impact of Confidence Threshold on Performance & Pseudo-Label Intake\n(20% Budget, Max Iter=10)", fontsize=12, fontweight="bold", pad=12)

    # Combine legends
    lines = line1 + line2
    labels_thresh = [l.get_label() for l in lines]
    ax1.legend(lines, labels_thresh, loc="center left", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    fig9_path = os.path.join(output_dir, "fig9_threshold_sensitivity.png")
    fig.savefig(fig9_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig9_path}")

    # -------------------------------------------------------------
    # Figure 10: Per-Class F1-Score Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    class_names = ["Negative", "Neutral", "Positive"]
    x_cls = np.arange(len(class_names))
    width_cls = 0.25

    # At 20% budget
    sup_20_row = df_summary[(df_summary["budget"] == 0.20) & (df_summary["model_type"] == "Supervised_Baseline")].iloc[0]
    ssl_20_row = df_summary[(df_summary["budget"] == 0.20) & (df_summary["model_type"] == "SemiSupervised_SelfTraining")].iloc[0]
    oracle_row = df_summary[df_summary["model_type"] == "Supervised_Oracle_100pct"].iloc[0]

    f1_sup_cls = [sup_20_row["f1_neg_mean"], sup_20_row["f1_neu_mean"], sup_20_row["f1_pos_mean"]]
    f1_ssl_cls = [ssl_20_row["f1_neg_mean"], ssl_20_row["f1_neu_mean"], ssl_20_row["f1_pos_mean"]]
    f1_ora_cls = [oracle_row["f1_neg_mean"], oracle_row["f1_neu_mean"], oracle_row["f1_pos_mean"]]

    b1 = ax.bar(x_cls - width_cls, f1_sup_cls, width_cls, label="Supervised Baseline (20% Labeled)", color="#3498db", alpha=0.9, edgecolor="#1f618d")
    b2 = ax.bar(x_cls, f1_ssl_cls, width_cls, label="Self-Training (20% Labeled)", color="#e67e22", alpha=0.9, edgecolor="#b96213")
    b3 = ax.bar(x_cls + width_cls, f1_ora_cls, width_cls, label="Oracle (100% Labeled)", color="#2c3e50", alpha=0.9, edgecolor="#1a252f")

    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Per-Class F1-Score Breakdown (20% Budget vs Oracle)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x_cls)
    ax.set_xticklabels(class_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0.40, 0.75)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig10_path = os.path.join(output_dir, "fig10_per_class_f1_comparison.png")
    fig.savefig(fig10_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig10_path}")


def generate_markdown_tables(output_dir="outputs/tables"):
    """
    Converts CSV tables generated during training into formatted Markdown tables
    ready for embedding into final_report.md.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary benchmark table
    summary_csv = os.path.join(output_dir, "budget_experiments_summary.csv")
    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        df_sum["Budget"] = df_sum["budget"].apply(lambda b: f"{int(b*100)}%" if b < 1.0 else "100% (Oracle)")
        df_sum["Model"] = df_sum["model_type"].replace({
            "Supervised_Baseline": "Supervised Baseline",
            "SemiSupervised_SelfTraining": "Self-Training SSL",
            "Supervised_Oracle_100pct": "Supervised Oracle"
        })
        df_sum["Macro F1"] = df_sum.apply(lambda r: f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}" if r["macro_f1_std"] > 0 else f"{r['macro_f1_mean']:.4f}", axis=1)
        df_sum["Accuracy"] = df_sum.apply(lambda r: f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}" if r["accuracy_std"] > 0 else f"{r['accuracy_mean']:.4f}", axis=1)
        df_sum["Weighted F1"] = df_sum.apply(lambda r: f"{r['weighted_f1_mean']:.4f} ± {r['weighted_f1_std']:.4f}" if r["weighted_f1_std"] > 0 else f"{r['weighted_f1_mean']:.4f}", axis=1)
        df_sum["Pseudo Added"] = df_sum["n_pseudo_mean"].apply(lambda n: f"{int(round(n)):,}" if n > 0 else "-")
        df_sum["Pseudo Acc"] = df_sum["pseudo_acc_mean"].apply(lambda a: f"{a*100:.2f}%" if a > 0 else "-")
        df_sum["Train Time (s)"] = df_sum["train_time_mean"].apply(lambda t: f"{t:.2f}s")

        cols = ["Budget", "Model", "Macro F1", "Accuracy", "Weighted F1", "Pseudo Added", "Pseudo Acc", "Train Time (s)"]
        md_table = df_sum[cols].to_markdown(index=False)
        with open(os.path.join(output_dir, "budget_experiments_summary.md"), "w", encoding="utf-8") as f:
            f.write(md_table)
        print("Generated budget_experiments_summary.md")

    # 2. Threshold sensitivity table
    thresh_csv = os.path.join(output_dir, "threshold_sensitivity.csv")
    if os.path.exists(thresh_csv):
        df_th = pd.read_csv(thresh_csv)
        df_th_disp = pd.DataFrame({
            "Threshold": df_th["threshold"].apply(lambda t: f"{t:.2f}"),
            "Macro F1": df_th["macro_f1"].apply(lambda v: f"{v:.4f}"),
            "Accuracy": df_th["accuracy"].apply(lambda v: f"{v:.4f}"),
            "Pseudo Added": df_th["n_pseudo_labeled"].apply(lambda v: f"{int(v):,}"),
            "Pseudo Accuracy": df_th["pseudo_label_accuracy"].apply(lambda v: f"{v*100:.2f}%"),
            "Iterations": df_th["actual_iterations"],
            "Pseudo Neg": df_th["pseudo_neg"],
            "Pseudo Neu": df_th["pseudo_neu"],
            "Pseudo Pos": df_th["pseudo_pos"],
        })
        with open(os.path.join(output_dir, "threshold_sensitivity.md"), "w", encoding="utf-8") as f:
            f.write(df_th_disp.to_markdown(index=False))
        print("Generated threshold_sensitivity.md")

    # 3. Secondary conflict table
    sec_csv = os.path.join(output_dir, "secondary_conflict_experiment.csv")
    if os.path.exists(sec_csv):
        df_sec = pd.read_csv(sec_csv)
        # Update first row with primary result
        if os.path.exists(summary_csv):
            df_sum = pd.read_csv(summary_csv)
            prim_sup_20 = df_sum[(df_sum["budget"] == 0.20) & (df_sum["model_type"] == "Supervised_Baseline")]["macro_f1_mean"].values[0]
            prim_ssl_20 = df_sum[(df_sum["budget"] == 0.20) & (df_sum["model_type"] == "SemiSupervised_SelfTraining")]["macro_f1_mean"].values[0]
            df_sec.loc[df_sec["treatment"].str.contains("Exclude_Conflicts"), "macro_f1"] = prim_sup_20

            # Add primary SSL row
            row_ssl_prim = pd.DataFrame([{
                "treatment": "Exclude_Conflicts (Primary)",
                "model": "SelfTraining_20%",
                "macro_f1": prim_ssl_20,
                "accuracy": df_sum[(df_sum["budget"] == 0.20) & (df_sum["model_type"] == "SemiSupervised_SelfTraining")]["accuracy_mean"].values[0]
            }])
            df_sec = pd.concat([df_sec.iloc[:1], row_ssl_prim, df_sec.iloc[1:]], ignore_index=True)
            df_sec.to_csv(sec_csv, index=False)

        with open(os.path.join(output_dir, "secondary_conflict_experiment.md"), "w", encoding="utf-8") as f:
            f.write(df_sec.to_markdown(index=False))
        print("Generated secondary_conflict_experiment.md")


def run_full_evaluation_pipeline(output_dir="outputs"):
    """
    Main entry point for generating all figures and evaluation tables.
    """
    print("\n================ EVALUATION & FIGURE GENERATION ================")
    raw_df = load_raw_dataset()
    clean_df = prepare_clean_dataset(raw_df, exclude_conflicts=True)

    print("Generating EDA figures (Figures 1-4)...")
    generate_eda_figures(raw_df, clean_df, output_dir=f"{output_dir}/figures")

    print("Loading benchmark results...")
    summary_path = f"{output_dir}/tables/budget_experiments_summary.csv"
    details_path = f"{output_dir}/tables/detailed_seed_results.csv"
    thresh_path = f"{output_dir}/tables/threshold_sensitivity.csv"

    if os.path.exists(summary_path) and os.path.exists(details_path) and os.path.exists(thresh_path):
        df_summary = pd.read_csv(summary_path)
        df_details = pd.read_csv(details_path)
        df_thresh = pd.read_csv(thresh_path)

        print("Generating Benchmark & Diagnostic figures (Figures 5-10)...")
        generate_benchmark_figures(df_summary, df_details, df_thresh, output_dir=f"{output_dir}/figures")

        print("Generating markdown tables...")
        generate_markdown_tables(output_dir=f"{output_dir}/tables")
    else:
        print("Benchmark result files not found. Please run src/train.py first.")


if __name__ == "__main__":
    run_full_evaluation_pipeline()
