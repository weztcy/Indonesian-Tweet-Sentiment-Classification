"""
app.py
Interactive Streamlit application for Indonesian Tweet Sentiment Classification.
Includes real-time sentiment inference, probability distributions, benchmark comparison,
pseudo-label dynamics explorer, and dataset insights.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import TweetSentimentPredictor
from src.data_preparation import LABEL_NAMES, MAPPED_LABEL_NAMES

# Page configuration
st.set_page_config(
    page_title="Indonesian Tweet Sentiment Classification",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished dark mode styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 10px rgba(56, 189, 248, 0.15);
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        color: #f8fafc;
    }
    .sentiment-badge-pos {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.35);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .sentiment-badge-neu {
        background-color: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        border: 1px solid rgba(203, 213, 225, 0.35);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .sentiment-badge-neg {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.35);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    model_path = "outputs/models/best_model.joblib"
    if os.path.exists(model_path):
        return TweetSentimentPredictor(model_path)
    return None


@st.cache_data
def load_tables():
    tables = {}
    summary_path = "outputs/tables/budget_experiments_summary.csv"
    thresh_path = "outputs/tables/threshold_sensitivity.csv"
    sec_path = "outputs/tables/secondary_conflict_experiment.csv"
    clean_data_path = "outputs/predictions/test_set.csv"

    if os.path.exists(summary_path):
        tables["summary"] = pd.read_csv(summary_path)
    if os.path.exists(thresh_path):
        tables["thresh"] = pd.read_csv(thresh_path)
    if os.path.exists(sec_path):
        tables["sec"] = pd.read_csv(sec_path)
    if os.path.exists(clean_data_path):
        tables["test_sample"] = pd.read_csv(clean_data_path)
    return tables


def main():
    st.markdown('<div class="main-header">Indonesian Tweet Sentiment Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Semi-Supervised Learning & Self-Training Pipeline for Indonesian Social Media Texts</div>', unsafe_allow_html=True)

    predictor = load_predictor()
    tables = load_tables()

    # Sidebar
    st.sidebar.title("Navigation & Settings")
    modules = [
        "Interactive Sentiment Predictor",
        "SSL Benchmark & Model Comparison",
        "Pseudo-Label Dynamics & Thresholds",
        "Dataset & Exploratory Analysis",
    ]
    default_idx = 0
    params = st.query_params
    if "tab" in params:
        try:
            val = int(params["tab"])
            if 0 <= val < len(modules):
                default_idx = val
        except ValueError:
            for i, m in enumerate(modules):
                if params["tab"].lower() in m.lower():
                    default_idx = i
                    break

    tab_selection = st.sidebar.radio(
        "Select Module:",
        modules,
        index=default_idx,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Specifications")
    st.sidebar.info(
        "**Model:** Self-Training Classifier\n\n"
        "**Base Estimator:** Logistic Regression (Balanced)\n\n"
        "**Features:** Word TF-IDF (1-2 ngrams) + Char TF-IDF (3-5 ngrams)\n\n"
        "**Labeled Budgets:** 10%, 20%, 30%\n\n"
        "**Metric:** Macro F1-score"
    )

    # -------------------------------------------------------------
    # TAB 1: INTERACTIVE PREDICTOR
    # -------------------------------------------------------------
    if tab_selection == "Interactive Sentiment Predictor":
        st.header("Real-Time Tweet Sentiment Inference")
        st.write("Enter an Indonesian tweet or colloquial social media statement to predict its sentiment and probability distribution.")

        col_input, col_preset = st.columns([2, 1])

        preset_text = ""
        with col_preset:
            st.markdown("##### Try Example Tweets:")
            if st.button("Positif: Pelayanan memuaskan"):
                preset_text = "Pelayanan sangat ramah, cepat banget sampainya dan packing rapi mantap jiwa!"
            if st.button("Negatif: Kecewa barang rusak"):
                preset_text = "Kecewa parah, barang rusak pas sampe dan sellernya ga mau tanggung jawab sama sekali!"
            if st.button("Netral: Informasi jadwal kuliah"):
                preset_text = "Tugas kuliah yg belum selesai harus dikumpulin besok pagi jam 8 di ruang dosen."
            if st.button("Gaul / Slang: Bahasanya santai"):
                preset_text = "Gila sih ini film keren bgt alur ceritanya ga ketebak sama sekali rekomen pol"

        with col_input:
            user_tweet = st.text_area(
                "Tweet Input:",
                value=preset_text if preset_text else "Pelayanan sangat ramah dan produknya bagus sekali, terima kasih!",
                height=120,
            )
            predict_btn = st.button("Analyze Sentiment", type="primary")

        if predictor is not None and (predict_btn or preset_text or user_tweet):
            res = predictor.predict(user_tweet)

            st.markdown("---")
            st.subheader("Prediction Analysis")

            m_col1, m_col2, m_col3 = st.columns([1, 1, 2])

            with m_col1:
                st.markdown("**Predicted Sentiment:**")
                if res["sentiment"] == "Positive":
                    st.markdown('<span class="sentiment-badge-pos">Positive (+1)</span>', unsafe_allow_html=True)
                elif res["sentiment"] == "Neutral":
                    st.markdown('<span class="sentiment-badge-neu">Neutral (0)</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="sentiment-badge-neg">Negative (-1)</span>', unsafe_allow_html=True)

            with m_col2:
                st.metric("Confidence Score", f"{res['confidence']:.2%}")

            with m_col3:
                st.markdown("**Probability Breakdown:**")
                for cls, prob in res["probabilities"].items():
                    color = "#2ecc71" if cls == "Positive" else ("#7f8c8d" if cls == "Neutral" else "#e74c3c")
                    st.write(f"{cls}: **{prob:.2%}**")
                    st.progress(float(prob))

        elif predictor is None:
            st.warning("Trained model artifact not found. Please train models using `python src/train.py`.")

    # -------------------------------------------------------------
    # TAB 2: BENCHMARK & COMPARISON
    # -------------------------------------------------------------
    elif tab_selection == "SSL Benchmark & Model Comparison":
        st.header("Semi-Supervised Learning Benchmark")
        st.write(
            "Evaluation comparing the **Supervised Baseline** against the **Semi-Supervised Self-Training** model "
            "across 10%, 20%, and 30% labeled training data budgets on an untouched test set (N=1,986)."
        )

        if "summary" in tables:
            df_sum = tables["summary"].copy()

            # High-level metric highlights
            c1, c2, c3, c4 = st.columns(4)
            sup_20 = df_sum[(df_sum["budget"] == 0.20) & (df_sum["model_type"] == "Supervised_Baseline")].iloc[0]
            ssl_20 = df_sum[(df_sum["budget"] == 0.20) & (df_sum["model_type"] == "SemiSupervised_SelfTraining")].iloc[0]
            oracle = df_sum[df_sum["model_type"] == "Supervised_Oracle_100pct"].iloc[0]

            with c1:
                st.metric("Supervised (20% Labeled)", f"{sup_20['macro_f1_mean']:.4f}", help="Macro F1 on test set")
            with c2:
                diff = ssl_20['macro_f1_mean'] - sup_20['macro_f1_mean']
                st.metric("Self-Training (20% Labeled)", f"{ssl_20['macro_f1_mean']:.4f}", delta=f"{diff:+.4f} F1", help="Macro F1 with 80% unlabeled pool")
            with c3:
                st.metric("Oracle (100% Labeled)", f"{oracle['macro_f1_mean']:.4f}", help="Theoretical upper bound")
            with c4:
                st.metric("Pseudo-Labels Added (20%)", f"{int(ssl_20['n_pseudo_mean']):,}", f"{ssl_20['pseudo_acc_mean']*100:.1f}% Accuracy")

            st.markdown("---")
            col_fig1, col_fig2 = st.columns([1, 1])
            with col_fig1:
                st.markdown("##### Macro F1-Score Comparison across Budgets")
                if os.path.exists("outputs/figures/fig5_macro_f1_comparison.png"):
                    st.image("outputs/figures/fig5_macro_f1_comparison.png", use_container_width=True)
            with col_fig2:
                st.markdown("##### Per-Class F1 Performance (20% Budget vs Oracle)")
                if os.path.exists("outputs/figures/fig10_per_class_f1_comparison.png"):
                    st.image("outputs/figures/fig10_per_class_f1_comparison.png", use_container_width=True)

            st.markdown("---")
            st.markdown("##### Quantitative Benchmark Summary Table")
            disp_df = df_sum.copy()
            disp_df["Budget"] = disp_df["budget"].apply(lambda b: f"{int(b*100)}%" if b < 1.0 else "100% (Oracle)")
            disp_df["Model"] = disp_df["model_type"].replace({
                "Supervised_Baseline": "Supervised Baseline",
                "SemiSupervised_SelfTraining": "Self-Training SSL",
                "Supervised_Oracle_100pct": "Supervised Oracle",
            })
            disp_df["Macro F1"] = disp_df.apply(lambda r: f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}" if r["macro_f1_std"] > 0 else f"{r['macro_f1_mean']:.4f}", axis=1)
            disp_df["Accuracy"] = disp_df.apply(lambda r: f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}" if r["accuracy_std"] > 0 else f"{r['accuracy_mean']:.4f}", axis=1)
            disp_df["F1 Negative"] = disp_df["f1_neg_mean"].apply(lambda v: f"{v:.4f}")
            disp_df["F1 Neutral"] = disp_df["f1_neu_mean"].apply(lambda v: f"{v:.4f}")
            disp_df["F1 Positive"] = disp_df["f1_pos_mean"].apply(lambda v: f"{v:.4f}")
            st.dataframe(disp_df[["Budget", "Model", "Macro F1", "Accuracy", "F1 Negative", "F1 Neutral", "F1 Positive"]], use_container_width=True)

            st.markdown("---")
            st.markdown("##### Confusion Matrix Inspection (20% Budget, Seed 42)")
            col_cm1, col_cm2 = st.columns(2)
            with col_cm1:
                if os.path.exists("outputs/figures/fig6_confusion_matrix_supervised.png"):
                    st.image("outputs/figures/fig6_confusion_matrix_supervised.png", use_container_width=True)
            with col_cm2:
                if os.path.exists("outputs/figures/fig7_confusion_matrix_selftraining.png"):
                    st.image("outputs/figures/fig7_confusion_matrix_selftraining.png", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: PSEUDO-LABEL DYNAMICS & THRESHOLDS
    # -------------------------------------------------------------
    elif tab_selection == "Pseudo-Label Dynamics & Thresholds":
        st.header("Pseudo-Label Dynamics & Confidence Sensitivity")
        st.write(
            "Detailed analysis of pseudo-label intake, class distribution, ground-truth pseudo-label accuracy, "
            "and confidence threshold sensitivity."
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("##### Pseudo-Label Acceptance & Class Breakdown")
            if os.path.exists("outputs/figures/fig8_pseudolabel_analysis.png"):
                st.image("outputs/figures/fig8_pseudolabel_analysis.png", use_container_width=True)
        with col_p2:
            st.markdown("##### Decision Threshold Sensitivity Curve (20% Budget)")
            if os.path.exists("outputs/figures/fig9_threshold_sensitivity.png"):
                st.image("outputs/figures/fig9_threshold_sensitivity.png", use_container_width=True)

        if "thresh" in tables:
            st.markdown("---")
            st.markdown("##### Threshold Sensitivity Data Table")
            df_th = tables["thresh"].copy()
            df_th["Threshold"] = df_th["threshold"].apply(lambda t: f"{t:.2f}")
            df_th["Macro F1"] = df_th["macro_f1"].apply(lambda v: f"{v:.4f}")
            df_th["Accuracy"] = df_th["accuracy"].apply(lambda v: f"{v:.4f}")
            df_th["Pseudo Added"] = df_th["n_pseudo_labeled"].apply(lambda v: f"{int(v):,}")
            df_th["Pseudo Accuracy"] = df_th["pseudo_label_accuracy"].apply(lambda v: f"{v*100:.2f}%")
            st.dataframe(df_th[["Threshold", "Macro F1", "Accuracy", "Pseudo Added", "Pseudo Accuracy", "actual_iterations", "pseudo_neg", "pseudo_neu", "pseudo_pos"]], use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: DATASET & EDA EXPLORER
    # -------------------------------------------------------------
    elif tab_selection == "Dataset & Exploratory Analysis":
        st.header("Dataset Quality Audit & Exploratory Data Analysis")
        st.write(
            "Exploration of the 10,806 tweet corpus, duplicate structure, conflicting labels, "
            "length distributions, and lexical patterns."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Dataset Size", "10,806 rows", "Tab-delimited")
        c2.metric("Exact Duplicates", "549 rows", "Removed")
        c3.metric("Conflicting Label Groups", "156 texts", "422 rows excluded")
        c4.metric("Clean Deduplicated Tweets", "9,926 unique", "Zero text leakage")

        st.markdown("---")
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("##### Class Distribution Comparison (Raw vs Clean)")
            if os.path.exists("outputs/figures/fig1_class_distribution.png"):
                st.image("outputs/figures/fig1_class_distribution.png", use_container_width=True)
        with col_eda2:
            st.markdown("##### Character and Word Length Distributions")
            if os.path.exists("outputs/figures/fig2_tweet_length_distribution.png"):
                st.image("outputs/figures/fig2_tweet_length_distribution.png", use_container_width=True)

        st.markdown("---")
        col_eda3, col_eda4 = st.columns([1, 1.2])
        with col_eda3:
            st.markdown("##### Tweet Length by Sentiment Class")
            if os.path.exists("outputs/figures/fig3_length_by_sentiment.png"):
                st.image("outputs/figures/fig3_length_by_sentiment.png", use_container_width=True)
        with col_eda4:
            st.markdown("##### Top N-grams / Terms per Sentiment Class")
            if os.path.exists("outputs/figures/fig4_top_ngrams_by_sentiment.png"):
                st.image("outputs/figures/fig4_top_ngrams_by_sentiment.png", use_container_width=True)

        if "test_sample" in tables:
            st.markdown("---")
            st.markdown("##### Holdout Test Set Sample Explorer (N=1,986)")
            test_df = tables["test_sample"].copy()
            sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Negative (-1)", "Neutral (0)", "Positive (1)"])
            if sentiment_filter != "All":
                val = int(sentiment_filter.split("(")[1].replace(")", ""))
                test_df = test_df[test_df["sentimen"] == val]

            st.dataframe(test_df[["sentimen", "Tweet"]].head(50), use_container_width=True)


if __name__ == "__main__":
    main()
