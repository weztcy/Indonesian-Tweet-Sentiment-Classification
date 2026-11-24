"""
predict.py
Inference module for Indonesian Tweet Sentiment Classification.
Provides programmatic and command-line interfaces for real-time predictions.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import joblib
import numpy as np
import pandas as pd

from src.data_preparation import MAPPED_LABEL_NAMES, INV_LABEL_MAP


class TweetSentimentPredictor:
    """
    Production-ready sentiment predictor wrapping the fitted feature extractor
    and trained semi-supervised model.
    """

    def __init__(self, model_bundle_path: str = "outputs/models/best_model.joblib"):
        if not os.path.exists(model_bundle_path):
            raise FileNotFoundError(f"Trained model not found at: {model_bundle_path}. Please train first.")

        bundle = joblib.load(model_bundle_path)
        self.model = bundle["model"]
        self.feature_extractor = bundle["feature_extractor"]
        self.budget = bundle.get("budget", 0.20)
        self.seed = bundle.get("seed", 42)
        self.metrics = bundle.get("metrics", {})
        self.class_labels = [MAPPED_LABEL_NAMES[i] for i in range(3)]

    def predict(self, text_or_list):
        """
        Predicts sentiment class and probability distribution.
        """
        is_single = isinstance(text_or_list, str)
        texts = [text_or_list] if is_single else list(text_or_list)

        # Transform features
        X_vec = self.feature_extractor.transform(texts)

        # Get predicted mapped class (0, 1, 2)
        preds_mapped = self.model.predict(X_vec)
        # Get probability distributions
        probas = self.model.predict_proba(X_vec)

        results = []
        for text, mapped_cls, prob_vec in zip(texts, preds_mapped, probas):
            orig_label = INV_LABEL_MAP[mapped_cls]
            label_name = MAPPED_LABEL_NAMES[mapped_cls]
            confidence = float(np.max(prob_vec))
            results.append({
                "tweet": text,
                "label": int(orig_label),
                "sentiment": label_name,
                "confidence": confidence,
                "probabilities": {
                    "Negative": float(prob_vec[0]),
                    "Neutral": float(prob_vec[1]),
                    "Positive": float(prob_vec[2]),
                },
            })

        return results[0] if is_single else results


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of Indonesian tweets.")
    parser.add_argument("--tweet", type=str, help="Single tweet text to predict.")
    parser.add_argument("--file", type=str, help="Path to text or CSV file for batch prediction.")
    parser.add_argument("--model", type=str, default="outputs/models/best_model.joblib", help="Path to model bundle.")
    args = parser.parse_args()

    predictor = TweetSentimentPredictor(args.model)

    if args.tweet:
        res = predictor.predict(args.tweet)
        print(f"\nTweet: {res['tweet']}")
        print(f"Sentiment: {res['sentiment']} (Label: {res['label']})")
        print(f"Confidence: {res['confidence']:.2%}")
        print("Class Probabilities:")
        for k, v in res["probabilities"].items():
            print(f"  - {k}: {v:.2%}")
    elif args.file:
        df = pd.read_csv(args.file)
        col = "Tweet" if "Tweet" in df.columns else df.columns[0]
        preds = predictor.predict(df[col].tolist())
        out_df = pd.DataFrame(preds)
        out_path = "outputs/predictions/batch_predictions.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Batch predictions saved to: {out_path}")
    else:
        # Interactive mode
        print("Interactive Sentiment Predictor (Type 'exit' to quit):")
        while True:
            text = input("\nEnter tweet text: ")
            if text.strip().lower() in ["exit", "quit", "q"]:
                break
            res = predictor.predict(text)
            print(f"-> Sentiment: {res['sentiment']} ({res['confidence']:.1%}) | P(Neg): {res['probabilities']['Negative']:.2f}, P(Neu): {res['probabilities']['Neutral']:.2f}, P(Pos): {res['probabilities']['Positive']:.2f}")


if __name__ == "__main__":
    main()
