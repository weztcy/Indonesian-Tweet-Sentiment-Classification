# Portfolio Project Data — Indonesian Tweet Sentiment Classification

- **Category:** `machine-learning` (`Machine Learning`)
- **Accent Color:** `indigo`
- **Target Assets Folder:** `public/images/projects/indonesian_tweet_sentiment_classification/`
- **Required Local Images:** `banner.webp`, `image01.webp` .. `image09.webp`

## TypeScript Data (`ProjectCaseStudy`)

```typescript
{
  "id": "indonesian-tweet-sentiment-classification",
  "code": "PRO-28 // ML-13",
  "title": "Indonesian Tweet Sentiment Classification",
  "category": "machine-learning",
  "categoryLabel": "Semi-Supervised Learning",
  "year": "2024",
  "status": "Completed",
  "accentColor": "indigo",
  "repoUrl": "https://github.com/weztcy/Indonesian-Tweet-Sentiment-Classification",
  "license": "MIT License",
  "banner": "/images/projects/indonesian_tweet_sentiment_classification/banner.webp",
  "summary": "Empirical study of Semi-Supervised Self-Training vs. Logistic Regression baselines for Indonesian tweets under 10%, 20%, and 30% labeled budgets.",
  "projectDescription": "Leakage-free semi-supervised NLP pipeline evaluating iterative Self-Training against supervised Logistic Regression on 9,926 deduplicated Indonesian tweets. Uses hybrid word and character TF-IDF features across 10% to 30% labeled budgets with a Streamlit analytics dashboard.",
  "architectureSummary": "Raw TSV Audit & Conflict Isolation -> Stratified 80/20 Split -> Hybrid Word/Char TF-IDF FeatureUnion -> SelfTrainingClassifier (LogisticRegression) -> Holdout Evaluation & Streamlit Dashboard",
  "techStack": [
    "Python",
    "Scikit-Learn",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Streamlit",
    "Joblib",
    "Tabulate"
  ],
  "telemetryMatrix": {
    "projectType": "Semi-Supervised NLP Pipeline",
    "targetPlatform": "Python CLI & Streamlit Web App",
    "primaryDomain": "Indonesian Social Media Sentiment",
    "architecturePipeline": "Hybrid TF-IDF & Self-Training",
    "testingAndAudit": "Verified",
    "projectStatus": "Completed",
    "projectYear": "2024"
  },
  "mlProjectInfo": {
    "projectType": "Semi-Supervised Text Classification",
    "learningParadigm": "Semi-Supervised Learning",
    "modelDomain": "Indonesian Twitter Sentiment Analysis",
    "projectStatus": "Completed",
    "developmentYear": "2024",
    "capabilityBadges": [
      "Iterative Self-Training",
      "Hybrid TF-IDF Features",
      "Zero Text Leakage"
    ]
  },
  "mlProblemContext": "Manual sentiment annotation for colloquial Indonesian Twitter text is costly due to slang, regional dialects, abbreviations, and sarcasm. This study rigorously tests whether Self-Training on unlabeled tweets improves three-class sentiment F1 under scarce 10% to 30% labeled budgets without text leakage.",
  "mlDatasetOverview": {
    "profileBadge": "10,806 Raw / 9,926 Clean",
    "source": "Indonesian Sentiment Twitter Dataset Labeled (Tab-Delimited TSV)",
    "modality": "Text (NLP)",
    "scale": "9,926 Unique Deduplicated Tweets (7,940 Train / 1,986 Holdout Test)",
    "inputShape": "97,183 Sparse Word + Character TF-IDF Features",
    "dataFormat": "TSV / CSV",
    "description": "Audited from 10,806 raw tweets by removing 549 exact duplicates and isolating 156 conflicting normalized text groups (422 rows), yielding 9,926 unique tweets stratified into an 80% training pool and a sealed 20% holdout test set.",
    "provenanceNote": "Public Indonesian Twitter sentiment corpus audited locally; secondary majority-vote conflict experiment included for cleaning sensitivity verification."
  },
  "mlFeaturesAndDataCharacteristics": {
    "spaceBadge": "97,183 Sparse N-Grams",
    "spaceNote": "Dual-stream word (1-2) and character (3-5) TF-IDF matrix preserving Indonesian negations and affixes",
    "objectiveOutput": {
      "label": "Target Output",
      "name": "sentimen",
      "type": "Multi-Class (3 Polarities)",
      "distribution": "Negative (-1): 26.19%, Neutral (0): 49.84%, Positive (1): 23.97%"
    },
    "inputSignals": [
      {
        "name": "Word TF-IDF (1-2 N-Grams)",
        "type": "Sparse Text Feature",
        "importanceOrRole": "Primary Lexical & Phrase Signal",
        "description": "Sublinear word unigrams and bigrams (min_df=2) capturing sentiment words, negation phrases like tidak suka, and colloquial expressions."
      },
      {
        "name": "Character TF-IDF (3-5 N-Grams)",
        "type": "Sparse Subword Feature",
        "importanceOrRole": "Morphological & Slang Robustness",
        "description": "Character substring n-grams (min_df=2, sublinear_tf=True) capturing Indonesian affixes, character elongation, typos, and abbreviations."
      },
      {
        "name": "Normalized Deduplication Key",
        "type": "Audit Grouping Signal",
        "importanceOrRole": "Zero-Leakage Split Guard",
        "description": "Lowercased and whitespace-collapsed text representation used exclusively during preparation to eliminate cross-split overlap and label conflicts."
      }
    ],
    "dataCharacteristicsSummary": "Clean tweets range from 2 to 140 characters (mean 80.4, median 91.0) and 1 to 28 words (mean 13.6, median 15.0). Nearly half of all records are Neutral (49.84%), creating class imbalance and substantial vocabulary overlap around colloquial negation markers."
  },
  "mlLearningApproach": {
    "paradigm": "Semi-Supervised Learning via Iterative Self-Training",
    "methodologyDescription": "Labels are mapped from {-1, 0, 1} to {0, 1, 2} so -1 marks unlabeled samples. Stratified budgets of 10%, 20%, and 30% train a class-balanced Logistic Regression base estimator inside SelfTrainingClassifier over 3 random seeds, benchmarked against supervised baselines and a 100% Oracle.",
    "justification": "Balanced multinomial Logistic Regression delivers well-calibrated class probabilities and fast convex convergence on 97,183 sparse TF-IDF dimensions, enabling selective high-confidence pseudo-labeling without deep transformer compute overhead."
  },
  "mlDataPreparation": [
    {
      "step": "01",
      "title": "Tab-Delimited Schema Audit",
      "description": "Loaded 10,806 TSV tweet records with UTF-8/Latin-1 fallback, validated zero missing values, and profiled 549 exact duplicate rows.",
      "badge": "Schema Audit"
    },
    {
      "step": "02",
      "title": "Conflict Group Isolation",
      "description": "Normalized text via lowercasing and whitespace collapsing to detect and exclude 156 conflicting text groups spanning 422 rows.",
      "badge": "Conflict Removal"
    },
    {
      "step": "03",
      "title": "Deduplication & Stratified Split",
      "description": "Deduplicated to 9,926 unique tweets and partitioned into 7,940 train and 1,986 sealed holdout test tweets with zero overlap.",
      "badge": "80/20 Split"
    },
    {
      "step": "04",
      "title": "Budget Masking & FeatureUnion",
      "description": "Masked 70% to 90% of training labels as -1 per seed and fitted the 97,183-dimensional word plus character TF-IDF pipeline strictly on training text.",
      "badge": "TF-IDF Fit"
    }
  ],
  "mlDataPreparationNote": "Test tweets (N=1,986) are isolated before label masking and TF-IDF fitting, and normalized text deduplication guarantees zero text overlap between partitions.",
  "mlModelArchitectureAndTraining": {
    "algorithmName": "Self-Training Classifier (Logistic Regression Base)",
    "architectureSummary": "Sklearn FeatureUnion concatenates Word TF-IDF (1-2) and Char TF-IDF (3-5) into a 97,183-dimensional sparse vector fed to a class-balanced multinomial Logistic Regression estimator wrapped in SelfTrainingClassifier (tau=0.80, max_iter=10).",
    "configurationBadge": "L-BFGS & Tau 0.80",
    "hyperparameters": [
      {
        "name": "C",
        "value": "1.0",
        "description": "Inverse L2 regularization strength for the base Logistic Regression classifier."
      },
      {
        "name": "class_weight",
        "value": "balanced",
        "description": "Inversely weights classes by training frequency to counter the 49.84% Neutral majority."
      },
      {
        "name": "threshold",
        "value": "0.80",
        "description": "Minimum predicted probability required to promote an unlabeled tweet into the pseudo-labeled pool."
      },
      {
        "name": "max_iter (SSL / L-BFGS)",
        "value": "10 / 1000",
        "description": "Caps iterative self-training rounds at 10 and base solver iterations at 1000."
      },
      {
        "name": "ngram_range (Word / Char)",
        "value": "(1, 2) / (3, 5)",
        "description": "Extracts lexical unigrams/bigrams alongside subword character sequences (min_df=2, sublinear_tf=True)."
      }
    ],
    "objectiveCriterion": "Class-Weighted Multinomial Cross-Entropy with L2 Penalty",
    "optimizationSolver": "L-BFGS (Base Estimator) + Confidence-Thresholded Self-Training Loop",
    "trainingProcess": "Each experiment trains across seeds 42, 123, and 456 for 10%, 20%, and 30% budgets. In each round, the L-BFGS model scores unlabeled tweets and promotes predictions exceeding threshold 0.80, achieving 86.54% pseudo-label accuracy at 20% and 86.01% at 30%."
  },
  "mlEvaluationStrategy": {
    "validationMethod": "3-Seed Stratified Budgets on Sealed 20% Holdout Test Set",
    "description": "Evaluated Supervised Baseline vs. Self-Training on 1,986 untouched holdout tweets across 10%, 20%, and 30% budgets (seeds 42, 123, 456), benchmarked against a 100% Oracle, threshold sweeps (0.70-0.90), and majority-vote conflict sensitivity.",
    "metrics": [
      {
        "label": "20% SSL MACRO F1",
        "value": "0.5330",
        "benchmark": "Baseline: 0.5294"
      },
      {
        "label": "30% SSL MACRO F1",
        "value": "0.5510",
        "benchmark": "Baseline: 0.5478"
      },
      {
        "label": "20% SSL ACCURACY",
        "value": "57.27%",
        "benchmark": "Baseline: 57.00%"
      },
      {
        "label": "30% SSL ACCURACY",
        "value": "58.43%",
        "benchmark": "Baseline: 58.12%"
      },
      {
        "label": "20% PSEUDO ACC",
        "value": "86.54%",
        "benchmark": "Target > 80.0%"
      },
      {
        "label": "100% ORACLE F1",
        "value": "0.5990",
        "benchmark": "Upper: 0.5990"
      }
    ]
  },
  "mlPerformanceVisualizations": [
    {
      "title": "Real-Time Tweet Sentiment Inference Module",
      "badge": "Streamlit Inference",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image01.webp",
      "telemetryNote": "Live dark-mode prediction UI displaying Positive (+1) classification at 70.62% confidence and probability bars."
    },
    {
      "title": "Semi-Supervised Learning Benchmark Dashboard",
      "badge": "Budget Benchmark",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image02.webp",
      "telemetryNote": "20% budget Self-Training reaches 0.5330 Macro F1 (+0.0036 over baseline) with 86.5% pseudo-label accuracy."
    },
    {
      "title": "Pseudo-Label Dynamics & Threshold Dashboard",
      "badge": "SSL Dynamics",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image03.webp",
      "telemetryNote": "Threshold sweep table and plots tracking pseudo-label volume, class share, and precision from 0.70 to 0.90."
    },
    {
      "title": "Dataset Quality Audit & EDA Dashboard",
      "badge": "Corpus Audit",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image04.webp",
      "telemetryNote": "Summarizes 10,806 raw rows, 549 duplicates, 156 conflict groups (422 rows), and 9,926 unique tweets."
    },
    {
      "title": "Threshold Sensitivity & Pseudo-Label Intake",
      "badge": "Sensitivity Plot",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image05.webp",
      "telemetryNote": "733 pseudo-labels added at tau=0.70 (77.76% acc) vs. 23 at tau=0.80 (95.65% acc) and label starvation above 0.85."
    },
    {
      "title": "Tweet Length & Top Sentiment N-Grams",
      "badge": "Lexical EDA",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image06.webp",
      "telemetryNote": "Clean corpus length averages 80.4 characters and 13.6 words alongside preserved negation terms across classes."
    },
    {
      "title": "Class Balance & Length-by-Sentiment Boxplots",
      "badge": "Distribution Plot",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image07.webp",
      "telemetryNote": "Verifies stable class proportions (26.2% Neg, 49.8% Neu, 24.0% Pos) after deduplication and conflict removal."
    },
    {
      "title": "Macro F1 Across Budgets & Per-Class Breakdown",
      "badge": "F1 Comparison",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image08.webp",
      "telemetryNote": "Self-Training outperforms baseline at 20% and 30% budgets and raises per-class F1 across all three sentiments."
    },
    {
      "title": "Normalized Confusion Matrices (20% Budget)",
      "badge": "Confusion Matrix",
      "imageUrl": "/images/projects/indonesian_tweet_sentiment_classification/image09.webp",
      "telemetryNote": "Neutral acts as main error sink (46.73% of Negative, 38.66% of Positive); SSL lifts Negative recall to 40.96%."
    }
  ],
  "mlVisualizationTelemetryNote": "All benchmark charts evaluate untouched holdout test tweets (N=1,986) averaged across seeds 42, 123, and 456.",
  "mlModelInterpretationAndLimitations": {
    "interpretationSummary": "Self-Training consistently adds value once the base classifier reaches sufficient calibration at 20% to 30% labeled budgets, lifting recall across all classes while achieving over 86% pseudo-label precision. However, 10% budgets suffer threshold starvation and Neutral remains the main error sink.",
    "keyDrivers": [
      {
        "factor": "Initial Labeled Budget Threshold (>= 20%)",
        "impact": "Raises mean accepted pseudo-labels from 2 (10% budget) to 43 (20%) and 86 (30%) with >86% ground-truth precision."
      },
      {
        "factor": "Confidence Decision Threshold (tau)",
        "impact": "Balances exploration and noise: tau=0.70 admits 733 samples (0.5322 F1, 77.76% acc) while tau=0.80 yields 95.65% acc on seed 42."
      },
      {
        "factor": "Hybrid Word + Char TF-IDF Features",
        "impact": "97,183 n-gram dimensions capture informal Indonesian abbreviations, affixes, and negated phrases without destructive stemming."
      }
    ],
    "limitations": [
      {
        "title": "Ultra-Low Budget Starvation (10%)",
        "description": "With 794 labeled tweets, uncalibrated probabilities rarely exceed 0.80, yielding virtually zero pseudo-label gain (0.4866 vs 0.4863 Macro F1)."
      },
      {
        "title": "Neutral Majority Error Sink",
        "description": "46.73% of Negative and 38.66% of Positive tweets are predicted as Neutral due to subtle Indonesian phrasing and sarcasm beyond linear TF-IDF."
      },
      {
        "title": "Simulated Unlabeled Pool",
        "description": "Unlabeled samples were masked from an annotated Twitter corpus rather than drawn from noisy real-time streams with spam or domain drift."
      }
    ],
    "futureImprovements": [
      {
        "title": "Class-Adaptive Confidence Thresholding",
        "target": "Replace fixed tau=0.80 with curriculum or per-class thresholds to boost Negative and Positive pseudo-label intake."
      },
      {
        "title": "IndoBERT Backbone & Active Learning",
        "target": "Combine pretrained Indonesian transformer embeddings with uncertainty sampling for sarcastic and borderline tweets."
      }
    ],
    "discoveryNote": "Audit confirmed 0 text overlap between train and test splits and consistent SSL gains under both conflict exclusion and majority-vote cleaning."
  },
  "mlModelArtifacts": [
    {
      "name": "best_model.joblib",
      "format": "JOBLIB"
    },
    {
      "name": "train_set.csv",
      "format": "CSV"
    },
    {
      "name": "test_set.csv",
      "format": "CSV"
    },
    {
      "name": "budget_experiments_summary.csv",
      "format": "CSV"
    }
  ],
  "mlMethodologySpecs": {
    "learningParadigm": "Semi-Supervised Self-Training",
    "modelArchitecture": "Multinomial Logistic Regression",
    "coreFramework": "Scikit-Learn",
    "validationMethod": "3-Seed Holdout Evaluation"
  }
}
```
