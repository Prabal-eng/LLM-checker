"""Trains an In-Domain AI Provenance Classifier on the extracted evaluation dataset."""
import os
import sys
import re
import json
import math
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from src.qa.stylometry import StylometricFeatureExtractor

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "training")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_evaluate():
    print("[1/4] Loading in-domain datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train, y_train = train_df["text"].values, train_df["label"].values
    X_val, y_val = val_df["text"].values, val_df["label"].values
    X_test, y_test = test_df["text"].values, test_df["label"].values

    print(f"      Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n[2/4] Building hybrid feature pipeline (Word/Char N-Grams + Stylometry)...")
    feature_union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=12000, sublinear_tf=True)),
        ("stylometry", StylometricFeatureExtractor())
    ])

    base_classifier = LogisticRegression(C=2.5, max_iter=1000, class_weight="balanced", random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator=base_classifier, method="sigmoid", cv=5)

    pipeline = Pipeline([
        ("features", feature_union),
        ("clf", calibrated_model)
    ])

    print("\n[3/4] Training In-Domain Provenance Model on evaluation justifications...")
    pipeline.fit(X_train, y_train)

    print("\n[4/4] Evaluating Model on Held-Out Test Set (700 rows)...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)  # False Positive Rate (human wrongly flagged as AI)

    print("\n" + "=" * 60)
    print("      IN-DOMAIN MODEL TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f" Test Accuracy              : {acc * 100:.2f}%")
    print(f" ROC-AUC Score              : {roc_auc:.4f}")
    print(f" False Positive Rate (Human): {fpr * 100:.2f}% ({fp} human texts flagged out of {tn + fp})")
    print(f" Recall on AI Text          : {tp / (tp + fn) * 100:.2f}% ({tp} detected out of {tp + fn})")
    print("-" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"]))

    # Save trained model artifact
    model_path = os.path.join(MODEL_DIR, "indomain_ai_classifier.joblib")
    joblib.dump(pipeline, model_path, compress=3)

    meta = {
        "model_type": "Hybrid Stylometric & N-Gram Classifier (Calibrated)",
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_file": model_path
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[OK] Trained Model successfully exported to: {model_path}")
    print(f"[OK] Model metadata saved to: {os.path.join(MODEL_DIR, 'model_meta.json')}\n")

    return acc, fpr


if __name__ == "__main__":
    train_and_evaluate()
