"""
Model training and evaluation: LightGBM (or sklearn fallback),
PR-AUC primary metric, F1-macro secondary metric.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    random_state: int = 42,
) -> Pipeline:
    """Train a LightGBM classifier inside a scikit-learn Pipeline."""
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Return PR-AUC, F1-macro, and classification report.
    PR-AUC is the primary metric for imbalanced fraud detection.
    """
    y_proba = pipe.predict_proba(X)[:, 1]
    y_pred = pipe.predict(X)

    pr_auc = average_precision_score(y, y_proba)
    f1_macro = f1_score(y, y_pred, average="macro")

    return {
        "pr_auc": float(pr_auc),
        "f1_macro": float(f1_macro),
        "report": classification_report(y, y_pred, output_dict=True),
    }