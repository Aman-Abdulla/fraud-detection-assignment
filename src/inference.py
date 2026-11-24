"""
Inference utilities: load trained pipeline, score single/batch transactions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
import pandas as pd


def load_pipeline(model_path: str | Path):
    """Load a persisted scikit-learn / LightGBM pipeline."""
    return joblib.load(model_path)


def predict_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    """Return fraud probabilities for a feature DataFrame."""
    return pipe.predict_proba(X)[:, 1]


def predict(pipe, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Return binary fraud predictions using a decision threshold."""
    return (predict_proba(pipe, X) >= threshold).astype(int)