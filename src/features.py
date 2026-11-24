"""
Feature engineering: encode categoricals, scale numerics, build X/y matrices.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Return a ColumnTransformer that scales numerics and one-hot encodes categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )


def split_xy(
    df: pd.DataFrame, target: str = "is_fraud"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features X and target y."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y