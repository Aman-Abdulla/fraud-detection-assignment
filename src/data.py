"""
Data loading, train/validation/test splitting, and leakage checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str | Path) -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    return pd.read_csv(path)


def check_leakage(df: pd.DataFrame, target: str = "is_fraud") -> None:
    """
    Sanity check: ensure no column is a perfect proxy for the target.
    Raises ValueError if a feature perfectly predicts the label.
    """
    for col in df.columns:
        if col == target:
            continue
        if df[col].nunique() == 1:
            continue
        # Perfect correlation between a feature and target indicates leakage
        try:
            corr = df[col].corr(df[target])
            if abs(corr) > 0.99:
                raise ValueError(
                    f"Potential leakage: column '{col}' correlates {corr:.4f} with target '{target}'"
                )
        except Exception:
            # Non-numeric column; skip numeric correlation check
            continue


def stratified_split(
    df: pd.DataFrame,
    target: str = "is_fraud",
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test while preserving class balance.
    Returns (train, val, test).
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[target],
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)