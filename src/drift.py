"""
Drift monitoring proposal: PSI (Population Stability Index) and KS test
scheduled checks on input feature distributions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between reference (expected) and current (actual)
    distributions for a numeric feature.
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    expected_counts = np.histogram(expected, bins=expected_perc)[0]
    actual_counts = np.histogram(actual, bins=expected_perc)[0]

    # Avoid division by zero
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)

    expected_dist = expected_counts / expected_counts.sum()
    actual_dist = actual_counts / actual_counts.sum()

    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))


def check_drift(reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str, dict]:
    """
    Compare current feature distribution against reference (training) distribution.
    Returns per-feature PSI and KS test results with drift flags.
    """
    results = {}
    for col in reference.select_dtypes(include=[np.number]).columns:
        if col not in current.columns:
            continue
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        psi_val = psi(ref_vals, cur_vals)
        ks_stat, ks_p = ks_2samp(ref_vals, cur_vals)

        results[col] = {
            "psi": round(psi_val, 4),
            "psi_drift": psi_val > 0.25,   # common rule-of-thumb threshold
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "ks_drift": ks_p < 0.05,
        }
    return results