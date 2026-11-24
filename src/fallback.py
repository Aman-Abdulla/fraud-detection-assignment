"""
Fallback mechanisms for production inference (criterion 4).

When the model service is unavailable or returns an error, fall back to
a rule-based heuristic so the system never goes fully dark.
"""

from __future__ import annotations

from typing import Optional


def rule_based_fallback(
    amount: float,
    foreign_transaction: int,
    location_mismatch: int,
    device_trust_score: float,
    velocity_last_24h: int,
) -> dict:
    """
    Simple rule-based fraud flag when ML model is unavailable.
    Returns a prediction with reason codes for auditability.
    """
    risk_score = 0
    reasons: list[str] = []

    if amount > 1000:
        risk_score += 2
        reasons.append("high_amount")
    if foreign_transaction == 1:
        risk_score += 1
        reasons.append("foreign_transaction")
    if location_mismatch == 1:
        risk_score += 2
        reasons.append("location_mismatch")
    if device_trust_score < 40:
        risk_score += 2
        reasons.append("low_device_trust")
    if velocity_last_24h > 5:
        risk_score += 1
        reasons.append("high_velocity")

    is_fraud = int(risk_score >= 3)
    return {
        "fraud_probability": None,  # model unavailable
        "is_fraud": is_fraud,
        "fallback": True,
        "risk_score": risk_score,
        "reasons": reasons,
    }