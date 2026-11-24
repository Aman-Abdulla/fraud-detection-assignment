"""
FastAPI inference service for fraud detection.

Endpoints:
  GET  /health          - service health check
  POST /predict         - single transaction inference
  POST /predict-batch   - batch inference (list of transactions)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud Detection API", version="1.0.0")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_pipeline.joblib"
pipe = None


class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    merchant_category: str = Field(..., description="Merchant category")
    foreign_transaction: int = Field(..., ge=0, le=1, description="1 if foreign, 0 otherwise")
    location_mismatch: int = Field(..., ge=0, le=1, description="1 if location mismatch, 0 otherwise")
    device_trust_score: float = Field(..., ge=0, le=100, description="Device trust score 0-100")
    velocity_last_24h: int = Field(..., ge=0, description="Transaction count in last 24h")
    cardholder_age: int = Field(..., ge=0, le=120, description="Cardholder age")


class PredictionResponse(BaseModel):
    transaction_id: int | None = None
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: int = Field(..., ge=0, le=1)
    threshold: float = 0.5


@app.on_event("startup")
def load_model() -> None:
    global pipe
    if MODEL_PATH.exists():
        pipe = joblib.load(MODEL_PATH)
    else:
        # Model not trained yet; service starts in stub mode
        pipe = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": pipe is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction) -> PredictionResponse:
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([txn.model_dump()])
    tid = df.pop("transaction_id", None) if "transaction_id" in df.columns else None
    proba = float(pipe.predict_proba(df)[:, 1][0])
    is_fraud = int(proba >= 0.5)

    return PredictionResponse(
        transaction_id=int(tid) if tid is not None else None,
        fraud_probability=round(proba, 4),
        is_fraud=is_fraud,
    )


@app.post("/predict-batch")
def predict_batch(txns: List[Transaction]) -> List[PredictionResponse]:
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([t.model_dump() for t in txns])
    tids = df.pop("transaction_id", None) if "transaction_id" in df.columns else None
    probas = pipe.predict_proba(df)[:, 1]

    results = []
    for i, proba in enumerate(probas):
        results.append(
            PredictionResponse(
                transaction_id=int(tids[i]) if tids is not None else None,
                fraud_probability=round(float(proba), 4),
                is_fraud=int(proba >= 0.5),
            )
        )
    return results