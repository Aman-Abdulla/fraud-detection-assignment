"""
Benchmarking script: fires 100+ requests through the inference API (or scores
a batch dataset locally) and reports p50 / p95 / p99 latency plus throughput.

Usage:
    python src/evaluate.py --mode api        # hits running FastAPI server
    python src/evaluate.py --mode local      # scores local CSV through pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

from src.inference import load_pipeline, predict_proba


def run_api_benchmark(url: str, n_requests: int = 100) -> dict:
    """Send n_requests single-transaction POSTs to /predict and time them."""
    sample_txn = {
        "amount": 247.50,
        "transaction_hour": 22,
        "merchant_category": "Electronics",
        "foreign_transaction": 1,
        "location_mismatch": 0,
        "device_trust_score": 55.0,
        "velocity_last_24h": 2,
        "cardholder_age": 34,
    }

    latencies: List[float] = []
    successes = 0

    for i in range(n_requests):
        start = time.perf_counter()
        try:
            resp = requests.post(f"{url}/predict", json=sample_txn, timeout=10)
            if resp.status_code == 200:
                successes += 1
        except Exception:
            pass
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mode": "api",
        "n_requests": n_requests,
        "successes": successes,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(latencies.mean()),
            "max": float(latencies.max()),
        },
        "records_per_sec": round(successes / (latencies.sum() / 1000), 2) if latencies.sum() > 0 else 0,
    }


def run_local_benchmark(csv_path: str | Path, model_path: str | Path, n_rows: int = 100) -> dict:
    """Score n_rows from a local CSV through the persisted pipeline."""
    df = pd.read_csv(csv_path).head(n_rows)
    pipe = load_pipeline(model_path)

    X = df.drop(columns=["is_fraud"], errors="ignore")

    latencies: List[float] = []
    for i in range(len(X)):
        start = time.perf_counter()
        predict_proba(pipe, X.iloc[[i]])
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "mode": "local",
        "n_requests": n_rows,
        "latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "mean": float(latencies.mean()),
            "max": float(latencies.max()),
        },
        "records_per_sec": round(len(X) / (latencies.sum() / 1000), 2) if latencies.sum() > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud detection benchmarking")
    parser.add_argument("--mode", choices=["api", "local"], default="local")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--csv", default="data/raw/credit_card_fraud_10k.csv")
    parser.add_argument("--model", default="models/fraud_pipeline.joblib")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "api":
        result = run_api_benchmark(args.url, args.n)
    else:
        result = run_local_benchmark(args.csv, args.model, args.n)

    print(json.dumps(result, indent=2))

    out_path = Path("reports/benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()