"""
Train fraud detection model:
- Load processed train/val/test splits
- Build preprocessing pipeline
- Train LightGBM classifier
- Evaluate with PR-AUC (primary) and F1-macro (secondary)
- Save trained pipeline to models/fraud_pipeline.joblib
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import stratified_split
from src.features import build_preprocessor, split_xy
from src.model import evaluate_model, train_model


def main() -> None:
    # --- Paths ---
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw data and create splits (if not already present) ---
    raw_path = Path("data/raw/credit_card_fraud_10k.csv")
    train_path = processed_dir / "train.csv"

    if not train_path.exists():
        from src.data import load_data

        df = load_data(raw_path)
        train, val, test = stratified_split(df, target="is_fraud", test_size=0.2, val_size=0.25, random_state=42)
        processed_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(processed_dir / "train.csv", index=False)
        val.to_csv(processed_dir / "val.csv", index=False)
        test.to_csv(processed_dir / "test.csv", index=False)
        print("Created train/val/test splits.")
    else:
        train = pd.read_csv(processed_dir / "train.csv")
        val = pd.read_csv(processed_dir / "val.csv")
        test = pd.read_csv(processed_dir / "test.csv")
        print("Loaded existing splits.")

    # --- Features ---
    target = "is_fraud"
    numeric_features = ["amount", "transaction_hour", "device_trust_score", "velocity_last_24h", "cardholder_age"]
    categorical_features = ["merchant_category", "foreign_transaction", "location_mismatch"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, y_train = split_xy(train, target)
    X_val, y_val = split_xy(val, target)
    X_test, y_test = split_xy(test, target)

    # --- Train ---
    print("Training LightGBM model...")
    pipe = train_model(X_train, y_train, preprocessor, random_state=42)

    # --- Evaluate ---
    val_metrics = evaluate_model(pipe, X_val, y_val)
    test_metrics = evaluate_model(pipe, X_test, y_test)

    print("\nValidation:")
    print(f"  PR-AUC: {val_metrics['pr_auc']:.4f}")
    print(f"  F1-macro: {val_metrics['f1_macro']:.4f}")

    print("\nTest:")
    print(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"  F1-macro: {test_metrics['f1_macro']:.4f}")

    # --- Save model ---
    model_path = models_dir / "fraud_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nSaved trained pipeline to {model_path}")


if __name__ == "__main__":
    main()