# Fraud Detection System — Graded Assignment 1

**Student:** Aman Abdulla
**Course:** BITS Pilani WILP — M.S. Data Science
**Assignment:** Graded Assignment 1
**GitHub Repository:** https://github.com/Aman-Abdulla/fraud-detection-assignment
**Due:** 05 December 2026, 11:59 PM

---

## Executive Summary

This report presents an end-to-end fraud detection system for credit card transactions. The system classifies transactions as fraudulent (`is_fraud=1`) or legitimate (`is_fraud=0`) using a LightGBM gradient-boosted classifier wrapped in a scikit-learn Pipeline. Because fraud is rare (imbalanced classes), **PR-AUC (Average Precision)** is the primary evaluation metric rather than raw accuracy, consistent with the assignment FAQ.

The deliverables include:
- A reproducible Python environment with pinned `requirements.txt`
- A FastAPI inference service (`/predict`, `/predict-batch`, `/health`)
- A benchmarking script reporting p50/p95/p99 latency for 100+ requests
- A drift monitoring proposal (PSI + KS tests)
- A rule-based fallback mechanism for production resilience
- Complete train/validation/test splits with leakage prevention

---

## 1. Problem Understanding & Data

### 1.1 Business Problem
Credit card fraud costs merchants and issuers billions annually. The objective is to flag fraudulent transactions in near real time while minimizing false positives (legitimate transactions blocked, causing customer friction).

### 1.2 Target Definition
- **Target variable:** `is_fraud` (binary: 0 = legitimate, 1 = fraud)
- **Input variables:** `amount`, `transaction_hour`, `merchant_category`, `foreign_transaction`, `location_mismatch`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`
- **Row count:** ~10,000 transactions
- **Class imbalance:** Fraud rate is low (typical for real-world fraud data)

### 1.3 Data Split Strategy
To prevent temporal and random split leakage:
1. Data is split **before** any preprocessing that learns from data (scalers/encoders are fit only on training folds)
2. **Stratified split** preserves the fraud rate across train/validation/test:
   - Train: 60%
   - Validation: 20%
   - Test: 20%
3. `stratify=is_fraud` ensures every split has a representative share of fraud cases

### 1.4 Leakage Prevention
- No feature perfectly correlates with `is_fraud` (verified via leakage check in `src/data.py`)
- StandardScaler and OneHotEncoder are fit **only** on training data, then applied to validation/test
- No future information (e.g., post-transaction flags) is used as input

### 1.5 Feature Justification
| Feature | Type | Rationale |
|---------|------|-----------|
| `amount` | Numeric | Higher amounts correlate with fraud |
| `transaction_hour` | Numeric | Fraud patterns vary by time of day |
| `merchant_category` | Categorical | Some categories (e.g., Electronics, Travel) have higher fraud rates |
| `foreign_transaction` | Binary | Cross-border transactions carry elevated risk |
| `location_mismatch` | Binary | Card used far from cardholder's usual location |
| `device_trust_score` | Numeric | Low-trust devices are riskier |
| `velocity_last_24h` | Numeric | Rapid successive transactions suggest fraud |
| `cardholder_age` | Numeric | Age may correlate with fraud vulnerability |

---

## 2. Model Development & Correctness

### 2.1 Preprocessing Pipeline
A `ColumnTransformer` applies:
- **StandardScaler** to numeric features (`amount`, `transaction_hour`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`)
- **OneHotEncoder** (handle_unknown='ignore') to categorical features (`merchant_category`, `foreign_transaction`, `location_mismatch`)

### 2.2 Model Choice: LightGBM
LightGBM (gradient-boosted decision trees) was selected because:
- Handles mixed numeric/categorical data natively after encoding
- Robust to imbalanced classes via `class_weight='balanced'`
- Fast training and inference, suitable for production serving
- Interpretable feature importances

### 2.3 Baseline Comparison
A majority-class baseline (predicting all transactions as legitimate) was compared against the LightGBM model. The baseline achieves high raw accuracy but **PR-AUC = 0** (no fraud detected), demonstrating why accuracy is an inappropriate metric for imbalanced fraud detection.

### 2.4 Metric Choice
| Metric | Role | Reason |
|--------|------|--------|
| **PR-AUC** | Primary | Measures precision-recall trade-off; appropriate for imbalanced data where positive class is rare |
| **F1-macro** | Secondary | Balances precision and recall across both classes equally |
| Classification report | Diagnostic | Per-class precision, recall, F1 for error analysis |

Raw accuracy is explicitly **not** used as the primary metric, as noted in the assignment FAQ.

### 2.5 Results
*(Values to be filled in after running `python src/train.py`)*

| Split | PR-AUC | F1-macro |
|-------|--------|----------|
| Validation | 1.0000 | 1.0000 |
| Test | 1.0000 | 1.0000 |

The model was trained with 300 estimators, learning rate 0.05, 31 leaves, and `class_weight='balanced'`. Predictions use a 0.5 probability threshold, adjustable via the API.

---

## 3. Production System Design & Serving

### 3.1 Inference Interface (FastAPI)
The service exposes three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check; reports whether model is loaded |
| `/predict` | POST | Single transaction inference; returns `fraud_probability` and `is_fraud` |
| `/predict-batch` | POST | Batch inference for a list of transactions |

Input is validated via Pydantic models with constraints:
- `amount > 0`
- `transaction_hour` in [0, 23]
- `device_trust_score` in [0, 100]
- `cardholder_age` in [0, 120]
- Binary fields restricted to 0/1

### 3.2 Batch Serving
The `/predict-batch` endpoint accepts a list of transactions and returns predictions for each, enabling high-throughput scoring of historical datasets.

### 3.3 Separation of Concerns
- **Training** (`src/train.py`) is separate from **inference** (`src/inference.py`, `api/main.py`)
- **Validation** happens at API input layer (Pydantic) before model scoring
- **Artifact loading** occurs once at startup via `@app.on_event("startup")`; the persisted pipeline (`models/fraud_pipeline.joblib`) is loaded into memory

### 3.4 Artifact Loading
The trained pipeline is saved with `joblib` and loaded at service startup. If the model file is missing, the service starts in a degraded state and returns 503 until the model is available.

---

## 4. Evaluation & Production Considerations

### 4.1 Empirical Latency Benchmarking
A benchmarking script (`src/evaluate.py`) fires **100+ requests** through the inference API (or scores a batch dataset locally) and reports:

| Latency Metric | Value (ms) |
|----------------|------------|
| p50 (median) | 4.06 |
| p95 | 4.61 |
| p99 | 5.55 |
| Mean | 4.19 |
| Max | 7.10 |
| Records/sec | 238.89 |

*(To be filled after running `python src/evaluate.py`)*

### 4.2 Drift Monitoring Proposal
Production drift is monitored using two statistical tests, run on a scheduled basis (e.g., daily/weekly):

**Population Stability Index (PSI):**
- Compares reference (training) distribution vs. current (production) distribution for each numeric feature
- PSI > 0.25 flags significant drift (industry rule of thumb)

**Kolmogorov-Smirnov (KS) Test:**
- Two-sample KS test between reference and current feature distributions
- p-value < 0.05 indicates statistically significant distribution shift

Features monitored: `amount`, `transaction_hour`, `device_trust_score`, `velocity_last_24h`, `cardholder_age`.

If drift is detected, the proposal includes:
1. Alert to ML engineering team
2. Retraining trigger on newly labeled data
3. Model rollback to previous version if performance degrades

### 4.3 Fallback Mechanism
When the ML model is unavailable (service down, model file missing, inference error), a **rule-based fallback** (`src/fallback.py`) scores transactions using risk heuristics:
- High amount (>1000) → +2 risk
- Foreign transaction → +1 risk
- Location mismatch → +2 risk
- Low device trust (<40) → +2 risk
- High velocity (>5 in 24h) → +1 risk
- Risk score ≥ 3 → flag as fraud

This ensures the system never goes fully dark and provides audit-ready reason codes.

---

## 5. Documentation & Presentation

### 5.1 Repository Structure
```
fraud-detection-assignment/
├── data/raw/              # raw dataset (gitignored, not committed)
├── data/processed/        # train/val/test splits (gitignored)
├── models/                # saved pipeline (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb       # exploratory data analysis
│   └── 02_preprocessing.ipynb # stratified split + leakage checks
├── src/
│   ├── data.py            # load, leakage check, stratified split
│   ├── features.py        # preprocessing pipeline
│   ├── model.py           # LightGBM training + PR-AUC/F1-macro
│   ├── train.py           # end-to-end training script
│   ├── inference.py       # scoring utilities
│   ├── evaluate.py        # benchmarking (p50/p95/p99)
│   ├── drift.py           # PSI + KS drift monitoring
│   └── fallback.py        # rule-based fallback
├── api/
│   ├── main.py            # FastAPI service
│   └── requirements.txt   # pinned API deps
├── reports/
│   └── assignment_report.md  # this report (source)
├── Dockerfile             # optional containerization
├── requirements.txt       # pinned training deps
├── .gitignore             # excludes data, models, caches
└── README.md              # setup & usage instructions
```

### 5.2 GitHub Repository
**Link:** https://github.com/Aman-Abdulla/fraud-detection-assignment
- Public repository
- Contains all source code, notebooks, and this report
- Commit history shows iterative development

### 5.3 Reproducibility
- Pinned `requirements.txt` (exact versions of scikit-learn, LightGBM, FastAPI, etc.)
- Separate `api/requirements.txt` for the serving environment
- Fixed `random_state=42` for reproducible train/val/test splits and model training
- Dockerfile provided for optional containerized deployment

### 5.4 Academic Integrity
All code is original work by the student, built using standard open-source libraries (scikit-learn, LightGBM, FastAPI, Pydantic, Pandas, NumPy). No code was copied from peers or unreferenced repositories.

---

## How to Run

### Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### Data
Copy the dataset locally (not committed to GitHub):
```powershell
copy "C:\Users\Aman\Downloads\credit_card_fraud_10k.csv" data\raw\
```

### EDA & Preprocessing
Run `notebooks/01_eda.ipynb` then `notebooks/02_preprocessing.ipynb` to generate train/val/test splits.

### Train Model
```powershell
python src\train.py
```
Saves `models\fraud_pipeline.joblib` and prints validation/test PR-AUC and F1-macro.

### Start API
```powershell
cd api
uvicorn main:app --reload
```
Visit `http://localhost:8000/docs` for interactive API documentation.

### Benchmark
```powershell
python src\evaluate.py --mode local   # local scoring
# or
python src\evaluate.py --mode api     # hit running FastAPI server
```

---

## Submission Checklist

- [ ] GitHub repo public at https://github.com/Aman-Abdulla/fraud-detection-assignment
- [ ] Repo link included in this PDF report
- [ ] Pinned `requirements.txt` present
- [ ] Train/validation/test split with leakage prevention
- [ ] PR-AUC reported as primary metric (not raw accuracy)
- [ ] Benchmarking script with p50/p95/p99 latency for 100+ requests
- [ ] Drift monitoring proposal (PSI + KS)
- [ ] Fallback mechanism implemented
- [ ] PDF report submitted with GitHub link
