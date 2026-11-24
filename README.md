# Fraud Detection Assignment - Graded Assignment 1

Binary classification of credit card transactions as fraud vs non-fraud.

## Repository structure

```
fraud-detection-assignment/
├── data/
│   ├── raw/              # raw dataset (do not commit)
│   └── processed/        # cleaned train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py           # load, split, leakage checks
│   ├── features.py       # feature engineering
│   ├── model.py          # training, PR-AUC, F1 macro
│   ├── inference.py      # REST / batch serving interface
│   ├── evaluate.py       # benchmarking script (100+ requests)
│   └── drift.py          # PSI / KS drift monitoring proposal
├── api/
│   ├── main.py           # FastAPI app
│   └── requirements.txt  # pinned API requirements
├── tests/
│   └── test_*.py
├── reports/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place your dataset at `data/raw/credit_card_fraud_10k.csv`.  
Target column: `is_fraud` (0 = legitimate, 1 = fraud).

## API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

## Benchmarking

```bash
python src/evaluate.py
```

Reports p50, p95, p99 latency for 100+ requests through the inference API.

## License

Academic use only - BITS Pilani WILP Graded Assignment 1.