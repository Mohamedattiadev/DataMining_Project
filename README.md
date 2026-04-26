# Credit Card Fraud Detection — Data Mining Course Project

## Overview
Comparative study of three supervised classification methods for credit card fraud detection on a severely imbalanced real-world dataset.

**Task:** Binary Classification  
**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud)  
**Presentation:** May 23–24, 2026 at 13:00

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9741 | 0.0539 | **0.8737** | 0.1015 | 0.9618 |
| Random Forest | **0.9994** | **0.8391** | 0.7684 | **0.8022** | **0.9842** |
| XGBoost | 0.9987 | 0.5778 | 0.8211 | 0.6783 | 0.9733 |

**Winner: Random Forest** (best F1 + ROC-AUC)

---

## Project Structure

```
DataMining_Project/
├── data/
│   ├── raw/            # creditcard.csv (download via script)
│   └── processed/      # train/test splits + SMOTE-balanced sets
├── notebooks/
│   ├── 01_eda.ipynb    # Exploratory Data Analysis
│   └── 04_evaluation.ipynb  # Evaluation & comparison
├── scripts/
│   ├── download_dataset.py      # Kaggle API download
│   ├── 02_preprocessing.py      # Full preprocessing pipeline
│   ├── 03a_logistic_regression.py
│   ├── 03b_random_forest.py
│   └── 03c_xgboost.py
├── models/             # Saved trained models (.pkl)
├── results/
│   ├── figures/        # All plots (ROC, confusion matrices, etc.)
│   └── metrics_summary.csv
├── paper/
│   ├── main.tex        # LaTeX source (IEEE format)
│   └── main.pdf        # Compiled paper (5 pages)
├── submission/         # Final submission package
├── Plan.md             # Full project plan
├── Tasks.md            # Task tracker
└── requirements.txt
```

---

## Setup & Run

```bash
python -m venv venv
source venv/bin/activate   # or: venv/Scripts/activate on Windows
pip install -r requirements.txt

# Download dataset (requires ~/.kaggle/kaggle.json)
python scripts/download_dataset.py

# Run preprocessing
python scripts/02_preprocessing.py

# Train models
python scripts/03a_logistic_regression.py
python scripts/03b_random_forest.py
python scripts/03c_xgboost.py

# Run evaluation notebook
jupyter notebook notebooks/04_evaluation.ipynb
```

---

## Tools & Libraries
Python 3.13 · pandas · numpy · scikit-learn · XGBoost · imbalanced-learn · matplotlib · seaborn · LaTeX (IEEEtran)

## Branch Checkpoints
| Branch | Content |
|---|---|
| `0-plan_phase` | Plan & task tracker |
| `1-project-setup` | Environment, dataset |
| `2-eda-preprocessing` | EDA, preprocessing pipeline |
| `3-method-implementation` | All 3 trained models |
| `4-evaluation` | Metrics, plots, comparison |
| `5-paper-writing` | LaTeX paper |
| `6-final-submission` | Final package |
