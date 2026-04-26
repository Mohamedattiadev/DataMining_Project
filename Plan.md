# Data Mining Course Project — Plan

## Overview
Apply data mining techniques on a real-world dataset, compare multiple methods, evaluate
performance with appropriate metrics, and deliver a scientific paper.

**Presentation deadline:** May 23–24 (starting 13:00, attendance list order)  
**Submission:** Final paper (PDF) + Source code + Dataset link + (optional) LaTeX files

---

## Chosen Task: Classification
**Rationale:** Classification offers the richest set of evaluation metrics (Accuracy, Precision,
Recall, F1, ROC-AUC, Confusion Matrix), clear problem framing, and strong comparability
between methods — maximizing both mandatory and bonus score potential.

## Chosen Dataset
**Dataset:** Credit Card Fraud Detection (or similar large, recent, meaningful dataset from Kaggle/UCI)  
- Source: Kaggle — [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Size: ~284,000 transactions, 30 features  
- Recent, real-world, non-trivial, supports classification with class imbalance challenge  

## Methods to Apply (minimum 2, targeting 3 for bonus depth)
1. **Logistic Regression** — baseline linear classifier, interpretable
2. **Random Forest** — ensemble method, handles imbalance well, feature importance
3. **XGBoost** *(bonus depth)* — gradient boosting, typically state-of-the-art on tabular data

---

## Phase Breakdown

### Phase 1 — Project Setup & Environment
- Initialize Git repository with branch structure
- Set up Python environment (conda/venv)
- Install required libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, imbalanced-learn
- Download and verify dataset
- Confirm dataset suitability

### Phase 2 — Exploratory Data Analysis (EDA) & Preprocessing
- Load dataset and inspect structure (shape, dtypes, head)
- Check for missing values and duplicates
- Analyze class distribution (imbalance analysis)
- Statistical summary of features
- Correlation heatmap and feature distributions
- Preprocessing steps:
  - Handle missing values (if any)
  - Feature scaling (StandardScaler for continuous features)
  - Handle class imbalance: SMOTE or class_weight parameter
  - Train/test split (stratified, 80/20)
- Save preprocessed data

### Phase 3 — Method Implementation
For each method (Logistic Regression, Random Forest, XGBoost):
- Train model on training set
- Tune hyperparameters (GridSearchCV or manual tuning)
- Document parameter settings and rationale
- Save trained model

### Phase 4 — Evaluation & Comparison
For each trained model compute:
- Accuracy
- Precision, Recall, F1-score (macro & weighted)
- ROC-AUC score
- Confusion Matrix (with visualization)
- ROC curve plot (all methods on same plot)
- Comparative results table
- Feature importance plots (RF & XGBoost)
- Discuss findings: which method performs best and why

### Phase 5 — Paper Writing (LaTeX — Bonus)
Write academic paper with all required sections:
1. **Title** — descriptive, concise
2. **Abstract** — problem, methods, key results, conclusion (~200 words)
3. **Keywords** — 4–6 keywords
4. **Introduction** — problem motivation, objectives
5. **Related Work** — 5–10 references to similar work
6. **Dataset and Preprocessing** — full description + preprocessing pipeline
7. **Proposed Method / Methodology** — describe all 3 methods, justify selection
8. **Experimental Results and Discussion** — tables, figures, comparative analysis
9. **Conclusion** — summary, limitations, future work
10. **References** — properly cited (IEEE or ACM format)

**Paper format:** LaTeX (for 5% bonus), conference/journal quality (for 10% bonus)

### Phase 6 — Final Submission & Presentation
- Review paper for plagiarism (target Turnitin < 20%)
- Final code cleanup and documentation
- Prepare presentation slides
- Package submission: PDF + source code + dataset link + LaTeX source

---

## Evaluation Criteria Checklist
| Criterion | Weight | Status |
|---|---|---|
| Dataset and problem selection | 10% | |
| Preprocessing and methodology | 15% | |
| Application of multiple methods | 15% | |
| Evaluation metrics usage | 10% | |
| Results and discussion | 10% | |
| Paper format and writing quality | 10% | |
| **Total Mandatory** | **70%** | |
| LaTeX usage | 5% | |
| Turnitin < 20% | 5% | |
| Novel contribution | 10% | |
| Publication-quality writing | 10% | |
| **Total Bonus** | **30%** | |

---

## Branch Structure (Git Checkpoints)
| Branch | Content |
|---|---|
| `main` | Stable, always holds the latest complete state |
| `1-project-setup` | Environment, dependencies, dataset download |
| `2-eda-preprocessing` | EDA notebooks, preprocessing scripts |
| `3-method-implementation` | Model training scripts for all methods |
| `4-evaluation` | Metrics, plots, comparison analysis |
| `5-paper-writing` | LaTeX paper source files |
| `6-final-submission` | Final paper PDF, cleaned code, submission package |

---

## Tools & Libraries
- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **ML:** scikit-learn, xgboost, imbalanced-learn
- **Visualization:** matplotlib, seaborn
- **Paper:** LaTeX (Overleaf or local TeX)
- **Version Control:** Git / GitHub
