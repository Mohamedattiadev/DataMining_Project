# Data Mining Project — Task Tracker

> Mark tasks as done by changing `[ ]` to `[x]`

---

## Phase 1 — Project Setup & Environment
**Git Branch:** `1-project-setup`

- [x] Initialize Git repository (`git init`)
- [x] Create branch structure (branches 1–6)
- [x] Create `.gitignore` (ignore data files, venv, __pycache__, etc.)
- [x] Set up Python virtual environment (`venv` or `conda`)
- [x] Install required libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, imbalanced-learn
- [x] Create `requirements.txt`
- [x] Download dataset (Credit Card Fraud Detection from Kaggle)
- [x] Verify dataset integrity (row count, columns, file size)
- [x] Document dataset source URL in README or paper draft
- [x] Commit checkpoint → merge to `main`

---

## Phase 2 — EDA & Preprocessing
**Git Branch:** `2-eda-preprocessing`

- [x] Create Jupyter notebook `01_eda.ipynb`
- [x] Load dataset and inspect (shape, dtypes, `.head()`, `.info()`)
- [x] Check for missing values and duplicates
- [x] Analyze class distribution (fraud vs. non-fraud counts & percentages)
- [x] Generate statistical summary (`.describe()`)
- [x] Plot feature distributions (histograms)
- [x] Plot correlation heatmap
- [x] Identify and document class imbalance severity
- [x] Create preprocessing script `02_preprocessing.py`
- [x] Apply feature scaling (StandardScaler on `Amount`, `Time`)
- [x] Handle class imbalance (SMOTE)
- [x] Perform stratified train/test split (80/20)
- [x] Save preprocessed splits (`X_train`, `X_test`, `y_train`, `y_test`)
- [x] Document all preprocessing decisions with justifications
- [x] Commit checkpoint → merge to `main`

---

## Phase 3 — Method Implementation
**Git Branch:** `3-method-implementation`

### Logistic Regression
- [x] Create script `03a_logistic_regression.py`
- [x] Train Logistic Regression with `class_weight='balanced'`
- [x] Tune hyperparameter C (regularization strength)
- [x] Document parameter settings and rationale
- [x] Save model

### Random Forest
- [x] Create script `03b_random_forest.py`
- [x] Train Random Forest classifier
- [x] Tune: `n_estimators`, `max_depth`, `class_weight`
- [x] Document parameter settings and rationale
- [x] Save model

### XGBoost (Bonus depth)
- [x] Create script `03c_xgboost.py`
- [x] Train XGBoost classifier
- [x] Tune: `scale_pos_weight`, `n_estimators`, `learning_rate`, `max_depth`
- [x] Document parameter settings and rationale
- [x] Save model

- [x] Commit checkpoint → merge to `main`

---

## Phase 4 — Evaluation & Comparison
**Git Branch:** `4-evaluation`

- [x] Create evaluation notebook `04_evaluation.ipynb`
- [x] Compute for ALL models:
  - [x] Accuracy
  - [x] Precision (macro + weighted)
  - [x] Recall (macro + weighted)
  - [x] F1-score (macro + weighted)
  - [x] ROC-AUC score
  - [x] Confusion Matrix
- [x] Plot Confusion Matrix heatmaps (one per model)
- [x] Plot ROC curves (all models on single plot)
- [x] Plot Feature Importance (Random Forest & XGBoost)
- [x] Build comparative results table (all metrics, all models side by side)
- [x] Write discussion: which model is best, why, what does it mean
- [x] Justify choice of evaluation metrics in writing
- [x] Commit checkpoint → merge to `main`

---

## Phase 5 — Paper Writing (LaTeX)
**Git Branch:** `5-paper-writing`

- [x] Set up LaTeX project (IEEE conference template)
- [x] Write **Title**
- [x] Write **Abstract** (~200 words: problem, methods, results, conclusion)
- [x] Write **Keywords** (7 terms)
- [x] Write **Introduction** (motivation, problem statement, objectives, paper structure)
- [x] Write **Related Work** (10 relevant papers with citations)
- [x] Write **Dataset and Preprocessing** section (full pipeline description)
- [x] Write **Methodology** section (describe all 3 methods with equations, justify selection)
- [x] Write **Experimental Results and Discussion** (all tables and figures)
- [x] Write **Conclusion** (summary, limitations, future work)
- [x] Compile **References** (IEEE format, 10 citations)
- [x] Insert all figures/plots (confusion matrices, ROC curves, feature importance)
- [x] Insert results comparison table
- [x] Compile to PDF — 5 pages, 509KB
- [x] Commit checkpoint → merge to `main`

---

## Phase 6 — Final Submission & Presentation
**Git Branch:** `6-final-submission`

- [x] Final code review and cleanup
- [x] Remove temporary files (install_tex.sh)
- [x] Verify all requirements met
- [x] Export final paper as PDF
- [x] Create submission/ package folder
- [x] Update Tasks.md — all done
- [ ] Prepare presentation slides
- [ ] **PRESENT: May 23–24 at 13:00**

---

## Progress Overview
| Phase | Status |
|---|---|
| 1 - Project Setup | ✅ Complete |
| 2 - EDA & Preprocessing | ✅ Complete |
| 3 - Method Implementation | ✅ Complete |
| 4 - Evaluation | ✅ Complete |
| 5 - Paper Writing | ✅ Complete |
| 6 - Final Submission | 🔄 In Progress (slides remaining) |
