# Data Mining Project — Task Tracker

> Mark tasks as done by changing `[ ]` to `[x]`

---

## Phase 1 — Project Setup & Environment
**Git Branch:** `1-project-setup`

- [ ] Initialize Git repository (`git init`)
- [ ] Create branch structure (branches 1–6)
- [ ] Create `.gitignore` (ignore data files, venv, __pycache__, etc.)
- [ ] Set up Python virtual environment (`venv` or `conda`)
- [ ] Install required libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, imbalanced-learn
- [ ] Create `requirements.txt`
- [ ] Download dataset (Credit Card Fraud Detection from Kaggle)
- [ ] Verify dataset integrity (row count, columns, file size)
- [ ] Document dataset source URL in README or paper draft
- [ ] Commit checkpoint → merge to `main`

---

## Phase 2 — EDA & Preprocessing
**Git Branch:** `2-eda-preprocessing`

- [ ] Create Jupyter notebook `01_eda.ipynb`
- [ ] Load dataset and inspect (shape, dtypes, `.head()`, `.info()`)
- [ ] Check for missing values and duplicates
- [ ] Analyze class distribution (fraud vs. non-fraud counts & percentages)
- [ ] Generate statistical summary (`.describe()`)
- [ ] Plot feature distributions (histograms)
- [ ] Plot correlation heatmap
- [ ] Identify and document class imbalance severity
- [ ] Create preprocessing script `02_preprocessing.py` (or notebook)
- [ ] Apply feature scaling (StandardScaler on `Amount`, `Time`)
- [ ] Handle class imbalance (SMOTE or class_weight)
- [ ] Perform stratified train/test split (80/20)
- [ ] Save preprocessed splits (`X_train`, `X_test`, `y_train`, `y_test`)
- [ ] Document all preprocessing decisions with justifications
- [ ] Commit checkpoint → merge to `main`

---

## Phase 3 — Method Implementation
**Git Branch:** `3-method-implementation`

### Logistic Regression
- [ ] Create script `03a_logistic_regression.py`
- [ ] Train Logistic Regression with `class_weight='balanced'`
- [ ] Tune hyperparameter C (regularization strength)
- [ ] Document parameter settings and rationale
- [ ] Save model

### Random Forest
- [ ] Create script `03b_random_forest.py`
- [ ] Train Random Forest classifier
- [ ] Tune: `n_estimators`, `max_depth`, `class_weight`
- [ ] Document parameter settings and rationale
- [ ] Save model

### XGBoost (Bonus depth)
- [ ] Create script `03c_xgboost.py`
- [ ] Train XGBoost classifier
- [ ] Tune: `scale_pos_weight`, `n_estimators`, `learning_rate`, `max_depth`
- [ ] Document parameter settings and rationale
- [ ] Save model

- [ ] Commit checkpoint → merge to `main`

---

## Phase 4 — Evaluation & Comparison
**Git Branch:** `4-evaluation`

- [ ] Create evaluation notebook `04_evaluation.ipynb`
- [ ] Compute for ALL models:
  - [ ] Accuracy
  - [ ] Precision (macro + weighted)
  - [ ] Recall (macro + weighted)
  - [ ] F1-score (macro + weighted)
  - [ ] ROC-AUC score
  - [ ] Confusion Matrix
- [ ] Plot Confusion Matrix heatmaps (one per model)
- [ ] Plot ROC curves (all models on single plot)
- [ ] Plot Feature Importance (Random Forest & XGBoost)
- [ ] Build comparative results table (all metrics, all models side by side)
- [ ] Write discussion: which model is best, why, what does it mean
- [ ] Justify choice of evaluation metrics in writing
- [ ] Commit checkpoint → merge to `main`

---

## Phase 5 — Paper Writing (LaTeX)
**Git Branch:** `5-paper-writing`

- [ ] Set up LaTeX project (Overleaf or local — IEEE or ACM template)
- [ ] Write **Title**
- [ ] Write **Abstract** (~200 words: problem, methods, results, conclusion)
- [ ] Write **Keywords** (4–6 terms)
- [ ] Write **Introduction** (motivation, problem statement, objectives, paper structure)
- [ ] Write **Related Work** (review 5–10 relevant papers with citations)
- [ ] Write **Dataset and Preprocessing** section (full pipeline description)
- [ ] Write **Methodology** section (describe all 3 methods, justify selection)
- [ ] Write **Experimental Results and Discussion** (include all tables and figures)
- [ ] Write **Conclusion** (summary, limitations, future work)
- [ ] Compile **References** (IEEE/ACM format, proper citations throughout)
- [ ] Insert all figures/plots (confusion matrices, ROC curves, feature importance)
- [ ] Insert results comparison table
- [ ] Proofread for academic writing quality
- [ ] Check that all tools/libraries are specified in the paper
- [ ] Compile to PDF — check formatting
- [ ] Commit checkpoint → merge to `main`

---

## Phase 6 — Final Submission & Presentation
**Git Branch:** `6-final-submission`

- [ ] Final code review and cleanup (comments, structure)
- [ ] Verify all requirements are met (checklist against Plan.md)
- [ ] Check paper for plagiarism (target Turnitin < 20%)
- [ ] Export final paper as PDF
- [ ] Prepare presentation slides (key results, methods, visuals)
- [ ] Practice presentation (target 10–15 min)
- [ ] Package submission folder:
  - [ ] `paper.pdf`
  - [ ] `source_code/` (all scripts/notebooks)
  - [ ] `dataset_link.txt`
  - [ ] `latex_source/` (optional)
- [ ] Final commit → merge to `main`
- [ ] **PRESENT: May 23–24 at 13:00**

---

## Progress Overview
| Phase | Status |
|---|---|
| 1 - Project Setup | Not started |
| 2 - EDA & Preprocessing | Not started |
| 3 - Method Implementation | Not started |
| 4 - Evaluation | Not started |
| 5 - Paper Writing | Not started |
| 6 - Final Submission | Not started |
