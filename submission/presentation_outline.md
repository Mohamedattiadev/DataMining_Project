# Presentation Outline — Credit Card Fraud Detection
**Date:** May 23–24, 2026 at 13:00  
**Duration:** ~10–12 minutes

---

## Slide 1 — Title
- Title: Credit Card Fraud Detection Using Ensemble and Linear Classification Methods
- Your name
- Data Mining Course, 2025–2026

## Slide 2 — Problem & Motivation (1 min)
- Global card fraud losses > $33 billion/year
- Challenge: 0.17% fraud rate → extreme class imbalance (1:578)
- Goal: detect fraud automatically with high recall AND precision

## Slide 3 — Dataset (1 min)
- 284,807 transactions, 30 features (28 PCA + Amount + Time)
- 492 fraud vs 284,315 legitimate
- Show: class_distribution.png

## Slide 4 — Preprocessing Pipeline (1.5 min)
- Remove 1,081 duplicates
- StandardScaler on Amount & Time
- Stratified 80/20 split
- SMOTE on training set only (226K → 453K samples)
- Why SMOTE only on train? → prevent data leakage

## Slide 5 — Methods (2 min)
- Logistic Regression: linear baseline, class_weight=balanced
- Random Forest: 200 trees, ensemble, non-linear
- XGBoost: gradient boosting, 300 estimators, lr=0.05
- All trained on same SMOTE-balanced set, evaluated on same test set

## Slide 6 — Results Table (1.5 min)
| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| LR | 0.054 | 0.874 | 0.102 | 0.962 |
| RF | **0.839** | 0.768 | **0.802** | **0.984** |
| XGB | 0.578 | 0.821 | 0.678 | 0.973 |

## Slide 7 — ROC Curves (1 min)
- Show: roc_curves_combined.png
- RF best discrimination at all thresholds

## Slide 8 — Confusion Matrices (1 min)
- Show: confusion_matrices_combined.png
- LR: many false positives (impractical)
- RF: 73 TP, only 14 FP — best balance
- XGBoost: 78 TP, 57 FP

## Slide 9 — Feature Importance (1 min)
- Show: feature_importance_comparison.png
- V14 most important in both RF & XGBoost
- Agreement validates the findings

## Slide 10 — Conclusion (1 min)
- Random Forest: best overall (F1=0.802, AUC=0.984)
- XGBoost: use when recall is priority
- LR: interpretable baseline only
- Future: threshold tuning, deep learning, streaming detection

---

## Key Points to Emphasize
1. SMOTE applied to training only (no data leakage)
2. Why accuracy alone is misleading for imbalanced data
3. The precision-recall trade-off in fraud systems
4. Both ensemble models agree on top features → robustness
