# Presentation Script — Credit Card Fraud Detection

**Authors:** Mohamed Attia, Kerim Hariri  
**Date:** May 23–24, 2026 | Duration: ~12 minutes

---

## SLIDE 1 — Title Slide

### What to say:

> "Good afternoon. Our project is titled _Credit Card Fraud Detection Using Ensemble and Linear Classification Methods: A Comparative Study_. The goal was simple on paper but hard in practice: can we build a model that catches fraudsters while not constantly blocking normal customers? We compared three algorithms to find out."

**Transition:** "Let's start with why this problem matters."

---

## SLIDE 2 — Problem & Motivation

### Slide content:

- $33 billion lost to card fraud annually (Nilson Report)
- Only 0.173% of transactions are fraud (1 in every 578)
- Two failure modes: missing fraud = financial loss; false alarms = angry customers
- Traditional rule-based systems fail against evolving fraud patterns

### What to say:

> "Credit card fraud costs the global economy over **$33 billion every year**. Every transaction you make is a target.

> The core challenge is not just detection — it's the **extreme imbalance**. In our dataset, only **492 out of 284,807 transactions** are fraudulent. That's 0.173%. Less than two in every thousand.

> This creates a brutal trap: a model that blindly labels everything as 'legitimate' would be **99.83% accurate**. Sounds great on paper. But it catches **zero fraud**. Accuracy is a completely useless metric here.

> The real challenge has two failure modes:
>
> - **Miss a fraud** → financial loss for the bank and customer
> - **False alarm** → a real customer's card gets blocked at the worst moment — restaurant, airport, emergency

> Our goal: build models that balance both failure modes, using proper metrics, and compare which algorithm handles this best."

**Transition:** "Before we talk methods, let me give you a quick picture of what we actually did."

---

## SLIDE 3 — Project Overview (What We Did)

### Slide content:

- 6-phase project: Setup → EDA → Preprocessing → Training → Evaluation → Paper
- 3 algorithms compared: Logistic Regression, Random Forest, XGBoost
- Full reproducible pipeline: raw data → trained models → results → LaTeX paper

### What to say:

> "This was a full end-to-end data mining project across 6 phases.

> Phase 1: Environment setup, dataset download from Kaggle.  
> Phase 2: Exploratory Data Analysis — understanding what the data looks like.  
> Phase 3: Preprocessing — cleaning, scaling, splitting, handling imbalance.  
> Phase 4: Training three classification models.  
> Phase 5: Evaluating and comparing results across 5 metrics.  
> Phase 6: Writing the final 5-page IEEE-format paper.

> Everything is version-controlled, reproducible, and submitted as a complete package.

> The three methods we compared were:
>
> - **Logistic Regression** — our linear baseline
> - **Random Forest** — ensemble of 200 decision trees
> - **XGBoost** — gradient boosting with 300 sequential trees

> Now let's look at the data."

---

## SLIDE 4 — The Dataset

### Slide content:

- Source: Kaggle — ULB Machine Learning Group
- 284,807 transactions, September 2013, European cardholders
- 30 features: V1–V28 (PCA), Amount, Time → Target: Class (0/1)
- 492 fraud (0.173%) vs. 284,315 legitimate (99.827%)
- No missing values

### What to say:

> "Our dataset comes from the **Université Libre de Bruxelles** and is one of the most-used benchmark datasets for fraud detection research.

> It contains **284,807 credit card transactions** from **two days in September 2013** by European cardholders.

> There are **30 features**:
>
> - **V1 through V28**: These are **PCA-transformed** — the original features were transformed using Principal Component Analysis to protect cardholder privacy. We don't know what they represent in real life, which is a limitation we'll come back to.
> - **Amount**: Transaction value in euros, ranging from 0 to 25,000
> - **Time**: Seconds since the first transaction in the dataset

> The **target variable is Class**: 0 for legitimate, 1 for fraud.

> The dataset has **zero missing values**. But the class distribution is brutal.

> _(show the pie chart / bar chart)_

> Look at this — the blue bar is 284,315 legitimate transactions. The red bar is 492 fraud cases. This is not a small imbalance. This is **1 fraud per 578 transactions**. Any model we build must specifically account for this."

**Transition:** "So the first major decision was: how do we preprocess this fairly?"

---

## SLIDE 5 — Preprocessing Pipeline

### Slide content:

```
Raw Data (284,807)
    ↓ Remove 1,081 duplicates
Cleaned (283,726)
    ↓ Scale Amount + Time (StandardScaler)
Scaled Data
    ↓ Stratified 80/20 split
Train (226,980) | Test (56,746)
    ↓ SMOTE on training set ONLY
SMOTE Train (453,204 balanced) | Test stays real
```

### What to say:

> "The preprocessing pipeline had four critical steps:

> **Step 1 — Duplicate removal:** We found and removed **1,081 duplicate rows**. Keeping duplicates would inflate performance metrics unfairly.

> **Step 2 — Feature scaling:** V1–V28 are already PCA-scaled, so we only scaled **Amount and Time** using StandardScaler — zero mean, unit variance. Importantly, we fit the scaler on training data only and apply it to test data. This prevents **data leakage**.

> **Step 3 — Stratified train/test split:** 80% training, 20% test. **Stratified** means we preserve the fraud rate in both sets. Test set reflects real-world distribution: 95 fraud in 56,746 transactions.

> **Step 4 — SMOTE:** This is where we handle the imbalance. SMOTE — Synthetic Minority Oversampling Technique — generates **synthetic fraud samples** by interpolating between real fraud cases.

> We went from 378 fraud in 226,980 training samples to **226,602 fraud vs. 226,602 legitimate** — a perfectly balanced training set.

> **Critical design decision: SMOTE only on the training set.** Never on test. If you apply SMOTE to test data, your evaluation is dishonest — you're testing on artificial samples, not real-world conditions. We evaluate on 56,746 real transactions with real imbalance."

**Transition:** "Now, the three models."

---

## SLIDE 6 — Methods: Three Algorithms

### Slide content:

Three columns with algorithm name, type, key config

### What to say:

---

### LOGISTIC REGRESSION (baseline)

> "First: **Logistic Regression**. This is our linear baseline — the simplest possible approach.

> The algorithm draws a **straight line** (or hyperplane in higher dimensions) through the feature space to separate fraud from legitimate. The output is a probability estimate for each class.

> Configuration:
>
> - Regularization C = 0.01 — strong regularization to prevent overfitting
> - class_weight = 'balanced' — tells the model to penalize fraud misclassifications more
> - Trained on SMOTE-balanced data

> Why include it? If logistic regression fails, it **justifies** using more complex models. It's the honest baseline."

---

### RANDOM FOREST

> "Second: **Random Forest**. Instead of one decision tree, we train **200 trees simultaneously**.

> Each tree is trained on a **random bootstrap sample** of the data, and at each split, only a **random subset of features** is considered. Final prediction is a majority vote across all 200 trees.

> Configuration:
>
> - 200 trees, max depth 20
> - class_weight = 'balanced' — double imbalance protection alongside SMOTE
> - Trained on SMOTE-balanced data

> Why does this work better? The **randomness** prevents any single tree from memorizing noise. The **ensemble voting** smooths out errors. And **non-linear boundaries** capture complex fraud patterns that a straight line can't."

---

### XGBOOST

> "Third: **XGBoost** — eXtreme Gradient Boosting. This is the opposite philosophy from Random Forest.

> Instead of training 300 trees in parallel, XGBoost trains them **sequentially**. Each new tree focuses on correcting the mistakes of the previous one. It's gradient descent, but on trees.

> Configuration:
>
> - 300 boosting rounds, shallow trees (max depth 6)
> - Learning rate 0.05 — slow learning = better generalization
> - Subsample = 0.8: each tree sees 80% of data, 80% of features
> - SMOTE already balanced training so scale_pos_weight ≈ 1.0

> XGBoost is considered **state-of-the-art for tabular data**. It's what wins most Kaggle competitions."

**Transition:** "Now the moment of truth — what did the numbers say?"

---

## SLIDE 7 — Results Table

### Slide content:

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 97.41%     | 5.39%      | **87.37%** | 0.1015     | 0.9618     |
| **Random Forest**   | **99.94%** | **83.91%** | 76.84%     | **0.8022** | **0.9842** |
| XGBoost             | 99.87%     | 57.78%     | 82.11%     | 0.6783     | 0.9733     |

### What to say:

> "Before I read these numbers, let me define the metrics — because on imbalanced data, you cannot just look at accuracy.

> - **Accuracy**: % of all predictions correct. Misleading — 99.83% just by predicting everything legitimate.
> - **Precision**: Of everything we flagged as fraud, how many were actually fraud? High precision = fewer false alarms.
> - **Recall**: Of all actual fraud cases, how many did we catch? High recall = fewer missed frauds.
> - **F1-Score**: Harmonic mean of precision and recall. Our primary metric — balances both failure modes.
> - **ROC-AUC**: How well the model ranks fraud above legitimate across all decision thresholds. 1.0 is perfect, 0.5 is random.

> Now the results:

> **Logistic Regression**: 87% recall — it catches most fraud. But precision is 5.39%. That means for every real fraud it catches, it generates **18 false alarms**. Look at the confusion matrix: 83 true positives but **1,461 false positives**. 1,461 innocent customers flagged. Unusable in production.

> **XGBoost**: Strong. 82% recall, 57.8% precision. Catches more fraud than RF but generates 57 false alarms. F1 of 0.6783.

> **Random Forest**: **Winner.** 76.8% recall, 83.9% precision, F1 of 0.8022, AUC of 0.9842. Only **14 false positives** in the entire test set of 56,746 transactions. This is the balance we were looking for."

---

## SLIDE 8 — ROC Curves

### Slide content:

_(show roc_curves_combined.png)_

### What to say:

> "The ROC curve shows how well each model separates fraud from legitimate across all possible decision thresholds.

> The X-axis is the **false positive rate** — how often we wrongly flag legitimate transactions.  
> The Y-axis is the **true positive rate** — how often we correctly catch fraud.

> A perfect model hugs the top-left corner. A random model follows the diagonal.

> All three models are significantly above the diagonal — they all learn real patterns.

> But look at the **area under the curve**:
>
> - Random Forest: **0.9842** — nearly perfect
> - XGBoost: 0.9733 — excellent
> - Logistic Regression: 0.9618 — good, but the gap is visible

> The Random Forest curve stays highest across all thresholds. That means regardless of how strict we set our fraud threshold, it consistently outperforms the others."

---

## SLIDE 9 — Confusion Matrices + Feature Importance

### Slide content:

_(show confusion_matrices_combined.png + feature_importance_comparison.png)_

### What to say (confusion matrices):

> "Confusion matrices make the trade-offs concrete.

> _(Point to LR matrix)_: Logistic Regression — 83 fraud caught, but 1,461 innocent customers flagged. Reject for production.

> _(Point to XGB matrix)_: XGBoost — 78 fraud caught, 57 false positives. Better, but we can do better.

> _(Point to RF matrix)_: Random Forest — 73 fraud caught, only **14 false positives**. Yes, we miss 22 frauds that XGBoost catches. But we also avoid flagging 43 innocent people. Depending on business priorities, this trade-off is usually worth it.

> **Key insight**: RF's precision of 83.9% means if your card gets flagged by this system, there's an **84% chance you're actually a fraudster**. That's actionable intelligence."

### What to say (feature importance):

> "Now something really interesting: feature importance.

> Both Random Forest and XGBoost independently agree on the same top features: **V14, V10, V12, V4, V17**.

> XGBoost attributes **44.7% of its decision weight to V14 alone**. Random Forest agrees V14 is dominant.

> Two completely different algorithms — one uses parallel voting, one uses sequential boosting — and they both converge on the same answer. That's **cross-validation through model diversity**. It means V14 is genuinely the strongest signal for fraud in this dataset, not an artifact of one model's quirks.

> We don't know what V14 represents in the real world because PCA anonymized it. But whatever it is — it separates fraudsters from legitimate cardholders better than anything else."

---

## SLIDE 10 — Conclusions & Recommendations

### Slide content:

- Winner: Random Forest (best F1=0.8022, AUC=0.9842, precision=83.91%)
- Use XGBoost when recall matters more (high-value accounts)
- SMOTE + stratified split = honest evaluation methodology
- V14 dominant in both ensemble models
- Limitations: PCA limits interpretability; static split; no streaming
- Future: threshold optimization, LSTM for time sequences, online learning

### What to say:

> "Let's summarize what we learned.

> **The answer**: Random Forest wins. It achieves the best F1-score, best ROC-AUC, and best precision. Only 14 false positives in 56,746 transactions. For a production fraud detection system, this minimizes customer disruption while catching 76.8% of fraud.

> **When to choose XGBoost instead**: If you're protecting high-value corporate accounts where missing fraud costs millions, XGBoost's higher recall (82.1%) might justify the extra false alarms. The right choice depends on business cost function.

> **Methodological lessons**:
>
> - Never apply SMOTE to test data
> - Never use accuracy alone on imbalanced problems
> - Always stratify your train/test split
> - Use F1 and AUC as primary metrics
> - Agreement across model types validates your features

> **Limitations**: The PCA transformation means we can't explain _why_ V14 matters to a bank manager. We also used a static temporal split — in reality, fraud patterns **evolve over time**, so time-aware validation would be more realistic.

> **Future directions**: Threshold optimization based on actual cost functions (how much does one fraud cost vs. one false alarm?), LSTM or autoencoder models for time-series fraud patterns, and online learning for concept drift as fraud tactics evolve.

> Thank you. We're happy to take questions."

---

## APPENDIX — Likely Questions & Answers

**Q: Why not just use a neural network?**

> "Neural networks require much more data to train reliably and are much harder to interpret. Random Forest gives us feature importance and explainability with 98.4% AUC — hard to justify the complexity tradeoff for this problem size."

**Q: Why SMOTE instead of just using class_weight='balanced'?**

> "We actually used both. class_weight='balanced' adjusts the loss function. SMOTE generates new training examples, giving the model more varied exposure to fraud patterns during training. The combination is more robust than either alone."

**Q: Isn't 76.8% recall low? You're missing 23% of fraud.**

> "On a dataset with 1:578 imbalance, 76.8% recall with 83.9% precision is exceptional. The alternative — Logistic Regression's 87.4% recall — comes with 1,461 false positives. That's not a trade-off most banks would accept. In production, threshold tuning can shift the recall/precision balance based on business priorities."

**Q: What does V14 represent?**

> "We don't know — the original features were PCA-anonymized to protect cardholder privacy. This is the core limitation of this dataset. We can say V14 is the strongest discriminative signal, but we can't translate that back into an actionable real-world feature like 'merchant category' or 'transaction time of day'."

**Q: Why did you remove duplicates? Couldn't those be real repeated transactions?**

> "Exact duplicates across all 30 features including Amount and Time would require the same cardholder to make the exact same purchase at the exact same second. That's statistically impossible for real transactions. These are data artifacts — likely row duplication errors in the original collection process."

**Q: How long did training take?**

> "Logistic Regression: seconds. Random Forest (200 trees, 453,000 SMOTE samples): a few minutes. XGBoost (300 rounds): similar. The SMOTE expansion was the bottleneck, not model training."

---

## SPEAKER NOTES — Key Numbers to Memorize

| Fact                | Number                     |
| ------------------- | -------------------------- |
| Total transactions  | 284,807                    |
| Fraud count         | 492 (0.173%)               |
| Duplicates removed  | 1,081                      |
| Train size (SMOTE)  | 453,204 balanced           |
| Test size           | 56,746 (real distribution) |
| RF false positives  | 14                         |
| LR false positives  | 1,461                      |
| RF F1               | 0.8022                     |
| RF AUC              | 0.9842                     |
| XGB recall          | 82.11%                     |
| RF precision        | 83.91%                     |
| V14 XGB importance  | 44.7%                      |
| Annual fraud losses | $33 billion                |

---

## TIMING GUIDE

| Slide     | Topic                         | Time                         |
| --------- | ----------------------------- | ---------------------------- |
| 1         | Title                         | 0:30                         |
| 2         | Problem & motivation          | 1:30                         |
| 3         | Project overview              | 1:00                         |
| 4         | Dataset                       | 1:30                         |
| 5         | Preprocessing                 | 2:00                         |
| 6         | Methods (3 algorithms)        | 2:30                         |
| 7         | Results table                 | 1:30                         |
| 8         | ROC curves                    | 0:45                         |
| 9         | Confusion matrices + features | 1:30                         |
| 10        | Conclusions                   | 1:15                         |
| **Total** |                               | **~14 min** (trim as needed) |

---

Your project starts with a real-world problem: every day, banks process thousands of credit card transactions, and among them, a very small number are fraudulent, but detecting them is extremely difficult because fraud is rare—only about 0.17% of all transactions, meaning roughly 1 out of every 578 is fraud; this creates a major challenge because a model can achieve over 99.8% accuracy simply by predicting everything as legitimate, yet completely fail to detect any fraud, which makes accuracy useless in this context, so the real objective becomes balancing two competing risks: catching fraud (so the bank doesn’t lose money) and avoiding false alarms (so real customers are not blocked and frustrated); to tackle this, you used a dataset of about 284,000 transactions with 492 fraud cases, where most features are anonymized (V1–V28 using PCA), along with transaction amount and time, and since the data is highly imbalanced, you first built a proper pipeline by removing duplicate records to avoid bias, scaling numerical features like amount and time so models can learn effectively, splitting the data into training and testing sets using a stratified approach to preserve the original fraud ratio, and then applying SMOTE only to the training set to artificially balance it by generating synthetic fraud examples while keeping the test set unchanged to ensure realistic evaluation; after preprocessing, you trained three different models with increasing complexity: Logistic Regression as a simple linear baseline that tries to separate fraud and legitimate transactions with a straight boundary, Random Forest which builds many decision trees in parallel and combines their decisions to capture complex patterns and reduce overfitting, and XGBoost which builds trees sequentially where each new tree focuses on correcting the mistakes of the previous ones, making it very powerful for structured data; instead of relying on accuracy, you evaluated the models using precision (how many flagged fraud cases are actually fraud), recall (how many actual fraud cases are detected), F1-score (the balance between precision and recall), and ROC-AUC (overall ability to distinguish fraud from legitimate transactions), and the results showed clear differences: Logistic Regression achieved high recall but extremely low precision, meaning it caught many fraud cases but also produced a huge number of false alarms—over 1,400—which makes it impractical, XGBoost provided a better balance with solid recall and moderate precision but still generated a noticeable number of false positives, while Random Forest delivered the best overall performance by achieving a strong balance between precision and recall, resulting in the highest F1-score and AUC, and most importantly, only 14 false positives out of more than 56,000 transactions, which is a critical advantage in real-world deployment; an additional insight came from feature importance analysis, where both Random Forest and XGBoost independently identified the same feature, V14, as the most influential, which increases confidence that this feature truly captures meaningful fraud-related patterns even though its real-world meaning is unknown due to anonymization; in the end, your conclusion is that Random Forest is the most suitable model for this problem because it provides the best trade-off between detecting fraud and minimizing disruption to legitimate users, and the key takeaway from the entire project is that in highly imbalanced problems like fraud detection, the goal is not to maximize accuracy or even just recall, but to carefully balance detection performance with real-world impact, while also recognizing limitations such as lack of feature interpretability and the need for future improvements like time-aware models and adaptive learning systems.
