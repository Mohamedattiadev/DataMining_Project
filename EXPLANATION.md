# Full Project Explanation — Credit Card Fraud Detection
> Read this before your presentation. Everything you need to explain is here.

---

## THE BIG PICTURE (What is this project about?)

We have a dataset of **284,807 credit card transactions** made by European cardholders in September 2013.
Only **492 of them are fraud** — that is 0.17% of all transactions. The rest (99.83%) are legitimate.

**The goal:** Build a machine learning model that can automatically detect which transactions are fraudulent.

**The challenge:** The data is extremely imbalanced — like finding 1 fraud in every 578 transactions. Normal models just predict "everything is legitimate" and get 99.8% accuracy without catching any fraud. That's useless. So we need special techniques.

**What we did:** We applied 3 different machine learning methods, compared them fairly, and wrote a scientific paper about it.

---

## THE DATA — What is in the dataset?

- **284,807 rows** — each row = one credit card transaction
- **31 columns:**
  - `Time` — seconds since the first transaction in the dataset
  - `Amount` — how much money was in the transaction (in euros)
  - `V1` to `V28` — 28 mysterious numbers. These are the original features (merchant, location, card type, etc.) transformed using PCA (a math technique) to hide private customer information. We can't know what they originally meant.
  - `Class` — **0 = legitimate, 1 = fraud** (this is what we want to predict)

**Source:** Kaggle — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Provided by the Machine Learning Group at Université Libre de Bruxelles (ULB).

---

## FILE BY FILE EXPLANATION

### `scripts/download_dataset.py`
**What it does:** Downloads the dataset automatically from Kaggle using the Kaggle API.
You run it once and it saves `creditcard.csv` into `data/raw/`.
It also verifies the download: checks the shape, missing values, and fraud rate.

---

### `scripts/02_preprocessing.py`
**What it does:** Cleans and prepares the raw data for machine learning. It has 5 steps:

1. **Load the raw CSV file**
2. **Remove duplicates** — found and removed 1,081 duplicate rows (same transaction recorded twice). Left with 283,726 samples.
3. **Feature Scaling** — `Amount` ranges from €0 to €25,000 and `Time` ranges from 0 to 172,792 seconds. These huge numbers would confuse the models. We apply `StandardScaler` which converts them to have mean=0 and standard deviation=1. The V1-V28 features are already scaled (they came out of PCA that way).
4. **Train/Test Split** — We split 80% for training (226,980 rows) and 20% for testing (56,746 rows). We use *stratified* splitting, meaning both halves keep the same 0.17% fraud rate. This is important so we test on realistic data.
5. **SMOTE** (Synthetic Minority Over-sampling Technique) — This is the key technique for handling imbalance. SMOTE creates *synthetic* (artificial but realistic) fraud examples by interpolating between real fraud cases. It brings the training set from 378 fraud vs 226,602 legitimate → to 226,602 fraud vs 226,602 legitimate (balanced!). **CRITICAL: SMOTE is applied ONLY on the training set, never on the test set.** If we applied it on the test set too, we'd be cheating — testing on fake data.

**Output files saved to `data/processed/`:**
- `X_train.csv` / `y_train.csv` — original unbalanced training features and labels
- `X_test.csv` / `y_test.csv` — test features and labels (real distribution, 95 fraud cases)
- `X_train_smote.csv` / `y_train_smote.csv` — SMOTE-balanced training set (453,204 rows)

---

### `notebooks/01_eda.ipynb` — Exploratory Data Analysis
**What it does:** A Jupyter notebook that explores the dataset visually and statistically before any modeling.
Think of it as "getting to know the data."

It produces these figures:

#### `results/figures/class_distribution.png`
**What it shows:** Two charts — a bar chart and a pie chart — showing how many legitimate vs fraud transactions exist.
**Key message:** The pie chart shows fraud is just 0.173% — the blue slice (legitimate) takes up almost the entire chart. The tiny red sliver is fraud. This visually proves why class imbalance is a problem.
**How made:** matplotlib bar chart + pie chart with the fraud/legitimate counts.

#### `results/figures/amount_time_distributions.png`
**What it shows:** 4 histograms — the distribution of `Amount` and `Time` for legitimate and fraud transactions separately.
**Key message:**
- Fraud transactions tend to have lower amounts (fraudsters often test with small amounts first)
- Legitimate transactions spike at two times of day (day shift and night shift patterns visible in Time)
**How made:** matplotlib histograms split by Class=0 and Class=1.

#### `results/figures/pca_feature_distributions.png`
**What it shows:** 28 small histograms (one per V1-V28 feature) showing how each feature is distributed differently for fraud vs legitimate transactions.
**Key message:** Features like V14, V12, V17 show very different distributions between fraud and legitimate — meaning they are useful for detection. Features like V22, V23 overlap heavily — less useful.
**How made:** matplotlib subplots 7×4, overlaying fraud (red) and legitimate (blue) distributions.

#### `results/figures/correlation_heatmap.png`
**What it shows:** A triangular heatmap of correlations between all features. Red = positive correlation, Blue = negative correlation.
**Key message:** Most V features are uncorrelated with each other (expected, because PCA produces independent components). The bottom row shows which features correlate most with Class (fraud label).
**How made:** seaborn heatmap using the Pearson correlation matrix of the full dataset.

#### `results/figures/top_features_boxplot.png`
**What it shows:** 10 boxplots — one per top feature — showing the value range for Class=0 (legitimate) vs Class=1 (fraud).
**Key message:** For features like V14 and V17, the boxes for fraud and legitimate don't overlap much — clear separation. This confirms these features are powerful fraud signals.
**How made:** pandas boxplot grouped by Class, for the 10 features with highest absolute correlation to Class.

---

### `scripts/03a_logistic_regression.py`
**What it does:** Trains a Logistic Regression model.

**What is Logistic Regression?**
It's the simplest classification model. It draws a straight line (or flat plane) to separate fraud from legitimate. It outputs a probability: "this transaction is 87% likely to be fraud."

**Our settings:**
- `C=0.01` — strong regularization (prevents the model from memorizing the training data)
- `class_weight='balanced'` — tells the model to pay more attention to fraud cases even though there are fewer of them
- `solver='lbfgs'` — the optimization algorithm used to find the best line
- Trained on SMOTE-balanced data

**Why we included it:** It's a baseline. If Logistic Regression already does well, we don't need complex models. If it does poorly, it shows us we need non-linear methods.

**Result:** Accuracy=97.4%, but Precision=5.4% — it flags almost everything as fraud (low precision). Not practical alone, but proves linear separability exists.

---

### `scripts/03b_random_forest.py`
**What it does:** Trains a Random Forest model.

**What is Random Forest?**
Imagine asking 200 different experts to each look at a transaction and vote: fraud or not? Each expert (decision tree) was trained on a slightly different random subset of the data and considers random subsets of features. The final answer is the majority vote of all 200 trees.

**Our settings:**
- `n_estimators=200` — 200 trees
- `max_depth=20` — each tree can be up to 20 levels deep
- `min_samples_split=5` — a node needs at least 5 samples to split further
- `class_weight='balanced'` — double protection against imbalance (SMOTE + balanced weights)
- Trained on SMOTE-balanced data (453,204 samples)

**Why we included it:** Ensemble methods are state-of-the-art for tabular data. RF handles non-linear patterns, is robust to outliers, and provides feature importance.

**Result:** Accuracy=99.9%, F1=0.8022, AUC=0.9842 — **best overall model.**

---

### `scripts/03c_xgboost.py`
**What it does:** Trains an XGBoost model.

**What is XGBoost?**
XGBoost (eXtreme Gradient Boosting) also builds many trees, but differently from Random Forest. Instead of building all trees independently and voting, it builds trees *sequentially*. Each new tree focuses on fixing the mistakes of the previous ones. It's like learning from errors step by step.

**Our settings:**
- `n_estimators=300` — 300 boosting rounds
- `max_depth=6` — shallow trees (each tree is simple, but 300 of them together are powerful)
- `learning_rate=0.05` — slow learning rate (more trees, smaller steps = better generalization)
- `subsample=0.8` — each tree sees 80% of the data randomly
- `colsample_bytree=0.8` — each tree uses 80% of features randomly
- Trained on SMOTE-balanced data

**Why we included it:** XGBoost consistently wins machine learning competitions on tabular data. It's the industry standard for structured data problems.

**Result:** Accuracy=99.9%, Recall=0.8211, AUC=0.9733 — strong but slightly below RF overall.

---

### `notebooks/04_evaluation.ipynb` — Evaluation & Comparison
**What it does:** Loads all 3 trained models and evaluates them on the test set. Produces all comparison plots.

#### `results/figures/roc_curves_combined.png`
**What it shows:** ROC (Receiver Operating Characteristic) curve for all 3 models on one chart. Each curve shows the trade-off between True Positive Rate (catching fraud) and False Positive Rate (falsely flagging legitimate) at different thresholds. The diagonal line = a random classifier (useless). The higher and more to the left the curve, the better.
**Key message:** Random Forest (green) has the highest curve. AUC=0.9842 means RF correctly ranks a random fraud case above a random legitimate case 98.4% of the time.
**How made:** sklearn `roc_curve()` for each model, plotted together with matplotlib.

#### `results/figures/confusion_matrices_combined.png`
**What it shows:** 3 confusion matrices side by side. Each matrix is a 2×2 table:
- **Top-left (TN):** Correctly identified as legitimate
- **Top-right (FP):** Legitimate flagged as fraud (false alarm)
- **Bottom-left (FN):** Fraud missed (dangerous!)
- **Bottom-right (TP):** Correctly caught fraud
**Key message:**
- LR: TP=83, FP=1,461 — catches most fraud but drowns you in false alarms
- RF: TP=73, FP=14 — catches 73 fraud cases with only 14 false alarms (best!)
- XGB: TP=78, FP=57 — catches more fraud but more false alarms than RF
**How made:** sklearn `confusion_matrix()`, plotted as seaborn heatmaps.

#### `results/figures/metrics_comparison_bar.png`
**What it shows:** Grouped bar chart comparing Precision, Recall, F1-Score, and ROC-AUC for all 3 models side by side.
**Key message:** Quick visual summary — RF (green bars) dominates in Precision, F1, and AUC. LR (blue) has high recall but terrible precision.
**How made:** matplotlib grouped bar chart with annotated values on top of each bar.

#### `results/figures/feature_importance_comparison.png`
**What it shows:** Two horizontal bar charts — one for Random Forest, one for XGBoost — showing which features matter most.
**Key message:** Both models independently agree: **V14 is by far the most important feature** (44.7% importance in XGBoost alone). V10, V12, V4, V17 also consistently appear in both top lists. This agreement between two different model types validates that these PCA components are genuine fraud signals, not noise.
**How made:** `model.feature_importances_` from sklearn/XGBoost, plotted as horizontal bars.

#### `results/metrics_summary.csv`
**What it is:** A simple CSV table with all 5 metrics for all 3 models. Used in the paper's results table.

---

### `paper/main.tex` + `paper/main.pdf`
**What it is:** The full 5-page academic paper written in LaTeX using the IEEE conference template.

**Why LaTeX?** LaTeX is the standard for scientific papers. It produces professional PDF output with proper equations, figure numbering, citations, and formatting. It's what real conferences and journals require. (Worth +5% bonus.)

**Paper sections:**
1. **Title & Abstract** — What the paper is about, methods used, key results, in ~200 words
2. **Introduction** — Why fraud detection matters, what we contribute, paper structure
3. **Related Work** — What other researchers have done before us (10 cited papers)
4. **Dataset & Preprocessing** — Full description of data + our 4-step preprocessing pipeline
5. **Methodology** — Mathematical description of LR, RF, XGBoost with equations and parameter justifications
6. **Results & Discussion** — All metrics, all figures, and deep analysis of what the numbers mean
7. **Conclusion** — Summary, limitations, future work directions
8. **References** — 10 papers cited in IEEE format

---

## THE 3 METRICS THAT MATTER MOST (and how to explain them)

**Why not just use Accuracy?**
If a model predicts "everything is legitimate," it gets 99.83% accuracy. But it catches ZERO fraud. Accuracy is useless here.

**Precision** = Of all transactions the model flagged as fraud, what % were actually fraud?
- High precision = few false alarms = customers don't get their cards blocked unnecessarily
- RF precision = 83.9% → 84 out of every 100 flagged transactions are actually fraud

**Recall** = Of all actual fraud transactions, what % did the model catch?
- High recall = fewer missed frauds = less financial loss
- RF recall = 76.8% → catches 77 out of every 100 actual fraud cases

**F1-Score** = The balance between Precision and Recall (harmonic mean)
- Best single metric for imbalanced problems
- RF F1 = 0.8022 → best balance overall

**ROC-AUC** = How well the model distinguishes fraud from legitimate at ANY threshold
- 1.0 = perfect, 0.5 = random guessing
- RF AUC = 0.9842 → near-perfect discrimination

---

## WHY RANDOM FOREST WON

1. **200 trees voting** catches patterns that a single model misses
2. **Non-linear** — fraud patterns are complex, not a straight line
3. **Built-in randomness** prevents overfitting
4. **Balanced class weights + SMOTE** = double protection against imbalance
5. **Consistent feature importance** — agrees with XGBoost on key features (V14, V10, V12)

---

## THE SMOTE EXPLANATION (most important to understand)

**Problem:** We have 226,602 legitimate training samples but only 378 fraud samples. Models trained on this will be heavily biased toward predicting "legitimate."

**What SMOTE does:** Takes each fraud sample, finds its nearest fraud neighbors in feature space, and creates new *synthetic* fraud samples by interpolating between them. Think of it like drawing new points on a line between existing fraud points.

**Result:** Training set goes from 378 fraud → 226,602 fraud. Now perfectly balanced.

**Key point to emphasize:** We apply SMOTE ONLY on the training data. The test set stays with the original real distribution (95 fraud out of 56,746). This way we train on balanced data but test on realistic data. If we applied SMOTE to test data too, we'd be testing on fake samples — the results would be meaningless.

---

## WHAT TO SAY WHEN PRESENTING (key phrases)

- *"We chose classification because it directly maps to the business problem: is this transaction fraud or not?"*
- *"The dataset has a 1:578 fraud ratio — standard models fail here, which is why we need SMOTE."*
- *"We applied SMOTE only on training data to prevent data leakage — this is critical for honest evaluation."*
- *"We use F1-Score and ROC-AUC as primary metrics because accuracy is misleading on imbalanced data."*
- *"Both ensemble models independently agree on V14 as the top feature — this cross-validation gives us confidence."*
- *"Random Forest wins overall: best precision means fewer false alarms — fewer customers with frozen cards."*
- *"Logistic Regression has the highest recall but near-zero precision — it flags everything. Useful as a baseline, not in production."*
- *"The paper is written in LaTeX using the IEEE conference template — same format used in real research publications."*

---

## NUMBERS TO MEMORIZE

| Fact | Number |
|---|---|
| Total transactions | 284,807 |
| Fraud transactions | 492 (0.17%) |
| Fraud ratio | 1 in every 578 |
| Duplicates removed | 1,081 |
| Train size (after SMOTE) | 453,204 |
| Test size | 56,746 (95 fraud) |
| Best model | Random Forest |
| Best F1 | 0.8022 (RF) |
| Best AUC | 0.9842 (RF) |
| Most important feature | V14 |
| Number of methods | 3 (LR, RF, XGBoost) |
| Number of metrics used | 5 (Acc, Prec, Rec, F1, AUC) |
| Paper pages | 5 pages, IEEE format |
| References cited | 10 papers |
