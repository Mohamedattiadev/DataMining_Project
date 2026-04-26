"""
Method 3: XGBoost — Credit Card Fraud Detection
Rationale: gradient boosting, typically best on tabular data, handles imbalance
           via scale_pos_weight, provides feature importance.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

RANDOM_STATE = 42
PROC_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    # XGBoost trained on SMOTE data for fair comparison
    X_train = pd.read_csv(os.path.join(PROC_DIR, 'X_train_smote.csv'))
    y_train = pd.read_csv(os.path.join(PROC_DIR, 'y_train_smote.csv')).squeeze()
    X_test  = pd.read_csv(os.path.join(PROC_DIR, 'X_test.csv'))
    y_test  = pd.read_csv(os.path.join(PROC_DIR, 'y_test.csv')).squeeze()
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    # scale_pos_weight for additional imbalance awareness
    neg   = int((y_train == 0).sum())
    pos   = int((y_train == 1).sum())
    spw   = neg / pos if pos > 0 else 1
    print(f"\nTraining XGBoost...")
    print(f"  scale_pos_weight={spw:.2f}, n_estimators=300, max_depth=6, lr=0.05")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("  Training complete.")
    return model


def plot_feature_importance(model, feature_names, model_name='XGBoost'):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importances.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    top20.sort_values().plot(kind='barh', ax=ax, color='tomato', edgecolor='black')
    ax.set_title(f'Top 20 Feature Importances — {model_name}', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    fname = model_name.lower()
    plt.savefig(os.path.join(FIG_DIR, f'feature_importance_{fname}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nTop 10 features:\n{top20.head(10).to_string()}")


def evaluate_model(model, X_test, y_test, model_name='XGBoost'):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f" RESULTS — {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    fname = model_name.lower()
    plt.savefig(os.path.join(FIG_DIR, f'cm_{fname}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    np.save(os.path.join(MODEL_DIR, f'roc_{fname}_fpr.npy'), fpr)
    np.save(os.path.join(MODEL_DIR, f'roc_{fname}_tpr.npy'), tpr)

    return {'model': model_name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'roc_auc': auc}


def main():
    X_train, y_train, X_test, y_test = load_data()
    model   = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, X_test.columns.tolist())
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgboost.pkl'))
    print(f"\nModel saved to models/xgboost.pkl")
    return metrics


if __name__ == '__main__':
    main()
