"""
Preprocessing Pipeline — Credit Card Fraud Detection
Steps:
  1. Load raw data
  2. Drop duplicates
  3. Scale Amount and Time (StandardScaler)
  4. Train/test split (stratified 80/20)
  5. Apply SMOTE on training set only (avoid data leakage)
  6. Save splits to data/processed/
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
TEST_SIZE    = 0.20
RAW_PATH     = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'creditcard.csv')
PROC_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_data():
    print("[1/5] Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"      Shape: {df.shape}")
    return df


def clean_data(df):
    print("[2/5] Cleaning data...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"      Dropped {before - after} duplicate rows. Remaining: {after:,}")
    return df


def scale_features(df):
    print("[3/5] Scaling 'Amount' and 'Time'...")
    scaler = StandardScaler()
    df = df.copy()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])
    # Save scaler for reproducibility in paper
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("      Scaler saved to models/scaler.pkl")
    return df, scaler


def split_data(df):
    print("[4/5] Splitting into train/test (80/20 stratified)...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")
    print(f"      Train fraud: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    print(f"      Test  fraud: {y_test.sum()}  ({y_test.mean()*100:.4f}%)")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    print("[5/5] Applying SMOTE to training set only...")
    print(f"      Before SMOTE — Class 0: {(y_train==0).sum():,}  Class 1: {(y_train==1).sum()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"      After  SMOTE — Class 0: {(y_resampled==0).sum():,}  Class 1: {(y_resampled==1).sum():,}")
    return X_resampled, y_resampled


def save_splits(X_train, X_test, y_train, y_test, X_train_sm, y_train_sm):
    os.makedirs(PROC_DIR, exist_ok=True)
    # Original splits (for evaluation against real distribution)
    pd.DataFrame(X_train).to_csv(os.path.join(PROC_DIR, 'X_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(PROC_DIR,  'X_test.csv'),  index=False)
    pd.Series(y_train, name='Class').to_csv(os.path.join(PROC_DIR, 'y_train.csv'), index=False)
    pd.Series(y_test,  name='Class').to_csv(os.path.join(PROC_DIR, 'y_test.csv'),  index=False)
    # SMOTE-resampled training set
    pd.DataFrame(X_train_sm).to_csv(os.path.join(PROC_DIR, 'X_train_smote.csv'), index=False)
    pd.Series(y_train_sm, name='Class').to_csv(os.path.join(PROC_DIR, 'y_train_smote.csv'), index=False)
    print(f"\nAll splits saved to {os.path.abspath(PROC_DIR)}/")
    print("  X_train.csv, y_train.csv")
    print("  X_test.csv,  y_test.csv")
    print("  X_train_smote.csv, y_train_smote.csv")


def main():
    print("=" * 50)
    print(" PREPROCESSING PIPELINE")
    print("=" * 50)
    df            = load_data()
    df            = clean_data(df)
    df, scaler    = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_sm, y_train_sm           = apply_smote(X_train, y_train)
    save_splits(X_train, X_test, y_train, y_test, X_train_sm, y_train_sm)
    print("\nPreprocessing complete.")
    print("=" * 50)


if __name__ == '__main__':
    main()
