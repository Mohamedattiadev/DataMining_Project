"""
Dataset download script.
Dataset: Credit Card Fraud Detection
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os
import zipfile
import sys

DATA_RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
DATASET_FILE = os.path.join(DATA_RAW, 'creditcard.csv')


def check_kaggle_api():
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False


def download_via_kaggle():
    import kaggle
    print("Downloading via Kaggle API...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'mlg-ulb/creditcardfraud',
        path=DATA_RAW,
        unzip=True
    )
    print(f"Dataset saved to {DATA_RAW}")


def verify_dataset():
    import pandas as pd
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: {DATASET_FILE} not found.")
        return False
    df = pd.read_csv(DATASET_FILE)
    print(f"\nDataset verified:")
    print(f"  Shape       : {df.shape}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    fraud = df['Class'].value_counts()
    print(f"  Class dist  : {fraud.to_dict()}")
    pct = fraud[1] / len(df) * 100
    print(f"  Fraud rate  : {pct:.4f}%")
    return True


if __name__ == '__main__':
    os.makedirs(DATA_RAW, exist_ok=True)

    if os.path.exists(DATASET_FILE):
        print("Dataset already present. Verifying...")
        verify_dataset()
        sys.exit(0)

    if check_kaggle_api():
        download_via_kaggle()
        verify_dataset()
    else:
        print("Kaggle API not installed.")
        print("Option 1: pip install kaggle  (then place kaggle.json in ~/.kaggle/)")
        print("Option 2: Download manually from:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"  and place creditcard.csv in: {os.path.abspath(DATA_RAW)}")
