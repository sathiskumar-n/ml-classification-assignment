"""Common data loading, preprocessing and evaluation functions
used by all 5 model files in this folder."""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_FILE = "bank-full.csv"
RANDOM_SEED = 42

NUMERIC_COLS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]


def load_data():
    """Download bank-full.csv from UCI (only if not already present) and load it."""
    if not os.path.exists(DATA_FILE):
        outer_zip = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(DATA_URL).read()))
        inner_zip = zipfile.ZipFile(io.BytesIO(outer_zip.read("bank.zip")))
        inner_zip.extract(DATA_FILE)
    return pd.read_csv(DATA_FILE, sep=";")


def load_and_preprocess():
    """Return preprocessed train/test data (preprocessing fitted on training data only)."""
    df = load_data()

    # Missing values: 'unknown' in job/education -> NaN (filled later by the imputer)
    df["job"] = df["job"].replace("unknown", np.nan)
    df["education"] = df["education"].replace("unknown", np.nan)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Drop 'duration' (data leakage) and separate features (X) and target (y)
    df = df.drop(columns=["duration"])
    X = df.drop(columns=["y"])
    y = df["y"].map({"yes": 1, "no": 0})

    # 80:20 stratified split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    # Numbers -> scale; text categories -> fill missing + one-hot encode
    numeric_steps = Pipeline([("scale", StandardScaler())])
    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_steps, NUMERIC_COLS),
        ("cat", categorical_steps, CATEGORICAL_COLS)
    ])
    X_train_prep = preprocessor.fit_transform(X_train)   # fit on TRAIN only
    X_test_prep = preprocessor.transform(X_test)         # apply to TEST
    return X_train_prep, X_test_prep, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    """Calculate and print the 6 evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    print(f"\n{name} - evaluation on test data")
    for metric, value in metrics.items():
        print(f"  {metric:<10}: {value:.4f}")
    return metrics
