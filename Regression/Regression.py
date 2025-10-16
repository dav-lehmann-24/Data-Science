# Regression Lab (Group members: Atai, Benni, David, Hafsa)
# Part 1: Data loading and preview
# Part 2: Exploratory Data Analysis (EDA)
# Part 3: Linear Regression
# Part 4: Logistic Regression (price > median)
# Part 5: Reflection

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

RANDOM_STATE = 42

# Data loading & cleaning
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().copy()
    if "price" not in df.columns:
        raise ValueError("Expected a 'price' column in the dataset.")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove invalid prices
    df = df[df["price"] > 0].copy()
    # Trim extreme outliers on target to prevent metric blow-ups
    lo, hi = df["price"].quantile([0.005, 0.995])
    df = df[(df["price"] >= lo) & (df["price"] <= hi)].copy()
    return df

# EDA
def eda(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("--------------------------------------------------------------------- \n")

    # Price distribution (linear)
    plt.figure()
    plt.hist(df["price"], bins=50, edgecolor='black', linewidth=1.5, color='skyblue')
    plt.title("Distribution of House Prices")
    plt.xlabel("Price"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.show()

    # Simple scatter diagnostics for up to 3 useful predictors
    for col in pick_plot_features(df, k=3):
        plt.figure()
        plt.scatter(df[col], df["price"], alpha=0.5)
        plt.xlabel(col); plt.ylabel("price")
        plt.title(f"{col} vs. price")
        plt.tight_layout(); plt.show()

def pick_plot_features(df: pd.DataFrame, k: int = 3) -> list[str]:
    preferred = [c for c in ["sqft_living", "bedrooms", "bathrooms"] if c in df.columns]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    chosen = list(preferred)
    for col in numeric_cols:
        if col != "price" and col not in chosen:
            chosen.append(col)
        if len(chosen) >= k:
            break
    return chosen[:k]

# Manual preprocessing (no sklearn preprocessors)
def preprocess_fit(X: pd.DataFrame) -> dict:
    X = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Impute stats
    num_medians = X[num_cols].median() if num_cols else pd.Series(dtype=float)
    cat_modes = X[cat_cols].mode().iloc[0] if cat_cols else pd.Series(dtype=object)

    # Apply imputations
    if num_cols:
        X[num_cols] = X[num_cols].fillna(num_medians)
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna(cat_modes)

    # Standardize numeric
    num_means = X[num_cols].mean() if num_cols else pd.Series(dtype=float)
    num_stds = X[num_cols].std(ddof=0).replace(0, 1) if num_cols else pd.Series(dtype=float)
    if num_cols:
        X[num_cols] = (X[num_cols] - num_means) / num_stds

    # One-hot
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Save preprocessing state
    state = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_medians": num_medians,
        "cat_modes": cat_modes,
        "num_means": num_means,
        "num_stds": num_stds,
        "columns_after_ohe": X_enc.columns.tolist(),
    }
    return {"X": X_enc, "state": state}

def preprocess_transform(X: pd.DataFrame, state: dict) -> pd.DataFrame:
    X = X.copy()
    num_cols = state["num_cols"]; cat_cols = state["cat_cols"]
    # Impute
    if num_cols:
        X[num_cols] = X[num_cols].fillna(state["num_medians"])
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna(state["cat_modes"])
    # Standardize numeric
    if num_cols:
        X[num_cols] = (X[num_cols] - state["num_means"]) / state["num_stds"]
    # One-hot, then align columns to training
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    # Reindex to match training columns (fill missing with 0)
    X_enc = X_enc.reindex(columns=state["columns_after_ohe"], fill_value=0)
    return X_enc

# Linear Regression
def run_linear_regression(df: pd.DataFrame) -> None:
    y = df["price"].copy()
    X_raw = df.drop(columns=["price"]).copy()

    # Train/test split on raw
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Fit preprocessing on train, transform train/test
    prep = preprocess_fit(X_train_raw)
    X_train = prep["X"]
    X_test = preprocess_transform(X_test_raw, prep["state"])

    # Train model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[Linear Regression] Test RMSE: {rmse:,.0f} $")
    print(f"[Linear Regression] Test MAE:  {mae:,.0f} $")
    print(f"[Linear Regression] Test R^2:  {r2:.3f}")

    # True vs. Predicted plot
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True price"); plt.ylabel("Predicted price")
    plt.title("Linear Regression: True vs Predicted")
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi])
    plt.tight_layout(); plt.show()

# Logistic Regression (price > median)
def run_logistic_regression(df: pd.DataFrame) -> None:
    y_cont = df["price"].copy()
    median_price = float(y_cont.median())
    y_bin = (y_cont > median_price).astype(int)
    X_raw = df.drop(columns=["price"]).copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_bin, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bin
    )

    prep = preprocess_fit(X_train_raw)
    X_train = prep["X"]
    X_test = preprocess_transform(X_test_raw, prep["state"])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"[Logistic Regression] Test Accuracy: {acc:.4f}")
    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"[Logistic Regression] ROC AUC: {auc:.4f}")
    except Exception:
        pass
    print(f"Positive class: price > median ({median_price:,.0f}); threshold=0.5")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Logistic Regression: Confusion Matrix")
    plt.tight_layout(); plt.show()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Logistic Regression: ROC Curve")
    plt.tight_layout(); plt.show()

# Reflection
def reflection() -> None:
    print("""
Reflection (Linear vs. Logistic Regression)
- Linear regression estimates a continuous target (price); MSE/RMSE/MAE/R^2 quantify fit quality.
- Logistic regression estimates the probability that price is above the median; we evaluate with accuracy and ROC/AUC.
- Preprocessing (median/mode imputation, standardizing numerics, one-hot encoding categoricals) is essential for stable models.
- Outlier trimming and removing invalid prices prevents metrics from being dominated by a few extreme points.
- Adding informative features (e.g., sqft_living, grade, view, lat/long) typically lowers error and improves classification.
""".strip())

# Orchestrator
def main(csv_path: str = "KC_housing_data.csv") -> None:
    df = load_data(csv_path)
    df = clean_data(df)
    eda(df)
    run_linear_regression(df)
    run_logistic_regression(df)
    reflection()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "KC_housing_data.csv"
    main(path)