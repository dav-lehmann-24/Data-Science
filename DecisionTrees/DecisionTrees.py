# Decision Trees Lab (Group members: Atai, David, Ephraim, Niki, Petra)
# Part 1: Data loading and preview
# Part 2: Exploratory Data Analysis (EDA)
# Part 3: Linear Regression
# Part 4: Logistic Regression (price > median)
# Part 5: Decision Tree Regression
# Part 6: Reflection

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
from sklearn.tree import DecisionTreeRegressor, plot_tree

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
    print("\n--------------------------------------------------------------------- \n")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Logistic Regression: Confusion Matrix")
    plt.tight_layout(); plt.show()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Logistic Regression: ROC Curve")
    plt.tight_layout(); plt.show()

# Decision Tree Regression (KC Housing)
def kc_build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds feature matrix
    Preference for strong predictors if present; otherwise use all non-price columns.
    """
    strong_candidates = [
        "sqft_living", "sqft_living15", "sqft_above",
        "bathrooms", "bedrooms",
        "grade", "view", "waterfront", "condition",
        "lat", "long", "floors",
        "yr_built", "yr_renovated", "zipcode",
    ]
    existing = [c for c in strong_candidates if c in df.columns]
    X_raw = df[existing].copy() if existing else df.drop(columns=["price"]).copy()
    y = df["price"].astype(float).copy()
    return X_raw, y

def run_decision_tree_regression(df: pd.DataFrame) -> None:
    """
    Satisfies new tasks:
    - Baseline DecisionTreeRegressor
    - Manual hyperparameter search (depth / min_samples_*), no extra libs
    - Evaluation on a held-out test set
    - Feature importance and compact tree plot
    """
    # 1) Build features/target
    X_raw, y = kc_build_features(df)

    # 2) Split: 20% test; keep a validation split for tuning
    X_train_full, X_test_raw, y_train_full, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=RANDOM_STATE
    )

    # 3) Preprocess (reusing your manual helpers)
    prep = preprocess_fit(X_train_raw)
    X_train = prep["X"]
    X_val = preprocess_transform(X_val_raw, prep["state"])
    X_test = preprocess_transform(X_test_raw, prep["state"])

    # 4) Baseline tree
    dt0 = DecisionTreeRegressor(random_state=RANDOM_STATE)
    dt0.fit(X_train, y_train)
    _ = _evaluate_dt(dt0, X_val, y_val, tag="Baseline (val)")

    # 5) Manual tuning (small grid)
    best = {"rmse": np.inf, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}
    best_model = None
    for max_depth in [None, 4, 6, 8, 10, 12, 15]:
        for min_split in [2, 5, 10, 20, 50]:
            for min_leaf in [1, 2, 5, 10]:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
                if rmse < best["rmse"]:
                    best = {
                        "rmse": rmse,
                        "max_depth": max_depth,
                        "min_samples_split": min_split,
                        "min_samples_leaf": min_leaf,
                    }
                    best_model = model
    print(f"[Decision Tree • Tuning] Best params: "
          f"max_depth={best['max_depth']}, min_samples_split={best['min_samples_split']}, "
          f"min_samples_leaf={best['min_samples_leaf']}, RMSE={best['rmse']:,.0f} (val)")

    # 6) Retrain on full train (train_full) with best params and evaluate on test
    X_train_full_enc = preprocess_transform(X_train_full, prep["state"])
    dt_best = DecisionTreeRegressor(
        max_depth=best["max_depth"],
        min_samples_split=best["min_samples_split"],
        min_samples_leaf=best["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )
    dt_best.fit(X_train_full_enc, y_train_full)
    _ = _evaluate_dt(dt_best, X_test, y_test, tag="Best (test)")

    # 7) Feature importance
    fi = pd.Series(dt_best.feature_importances_, index=X_train_full_enc.columns).sort_values(ascending=False)
    print("\n[Decision Tree] Top feature importance:")
    print(fi.head(15))
    print("\n--------------------------------------------------------------------- \n")

    # 8) Compact tree plot (top levels)
    plt.figure(figsize=(28, 10), dpi=150)
    plot_tree(
        dt_best,
        feature_names=X_train_full_enc.columns.tolist(),
        filled=True,
        impurity=False,
        max_depth=3,
        fontsize=10,
        precision=0
    )
    plt.title("Decision Tree (top levels) — King County")
    plt.tight_layout(); plt.show()

def _evaluate_dt(model: DecisionTreeRegressor, X: pd.DataFrame, y: pd.Series, tag: str = "") -> dict:
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    prefix = f"[Decision Tree {tag}]".strip()
    print(f"{prefix} RMSE: {rmse:,.0f} $")
    print(f"{prefix} MAE:  {mae:,.0f} $")
    print(f"{prefix} R^2:  {r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

# Reflection
def reflection() -> None:
    print("""
Reflection (Linear, Logistic & Decision Tree Regression)
- Linear regression fits a global linear relationship; errors and R^2 reflect how much variance is explained by linear terms.
- Logistic regression (price > median) cleanly separates cheaper vs. expensive homes; accuracy and ROC/AUC quantify ranking and thresholding quality.
- Decision trees capture non-linearities and interactions automatically, but can overfit; limiting depth / min_samples_* is crucial.
- Manual hyperparameter tuning (depth, min_samples_split/leaf) balances bias vs variance; feature importance helps interpret drivers (often lat/long, sqft_living, grade, view).
- Data hygiene (removing invalid prices, trimming extreme tails) is essential so metrics aren't dominated by outliers.
""".strip())

# Orchestrator
def main(csv_path: str = "KC_housing_data.csv") -> None:
    df = load_data(csv_path)
    df = clean_data(df)
    eda(df)
    run_linear_regression(df)
    run_logistic_regression(df)
    run_decision_tree_regression(df)
    reflection()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "KC_housing_data.csv"
    main(path)