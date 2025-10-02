# Exploring KNN and Distance Metrics (Group members: David, Daniela, Emin, Joaquin)
# Part 1: Euclidean KNN on Iris
# Part 2: Manhattan KNN on gird-like synthetic data
# Part 3: Decision boundary visualization (Euclidean vs Manhattan)
# Part 4: Experimenting with k (1, 3, 5, 7, 15) + accuracy table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

# Part 1: Euclidean KNN on Iris
def part1_iris_euclidean(test_size=0.3, k=5):
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Full-feature model for accuracy
    knn_full = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_full.fit(X_train, y_train)
    y_pred = knn_full.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Part 1] Iris (Euclidean, k={k}) Accuracy: {acc:.3f}")

    # 2D boundary using the first two features (as suggested in lab tips)
    X2_train = X_train[:, :2]
    y2_train = y_train
    knn_2d = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_2d.fit(X2_train, y2_train)

    # Mesh over feature space
    h = 0.02
    padding = 0.5
    x_min, x_max = X2_train[:, 0].min() - padding, X2_train[:, 0].max() + padding
    y_min, y_max = X2_train[:, 1].min() - padding, X2_train[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = knn_2d.predict(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(len(np.unique(y2_train)) + 2) - 0.5)
    plt.scatter(X2_train[:, 0], X2_train[:, 1], c=y2_train, edgecolor='k', s=30)
    plt.title(f"Iris Decision Boundary (Euclidean, k={k}) — Train 2D")
    plt.xlabel("Feature 1 (sepal length)")
    plt.ylabel("Feature 2 (sepal width)")
    plt.show()

    return acc

# Part 2: Manhattan Distance on Gird-like Dataset
def make_grid_like_classification(n_samples=300, noise=0.2, round_to=1):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=RANDOM_STATE
    )
    rng = np.random.RandomState(RANDOM_STATE)
    X = X + noise * rng.randn(*X.shape)
    X_grid = np.round(X, round_to)  # simulate grid/axis-aligned structure
    return X_grid, y


def part2_grid_manhattan(test_size=0.3, k=5):
    Xg, yg = make_grid_like_classification()
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        Xg, yg, test_size=test_size, random_state=RANDOM_STATE, stratify=yg
    )

    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(Xg_train, yg_train)
    yg_pred = knn.predict(Xg_test)
    acc = accuracy_score(yg_test, yg_pred)
    print(f"[Part 2] Grid-like (Manhattan, k={k}) Accuracy: {acc:.3f}")

    # Inline boundary plot for the training set
    h = 0.02
    padding = 0.5
    x_min, x_max = Xg_train[:, 0].min() - padding, Xg_train[:, 0].max() + padding
    y_min, y_max = Xg_train[:, 1].min() - padding, Xg_train[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(len(np.unique(yg_train)) + 2) - 0.5)
    plt.scatter(Xg_train[:, 0], Xg_train[:, 1], c=yg_train, edgecolor='k', s=30)
    plt.title(f"Grid-like Decision Boundary (Manhattan, k={k}) — Train")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    return acc

# Part 3: Compare Euclidean vs Manhattan
def part3_compare_on_grid(k=5):
    Xg, yg = make_grid_like_classification()
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        Xg, yg, test_size=0.3, random_state=RANDOM_STATE, stratify=yg
    )

    knn_e = KNeighborsClassifier(n_neighbors=k, metric='euclidean').fit(Xg_train, yg_train)
    knn_m = KNeighborsClassifier(n_neighbors=k, metric='manhattan').fit(Xg_train, yg_train)

    # Create a common mesh for fair visual comparison
    h = 0.02
    padding = 0.5
    x_min, x_max = Xg_train[:,0].min()-padding, Xg_train[:,0].max()+padding
    y_min, y_max = Xg_train[:,1].min()-padding, Xg_train[:,1].max()+padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict and plot side by side
    Z_e = knn_e.predict(grid).reshape(xx.shape)
    Z_m = knn_m.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, Z, title in [
        (axes[0], Z_e, f"Euclidean (k={k})"),
        (axes[1], Z_m, f"Manhattan (k={k})"),
    ]:
        ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(len(np.unique(yg_train))+2)-0.5)
        ax.scatter(Xg_train[:,0], Xg_train[:,1], c=yg_train, edgecolor='k', s=30)
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    fig.suptitle("Grid-like: Decision Boundary Comparison")
    plt.tight_layout()
    plt.show()

    acc_e = accuracy_score(yg_test, knn_e.predict(Xg_test))
    acc_m = accuracy_score(yg_test, knn_m.predict(Xg_test))
    print(f"[Part 3] Test Accuracy — Euclidean: {acc_e:.3f} | Manhattan: {acc_m:.3f}")
    return acc_e, acc_m

# Part 4: Experimenting with k values
def part4_sweep_k(ks=(1,3,5,7,15)):
    iris = load_iris()
    Xi, yi = iris.data, iris.target
    Xi_train, Xi_test, yi_train, yi_test = train_test_split(
        Xi, yi, test_size=0.3, random_state=RANDOM_STATE, stratify=yi
    )

    Xg, yg = make_grid_like_classification()
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        Xg, yg, test_size=0.3, random_state=RANDOM_STATE, stratify=yg
    )

    rows = []
    for k in ks:
        acc_i = accuracy_score(yi_test,
                               KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                               .fit(Xi_train, yi_train).predict(Xi_test))
        acc_g_m = accuracy_score(yg_test,
                                 KNeighborsClassifier(n_neighbors=k, metric='manhattan')
                                 .fit(Xg_train, yg_train).predict(Xg_test))
        acc_g_e = accuracy_score(yg_test,
                                 KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                                 .fit(Xg_train, yg_train).predict(Xg_test))
        rows.append({"k": k,
                     "Iris (Euclidean)": acc_i,
                     "Grid (Manhattan)": acc_g_m,
                     "Grid (Euclidean)": acc_g_e})

    df = pd.DataFrame(rows)
    print("\n[Part 4] Accuracy vs k")
    print(df.to_string(index=False))

    # Optional: CV on Iris
    cv_rows = []
    for k in ks:
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k, metric='euclidean'),
                                 Xi, yi, cv=5)
        cv_rows.append({"k": k, "Iris CV mean acc (5-fold)": scores.mean()})
    df_cv = pd.DataFrame(cv_rows)
    print("\n[Optional] Iris 5-fold CV accuracy by k")
    print(df_cv.to_string(index=False))

    return df, df_cv

# Run everything
if __name__ == "__main__":
    acc_iris = part1_iris_euclidean(k=5)
    acc_grid_manhattan = part2_grid_manhattan(k=5)
    acc_euc, acc_man = part3_compare_on_grid(k=5)
    df_k, df_cv = part4_sweep_k(ks=(1, 3, 5, 7, 15))

    print("\n=== Short Answers Summary ===")
    print("- Iris uses Euclidean distance: continuous features; straight-line similarity is appropriate.")
    print("- Changing k trades bias/variance: small k overfits, large k underfits.")
    print("- Grid-like data suits Manhattan: axis-aligned L1 matches the discretized structure.")
    print("- Euclidean vs Manhattan boundaries differ due to L2 circles vs L1 diamonds (unit balls).")
    print("- k-sweep + (optional) CV show which k performs best for each dataset.")