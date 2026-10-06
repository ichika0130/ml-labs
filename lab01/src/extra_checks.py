"""Supplementary checks (beyond the spec) used to support the conclusions.

1. Is the accuracy gap between the two baselines real? 10-fold stratified CV on all data.
2. Why is LogisticRegression slow to train? Compare unscaled vs standardized features.
The graded baselines in baseline.py / measure.py stay exactly as specified.
"""
import csv
import statistics
import time
import warnings

from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from common import RESULTS_DIR, SEED, load_split, make_models, set_seeds

TRAIN_REPEATS = 5

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def median_fit_ms(factory, X, y):
    factory().fit(X, y)  # warm-up
    times = []
    for _ in range(TRAIN_REPEATS):
        model = factory()
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def cross_validation():
    X, y = load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    rows = []
    for name, factory in make_models().items():
        scores = cross_val_score(factory(), X, y, cv=cv)
        rows.append({"model": name, "cv_folds": 10,
                     "cv_accuracy_mean": f"{scores.mean():.4f}",
                     "cv_accuracy_std": f"{scores.std():.4f}"})
    return rows


def scaling_effect():
    X_train, X_test, y_train, y_test = load_split()
    scaler = StandardScaler().fit(X_train)
    variants = {
        "unscaled (spec)": (X_train, X_test),
        "standardized": (scaler.transform(X_train), scaler.transform(X_test)),
    }
    factory = lambda: LogisticRegression(max_iter=1000, random_state=SEED)
    rows = []
    for label, (Xtr, Xte) in variants.items():
        model = factory().fit(Xtr, y_train)
        rows.append({"features": label,
                     "lbfgs_iterations": int(model.n_iter_[0]),
                     "converged": bool(model.n_iter_[0] < 1000),
                     "train_time_median_ms": round(median_fit_ms(factory, Xtr, y_train), 3),
                     "test_accuracy": f"{model.score(Xte, y_test):.4f}"})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seeds()
    cv_rows = cross_validation()
    set_seeds()
    scale_rows = scaling_effect()
    write_csv(RESULTS_DIR / "cv_accuracy.csv", cv_rows)
    write_csv(RESULTS_DIR / "logreg_scaling.csv", scale_rows)
    for r in cv_rows + scale_rows:
        print(r)


if __name__ == "__main__":
    main()
