"""Task 2: train both baselines on a 70/30 stratified split and record test accuracy."""
import csv

from sklearn.metrics import accuracy_score

from common import RESULTS_DIR, load_split, make_models, set_seeds


def main() -> None:
    set_seeds()
    X_train, X_test, y_train, y_test = load_split()
    rows = []
    for name, factory in make_models().items():
        model = factory().fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})
        print(f"{name}: {acc:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "baseline_accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
