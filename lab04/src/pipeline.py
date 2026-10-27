"""Lab 4: the whole data pipeline in one command.

ingest -> validate -> split (70/30 stratified, before any cleaning) -> drop exact
duplicates from train only -> fit preprocessor on train -> transform train and test ->
baseline with and without derived features -> label audit -> hash processed data
and store it with the config.
"""
import platform
import sys
from importlib.metadata import version

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from audit import audit
from common import (CONFIG_PATH, DATA_DIR, RESULTS_DIR, categorical_columns, load_config,
                    numeric_columns, set_seeds, sha256_file, write_csv)
from ingest import ingest
from preprocess import AdultPreprocessor
from validate import validate

LOG_LINES = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


def split(raw, cfg):
    target = cfg["dataset"]["target_column"]
    X, y = raw.drop(columns=target), raw[target].astype("object")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], random_state=cfg["seed"],
        stratify=y if cfg["split"]["stratify"] else None)
    log(f"[split] train {len(X_tr)} / test {len(X_te)} (stratified, seed {cfg['seed']}); "
        f"positive share train {(y_tr == cfg['dataset']['positive_label']).mean():.4f}, "
        f"test {(y_te == cfg['dataset']['positive_label']).mean():.4f}")
    return X_tr, X_te, y_tr, y_te


def drop_train_duplicates(X_tr, y_tr, cfg):
    if not cfg["cleaning"]["drop_train_duplicates"]:
        return X_tr, y_tr, 0
    both = X_tr.assign(_y=y_tr.values)
    keep = ~both.astype("object").duplicated()
    dropped = int((~keep).sum())
    log(f"[clean] dropped {dropped} exact duplicate rows from train; test untouched")
    return X_tr[keep.values], y_tr[keep.values], dropped


def make_preprocessor(cfg, add_features):
    return AdultPreprocessor(
        numeric=numeric_columns(cfg), categorical=categorical_columns(cfg),
        missing_token=cfg["schema"]["missing_token"],
        clip_percentiles=tuple(cfg["cleaning"]["clip_percentiles"]),
        add_features=add_features, age_offset=cfg["features"]["age_offset"])


def baseline(X_tr, X_te, y_tr, y_te, cfg, add_features):
    pre = make_preprocessor(cfg, add_features).fit(X_tr)
    tr, te = pre.transform(X_tr), pre.transform(X_te)
    num = [c for c in tr.columns if c not in categorical_columns(cfg)]
    model = Pipeline([
        ("encode", ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns(cfg)),
        ])),
        ("clf", LogisticRegression(max_iter=cfg["baseline"]["max_iter"], random_state=cfg["seed"])),
    ]).fit(tr, y_tr)
    pred = model.predict(te)
    pos = cfg["dataset"]["positive_label"]
    return {
        "features": "with derived features" if add_features else "original 14 features",
        "n_features_in": tr.shape[1],
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred, pos_label=pos), 4),
        "precision": round(precision_score(y_te, pred, pos_label=pos), 4),
        "recall": round(recall_score(y_te, pred, pos_label=pos), 4),
    }


def baseline_cv(X_tr, y_tr, cfg, folds=5):
    """Supplementary: the same comparison with 5-fold CV inside the training split.
    The preprocessor is refitted on each training fold, so no fold sees its own validation data."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
    rows = []
    for k, (a, b) in enumerate(cv.split(X_tr, y_tr), 1):
        for add in (False, True):
            r = baseline(X_tr.iloc[a], X_tr.iloc[b], y_tr.iloc[a], y_tr.iloc[b], cfg, add)
            rows.append({"fold": k, **r})
    df = pd.DataFrame(rows)
    wide = df.pivot(index="fold", columns="features", values=["accuracy", "f1"])
    with_f, without = "with derived features", "original 14 features"
    for m in ("accuracy", "f1"):
        diff = wide[(m, with_f)] - wide[(m, without)]
        log(f"[baseline-cv] {m}: derived features better in {int((diff > 0).sum())}/{folds} folds, "
            f"mean change {diff.mean():+.4f} (min {diff.min():+.4f}, max {diff.max():+.4f})")
    return df


def main() -> None:
    cfg = load_config()
    set_seeds(cfg["seed"])
    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = cfg["dataset"]["target_column"]

    raw = ingest(cfg, log)                                   # task 2
    validate(raw, cfg, log)                                  # task 3

    X_tr, X_te, y_tr, y_te = split(raw, cfg)                 # task 4: split before cleaning
    cross = X_te.assign(_y=y_te.values).astype("object").merge(
        X_tr.assign(_y=y_tr.values).astype("object").drop_duplicates(), how="inner").shape[0]
    log(f"[split] test rows that exactly duplicate a train row (kept, reported only): {cross}")
    X_tr, y_tr, n_dropped = drop_train_duplicates(X_tr, y_tr, cfg)
    pre = make_preprocessor(cfg, add_features=True).fit(X_tr)
    joblib.dump(pre, RESULTS_DIR / "preprocessor.joblib")
    log("[clean] preprocessor fitted on train only, saved to results/preprocessor.joblib")
    report = pre.report(X_tr).add_prefix("train_").rename(columns={"train_column": "column", "train_kind": "kind"})
    report_te = pre.report(X_te)[["imputed", "clipped_low", "clipped_high"]].add_prefix("test_")
    pd.concat([report, report_te], axis=1).to_csv(RESULTS_DIR / "cleaning_report.csv", index=False)

    train_clean = pre.transform(X_tr).assign(**{target: y_tr.values})
    test_clean = pre.transform(X_te).assign(**{target: y_te.values})
    for name, df in (("train", train_clean), ("test", test_clean)):
        assert not df.isna().any().any(), f"{name} still has missing values"

    rows = [baseline(X_tr, X_te, y_tr, y_te, cfg, add) for add in (False, True)]  # task 5
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    for r in rows:
        log(f"[baseline] {r['features']}: accuracy {r['accuracy']}, F1 {r['f1']}")
    baseline_cv(X_tr, y_tr, cfg).to_csv(RESULTS_DIR / "baseline_cv.csv", index=False)

    audit(raw, cfg, log)                                     # task 6

    hashes = {}                                              # task 7
    for name, df in (("train", train_clean), ("test", test_clean)):
        path = DATA_DIR / f"adult_processed_{name}.csv"
        write_csv(df, path)
        hashes[f"processed_{name}"] = {"file": f"data/lab04/{path.name}", "rows": len(df),
                                       "columns": df.shape[1], "sha256": sha256_file(path)}
    hashes["raw_snapshot"] = {"file": "data/lab04/adult_raw.csv", "rows": len(raw),
                              "columns": raw.shape[1], "sha256": sha256_file(DATA_DIR / "adult_raw.csv")}
    manifest = {
        "dataset_hashes": hashes,
        "train_duplicates_dropped": n_dropped,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "numpy": version("numpy"),
                        "pyyaml": version("pyyaml")},
        "config": cfg,
    }
    with open(RESULTS_DIR / "dataset_manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    for k, v in hashes.items():
        log(f"[hash] {k}: {v['sha256']}")

    (RESULTS_DIR / "pipeline_log.txt").write_text("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
