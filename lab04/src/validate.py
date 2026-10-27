"""Task 3: schema, type, range, missing-value, label and duplicate checks.

Nothing is dropped or changed here: every check only counts and reports.
'?' means missing (the manual's null policy). OpenML may already deliver missing
values as NaN, so both are counted as missing.
"""
import numpy as np
import pandas as pd

from common import RESULTS_DIR


def missing_mask(s: pd.Series, token: str) -> pd.Series:
    return s.isna() | (s.astype("object") == token)


def check_type(s: pd.Series, expected: str, token: str):
    """int: numeric dtype and every non-missing value a whole number.
    category: non-numeric values. Returns (ok, observed dtype, number of bad values)."""
    observed = str(s.dtype)
    present = s[~missing_mask(s, token)]
    if expected == "int":
        if not pd.api.types.is_numeric_dtype(s):
            values = pd.to_numeric(present, errors="coerce")
            bad = int(values.isna().sum() + (values.dropna() % 1 != 0).sum())
            return False, observed, bad
        bad = int((present % 1 != 0).sum())
        return bad == 0, observed, bad
    ok = not pd.api.types.is_numeric_dtype(s)
    return ok, observed, 0 if ok else int(len(present))


def validate(raw: pd.DataFrame, cfg: dict, log):
    schema = cfg["schema"]
    token = schema["missing_token"]
    target = cfg["dataset"]["target_column"]
    rows = []

    expected_cols = list(schema["types"])
    missing_cols = [c for c in expected_cols if c not in raw.columns]
    extra_cols = [c for c in raw.columns if c not in expected_cols]

    for col, expected in schema["types"].items():
        if col not in raw.columns:
            continue
        s = raw[col]
        type_ok, observed, bad_type = check_type(s, expected, token)
        miss = missing_mask(s, token)
        row = {
            "column": col, "expected_type": expected, "observed_dtype": observed,
            "type_ok": type_ok, "type_violations": bad_type,
            "missing_nan": int(s.isna().sum()),
            "missing_question_mark": int((s.astype("object") == token).sum()),
            "missing_total": int(miss.sum()),
            "missing_pct": round(100 * miss.mean(), 2),
            "n_unique": int(s[~miss].nunique()),
            "range": "", "below_min": "", "above_max": "", "observed_min": "", "observed_max": "",
        }
        if col in schema["ranges"]:
            lo, hi = schema["ranges"][col]
            v = pd.to_numeric(s[~miss], errors="coerce")
            row.update({"range": f"{lo}-{hi}", "below_min": int((v < lo).sum()),
                        "above_max": int((v > hi).sum()),
                        "observed_min": v.min(), "observed_max": v.max()})
        rows.append(row)
    checks = pd.DataFrame(rows)

    labels = raw[target].astype("object").value_counts(dropna=False)
    expected_labels = {cfg["dataset"]["positive_label"], "<=50K"}
    unexpected_labels = {str(k): int(v) for k, v in labels.items() if k not in expected_labels}

    features = [c for c in raw.columns if c != target]
    exact_dups = int(raw.duplicated().sum())
    feature_dups = int(raw.duplicated(subset=features).sum())

    range_violations = int(checks["below_min"].replace("", 0).sum() + checks["above_max"].replace("", 0).sum())
    type_failures = checks.loc[~checks["type_ok"], "column"].tolist()

    checks.to_csv(RESULTS_DIR / "validation_checks.csv", index=False)
    write_report(raw, checks, labels, unexpected_labels, missing_cols, extra_cols,
                 exact_dups, feature_dups, range_violations, type_failures, token)

    log(f"[validate] columns missing={missing_cols} extra={extra_cols}")
    log(f"[validate] type failures: {type_failures or 'none'}")
    log(f"[validate] range violations: {range_violations}")
    log(f"[validate] missing values: " + ", ".join(
        f"{r.column}={r.missing_total}" for r in checks.itertuples() if r.missing_total))
    log(f"[validate] labels: { {str(k): int(v) for k, v in labels.items()} }; unexpected: {unexpected_labels or 'none'}")
    log(f"[validate] exact duplicate rows: {exact_dups}; duplicate feature rows: {feature_dups}")
    log("[validate] no rows dropped")
    return checks


def write_report(raw, checks, labels, unexpected_labels, missing_cols, extra_cols,
                 exact_dups, feature_dups, range_violations, type_failures, token):
    n = len(raw)
    lines = [
        "# Validation report (raw UCI Adult, before any split or cleaning)",
        "",
        f"Rows checked: {n}. Columns: {raw.shape[1]}. Missing-value policy: '{token}' and NaN both count as missing.",
        "No rows were dropped or changed by validation.",
        "",
        "## Summary",
        "",
        f"- Schema columns missing: {missing_cols or 'none'}; unexpected columns: {extra_cols or 'none'}",
        f"- Type check failures: {type_failures or 'none'}",
        f"- Range violations (all checked columns): {range_violations}",
        f"- Rows with at least one missing value: {int(checks_any_missing(raw, token).sum())} "
        f"({100 * checks_any_missing(raw, token).mean():.1f}%)",
        f"- Label values: " + ", ".join(f"'{k}': {v}" for k, v in labels.items())
        + f"; unexpected values: {unexpected_labels or 'none'}",
        f"- Exact duplicate rows: {exact_dups}; rows whose 14 features duplicate another row: {feature_dups}",
        "",
        "## Per-column checks",
        "",
        "| Column | Expected | Observed dtype | Type OK | Missing | Missing % | Range | Below min | Above max | Observed min–max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in checks.itertuples():
        obs = f"{r.observed_min:g}–{r.observed_max:g}" if r.range else ""
        lines.append(f"| {r.column} | {r.expected_type} | {r.observed_dtype} | {'yes' if r.type_ok else 'NO'} "
                     f"| {r.missing_total} | {r.missing_pct} | {r.range} | {r.below_min} | {r.above_max} | {obs} |")
    lines += ["", "Integer columns are accepted if every value is a whole number, even if pandas stores "
              "them as float.", ""]
    (RESULTS_DIR / "validation_report.md").write_text("\n".join(lines))


def checks_any_missing(raw, token):
    return pd.concat([missing_mask(raw[c], token) for c in raw.columns], axis=1).any(axis=1)
