"""Task 6: label-quality checks.

1. Duplicate-conflict audit: rows with identical features but different income labels.
2. A 50-row random sample, checked against the UCI data dictionary for impossible or
   implausible values. Rule results are written next to each row in
   results/label_audit_sample.csv so the sample can be re-checked by eye.
"""
import pandas as pd

from common import RESULTS_DIR

# UCI data dictionary: each education level has exactly one education-num.
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
    "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
    "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}


def conflicts(raw, features, target):
    """Groups of rows with identical features but more than one label: (groups, rows)."""
    n_labels = raw.groupby(features, dropna=False, observed=True)[target].transform("nunique")
    rows = n_labels > 1
    groups = raw[rows].groupby(features, dropna=False, observed=True).ngroups if rows.any() else 0
    return int(groups), int(rows.sum())


def rule_checks(df, cfg):
    """Return a DataFrame of boolean flags (True = problem) for each row."""
    flags = pd.DataFrame(index=df.index)
    for col, (lo, hi) in cfg["schema"]["ranges"].items():
        v = pd.to_numeric(df[col], errors="coerce")
        flags[f"{col}_out_of_range"] = (v < lo) | (v > hi)
    edu = df["education"].astype("object").map(EDUCATION_NUM)
    flags["education_num_mismatch"] = edu.notna() & (edu != pd.to_numeric(df["education-num"]))
    rel, sex = df["relationship"].astype("object"), df["sex"].astype("object")
    flags["husband_not_male"] = (rel == "Husband") & (sex != "Male")
    flags["wife_not_female"] = (rel == "Wife") & (sex != "Female")
    age = pd.to_numeric(df["age"])
    flags["degree_too_young"] = (age < 20) & (pd.to_numeric(df["education-num"]) >= 13)
    flags["gain_and_loss_both"] = (pd.to_numeric(df["capital-gain"]) > 0) & (pd.to_numeric(df["capital-loss"]) > 0)
    return flags


def missing_profile(raw, cfg, log):
    """Are missing values random? Compare people with a missing value to everyone else."""
    target, pos = cfg["dataset"]["target_column"], cfg["dataset"]["positive_label"]
    token = cfg["schema"]["missing_token"]
    is_pos = raw[target].astype("object") == pos
    age, hours = pd.to_numeric(raw["age"]), pd.to_numeric(raw["hours-per-week"])
    occ = raw["occupation"].astype("object")
    occ_missing = occ.isna() | (occ == token)
    any_missing = pd.concat([raw[c].isna() | (raw[c].astype("object") == token) for c in raw.columns],
                            axis=1).any(axis=1)
    rows = []
    for name, mask in (("any value missing", any_missing), ("no value missing", ~any_missing),
                       ("occupation missing", occ_missing), ("occupation present", ~occ_missing),
                       ("everyone", pd.Series(True, index=raw.index))):
        rows.append({"group": name, "rows": int(mask.sum()),
                     "share_gt50k": round(is_pos[mask].mean(), 4),
                     "share_age_under_25": round((age[mask] < 25).mean(), 4),
                     "share_age_60_plus": round((age[mask] >= 60).mean(), 4),
                     "share_hours_under_35": round((hours[mask] < 35).mean(), 4),
                     "median_age": float(age[mask].median()),
                     "median_hours": float(hours[mask].median())})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "missing_profile.csv", index=False)
    wc = raw["workclass"].astype("object")
    wc_missing = wc.isna() | (wc == token)
    log(f"[audit] >50K share: any value missing {df.loc[0, 'share_gt50k']}, everyone {df.loc[4, 'share_gt50k']}; "
        f"workclass missing {int(wc_missing.sum())}, of which occupation also missing {int((wc_missing & occ_missing).sum())}")
    return df


def audit(raw, cfg, log):
    target = cfg["dataset"]["target_column"]
    features = [c for c in raw.columns if c != target]
    g_all, r_all = conflicts(raw, features, target)
    no_w = [c for c in features if c != "fnlwgt"]
    g_now, r_now = conflicts(raw, no_w, target)

    n = cfg["label_audit"]["sample_size"]
    sample = raw.sample(n=n, random_state=cfg["seed"])
    flags = rule_checks(sample, cfg)
    out = sample.copy()
    out.insert(0, "row_id", sample.index)
    out["problems"] = flags.apply(lambda r: ";".join(c for c, v in r.items() if v), axis=1)
    out.to_csv(RESULTS_DIR / "label_audit_sample.csv", index=False)
    full_flags = rule_checks(raw, cfg)

    summary = pd.DataFrame([
        {"check": "duplicate feature groups with conflicting labels (14 features)", "groups": g_all, "rows": r_all},
        {"check": "duplicate feature groups with conflicting labels (13 features, fnlwgt excluded)", "groups": g_now, "rows": r_now},
        {"check": f"sample of {n}: rows with at least one rule flag", "groups": "", "rows": int((flags.any(axis=1)).sum())},
    ] + [{"check": f"sample of {n}: {c}", "groups": "", "rows": int(flags[c].sum())} for c in flags.columns]
      + [{"check": f"full dataset: {c}", "groups": "", "rows": int(full_flags[c].sum())} for c in full_flags.columns])
    summary.to_csv(RESULTS_DIR / "label_audit_summary.csv", index=False)
    missing_profile(raw, cfg, log)
    log(f"[audit] conflicting duplicates, 14 features: {g_all} groups / {r_all} rows; "
        f"fnlwgt excluded: {g_now} groups / {r_now} rows")
    log(f"[audit] 50-row sample: {int(flags.any(axis=1).sum())} rows flagged by rules")
    return summary
