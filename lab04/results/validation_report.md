# Validation report (raw UCI Adult, before any split or cleaning)

Rows checked: 48842. Columns: 15. Missing-value policy: '?' and NaN both count as missing.
No rows were dropped or changed by validation.

## Summary

- Schema columns missing: none; unexpected columns: none
- Type check failures: none
- Range violations (all checked columns): 0
- Rows with at least one missing value: 3620 (7.4%)
- Label values: '<=50K': 37155, '>50K': 11687; unexpected values: none
- Exact duplicate rows: 52; rows whose 14 features duplicate another row: 57

## Per-column checks

| Column | Expected | Observed dtype | Type OK | Missing | Missing % | Range | Below min | Above max | Observed min–max |
|---|---|---|---|---|---|---|---|---|---|
| age | int | int64 | yes | 0 | 0.0 | 17-90 | 0 | 0 | 17–90 |
| workclass | category | category | yes | 2799 | 5.73 |  |  |  |  |
| fnlwgt | int | int64 | yes | 0 | 0.0 |  |  |  |  |
| education | category | category | yes | 0 | 0.0 |  |  |  |  |
| education-num | int | int64 | yes | 0 | 0.0 |  |  |  |  |
| marital-status | category | category | yes | 0 | 0.0 |  |  |  |  |
| occupation | category | category | yes | 2809 | 5.75 |  |  |  |  |
| relationship | category | category | yes | 0 | 0.0 |  |  |  |  |
| race | category | category | yes | 0 | 0.0 |  |  |  |  |
| sex | category | category | yes | 0 | 0.0 |  |  |  |  |
| capital-gain | int | int64 | yes | 0 | 0.0 | 0-99999 | 0 | 0 | 0–99999 |
| capital-loss | int | int64 | yes | 0 | 0.0 | 0-4356 | 0 | 0 | 0–4356 |
| hours-per-week | int | int64 | yes | 0 | 0.0 | 1-99 | 0 | 0 | 1–99 |
| native-country | category | category | yes | 857 | 1.75 |  |  |  |  |
| class | category | category | yes | 0 | 0.0 |  |  |  |  |

Integer columns are accepted if every value is a whole number, even if pandas stores them as float.
