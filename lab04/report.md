# Lab 4: Data pipeline and quality

## 1. Goal

Build a reproducible data pipeline for the UCI Adult dataset that does not leak test information. The pipeline validates the raw data, cleans it, adds derived features, audits label quality, and records a hash of the processed data. The whole pipeline runs with one command.

### Problem definition

- **Task.** Predict whether a person earns more than $50,000 per year. This is a binary classification problem on the 1994 US Census extract (UCI Adult).
- **Inputs.** 14 features about one person: 6 numeric (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country).
- **Output.** The label `income` (stored as `class` in OpenML): `>50K` or `<=50K`.
- **Success metrics.** F1 score for the `>50K` class, and accuracy. Only 23.9% of people earn >50K, so a model that always says `<=50K` already gets about 76% accuracy. F1 shows whether the model really finds the high-income people.
- **Constraints.**
  - No leakage: the test split is separated before any cleaning, and every statistic (medians, modes, clipping bounds) is learned from the training split only.
  - Reproducible: fixed seed, all settings in `config.yaml`, a SHA256 hash of the processed data, and one command (`bash lab04/src/run_all.sh`) that rebuilds every result.
  - Nothing is dropped silently: validation only reports problems, and every cleaning step is counted.

## 2. Method

**Environment.** Python 3.11.8, pandas 2.2.2, scikit-learn 1.5.0, NumPy 1.26.4 (all versions in `results/versions.txt`). Seed 42.

**One command.** `bash lab04/src/run_all.sh` runs `src/pipeline.py`, which calls each step in order:

| Step | File | What it does |
|---|---|---|
| Ingest | `src/ingest.py` | `fetch_openml(name='adult', version=2, as_frame=True)`. Logs the row and column count and saves an untouched copy of the raw data with its SHA256 |
| Validate | `src/validate.py` | Checks column names, types, ranges, missing values, label values and duplicates against the schema in `config.yaml`. Reports only; no row is changed or dropped |
| Split | `src/pipeline.py` | 70/30 stratified train/test split, **before any cleaning** |
| Clean | `src/preprocess.py` | Drops exact duplicates from train only. Fits the preprocessor on train only: median for numeric and mode for categorical missing values, then clips numeric features to the train 1st/99th percentiles. The same fitted object transforms the test split. Saved to `results/preprocessor.joblib` |
| Features | `src/preprocess.py` | `capital_net = capital-gain − capital-loss` and `hours_per_age = hours-per-week / (age − 16)`. The minimum age is 17, so the denominator is never 0 |
| Baseline | `src/pipeline.py` | One-hot encoding + standard scaling + `LogisticRegression(max_iter=1000)`, with and without the derived features. Also 5-fold CV inside the training split, refitting the preprocessor in every fold |
| Label audit | `src/audit.py` | Finds rows with identical features but different labels. Draws 50 random rows and checks them against the UCI data dictionary |
| Hash | `src/pipeline.py` | Writes the processed train and test data, computes SHA256, and stores the hashes together with the full config in `results/dataset_manifest.yaml` |

**Schema.** Types, ranges (age 17–90, hours-per-week 1–99, capital-gain 0–99999, capital-loss 0–4356) and the missing-value token `?` are taken from the manual. OpenML already delivers missing values as NaN instead of `?`, so both are counted as missing.

**Label audit rules** for the 50-row sample, from the UCI data dictionary:
- every value inside its range;
- education matches education-num (each education level has exactly one number);
- relationship "Husband" goes with sex "Male", and "Wife" with "Female";
- nobody under 20 has a bachelor's degree or higher.

The sample with the rule results next to each row is in `results/label_audit_sample.csv`, so it can be re-checked by hand.

## 3. Results

### 3.1 Ingestion (`results/pipeline_log.txt`)

48,842 rows and 15 columns, as expected. The raw snapshot hash is `8f970e20…0adddc` (full hash in `results/dataset_manifest.yaml`).

### 3.2 Validation (`results/validation_report.md`, `results/validation_checks.csv`)

| Check | Result |
|---|---|
| Columns | All 15 schema columns present, no extra columns |
| Types | All pass. Integer columns arrive as int64, categorical columns as category |
| Ranges | 0 violations. Observed values exactly hit the limits: age 17–90, hours 1–99, capital-gain 0–99,999, capital-loss 0–4,356 |
| Missing values | workclass 2,799 (5.7%), occupation 2,809 (5.8%), native-country 857 (1.8%). 3,620 rows (7.4%) have at least one |
| Labels | `<=50K` 37,155, `>50K` 11,687 (23.9%). No unexpected values |
| Duplicates | 52 exact duplicate rows; 57 rows whose 14 features duplicate another row |

No rows were dropped at this step.

### 3.3 Split and cleaning (`results/cleaning_report.csv`)

- **Split:** 34,189 train and 14,653 test rows. Both have 23.93% `>50K`, so stratification worked.
- **Duplicates:** 24 exact duplicates dropped from train, which leaves 34,165 rows. The test split was not touched. 24 test rows are exact copies of a train row; they were kept and only reported.

| Column | Learned from train | Train rows changed | Test rows changed |
|---|---|---|---|
| workclass | mode "Private" | 1,923 imputed | 873 imputed |
| occupation | mode "Prof-specialty" | 1,933 imputed | 873 imputed |
| native-country | mode "United-States" | 590 imputed | 267 imputed |
| age | clip 17–74 | 302 clipped | 143 clipped |
| fnlwgt | clip 27,035–509,629 | 683 clipped | 272 clipped |
| education-num | clip 3–16 | 240 clipped | 88 clipped |
| capital-gain | clip 0–15,024 | 255 clipped | 136 clipped |
| capital-loss | clip 0–1,980 | 340 clipped | 151 clipped |
| hours-per-week | clip 8–80 | 537 clipped | 236 clipped |

The numeric columns had no missing values, so the median was never used. After cleaning, neither split has any missing value.

### 3.4 Derived features (`results/baseline_metrics.csv`, `results/baseline_cv.csv`)

Logistic regression on the test split:

| Features | Accuracy | F1 (>50K) | Precision | Recall |
|---|---|---|---|---|
| Original 14 features | 0.8540 | 0.6601 | 0.7448 | 0.5927 |
| + capital_net, hours_per_age | **0.8578** | **0.6696** | 0.7537 | 0.6024 |

5-fold CV inside the training split:

| Metric | Folds where the derived features help | Mean change | Range |
|---|---|---|---|
| Accuracy | 4 of 5 (1 tie) | +0.0021 | +0.0000 to +0.0041 |
| F1 | 5 of 5 | +0.0065 | +0.0004 to +0.0099 |

### 3.5 Label quality (`results/label_audit_summary.csv`, `results/label_audit_sample.csv`)

| Check | Result |
|---|---|
| Identical 14 features, different labels | 5 groups, 10 rows |
| Identical features without fnlwgt, different labels | 1,029 groups, 3,873 rows |
| 50-row random sample, rule checks | 0 rows flagged |
| Whole dataset, rule checks | 1 "Husband" who is not male, 3 "Wife" rows that are not female, 3 people under 20 with a bachelor's degree or higher. 0 range or education mismatches |

The manual's Cohen's kappa step is only for subjective labels, so it does not apply to Adult.

**Are missing values random?** (`results/missing_profile.csv`)

| Group | Rows | Share >50K | Age under 25 | Age 60+ | Under 35 hours/week |
|---|---|---|---|---|---|
| Any value missing | 3,620 | 13.2% | 31.1% | 22.4% | 37.9% |
| Occupation missing | 2,809 | 9.4% | 36.8% | 26.6% | 44.8% |
| Occupation present | 46,033 | 24.8% | 16.1% | 7.2% | 15.5% |
| Everyone | 48,842 | 23.9% | 17.3% | 8.3% | 17.2% |

All 2,799 rows with a missing workclass also have a missing occupation.

### 3.6 Reproducibility (`results/dataset_manifest.yaml`)

| File | Rows × columns | SHA256 |
|---|---|---|
| Processed train | 34,165 × 17 | `a93a7174fc8309317352c91227d759e62150aa488bfd37a18b40b1a910cb3338` |
| Processed test | 14,653 × 17 | `e41edab38eff091a8a930672be3a1f073cd0bd7c5752b4a9e0755234f2dd1069` |

The processed files (14 cleaned features + 2 derived features + label) are written to `data/lab04/`, which is git-ignored. Running `bash lab04/src/run_all.sh` again rebuilds them; the same hashes confirm the result is identical. The data card is in `data_card.md`.

## 4. Conclusions

1. The two derived features help, but only a little. With them, accuracy goes from 85.40% to 85.78% and F1 from 0.660 to 0.670. In 5-fold cross-validation F1 improved in all 5 folds, so the gain is not just luck. The gain is small because LogReg can already do subtraction by itself, so capital_net brings very little new information. LogReg cannot do division, so only hours_per_age is really new information. So when we create features for a linear model, we should create the ones it cannot build by itself, such as ratios.
2. The features cannot fully decide a person's income: many people have the same features but different incomes. With all 14 features, only 5 groups (10 rows) have the same features and conflicting labels. Without fnlwgt there are 1,029 groups (3,873 rows). With fnlwgt we only see 5 groups because fnlwgt is different in almost every row, so almost no two rows are exactly the same and most conflicts are hidden. fnlwgt is only a census sampling weight, not something about the person. So no model can reach 100% accuracy on this data. To make the model better we need better data, with more features that explain income, not only a stronger model.
3. The missing values are not random. Only 13.2% of the people with a missing value earn >50K, against 23.9% overall, and workclass and occupation are always missing together. This is probably because many of them are students, retired people or part-time workers: among people with a missing occupation, 36.8% are under 25 and 26.6% are 60 or older (16.1% and 7.2% for everyone else), and 44.8% work under 35 hours a week (15.5% for everyone else). They have no fixed job, so they cannot fill in these two fields, and they also earn less. Filling them with the mode turns a student or a retired person into "Prof-specialty", a high-income job, so the model learns something wrong. The manual asks for mode imputation, so I used it, but a better way is to keep "Unknown" as its own category and not guess.
