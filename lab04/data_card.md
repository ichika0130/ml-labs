# Data Card: UCI Adult (Census Income)

## Source
- **Original:** UCI Machine Learning Repository, "Adult" dataset, donated by Ronny Kohavi and Barry Becker (1996). It was extracted from the 1994 US Census Bureau database.
- **How we load it:** `sklearn.datasets.fetch_openml(name='adult', version=2, as_frame=True)`. This OpenML version combines the original train file (adult.data) and test file (adult.test).
- **Task:** predict whether a person's yearly income is above $50,000 (`class` = `>50K` or `<=50K`).

## License
Creative Commons Attribution 4.0 (CC BY 4.0), the license used by the UCI Machine Learning Repository. Reuse is allowed if the source is credited.

## Size
- 48,842 rows, 15 columns: 14 features and 1 label.
- 6 numeric features: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week.
- 8 categorical features: workclass, education, marital-status, occupation, relationship, race, sex, native-country.
- Missing values (`?` in the original UCI files; OpenML delivers them as NaN): workclass 2,799, occupation 2,809, native-country 857. 3,620 rows (7.4%) have at least one missing value.
- Label balance: 23.9% `>50K`, 76.1% `<=50K`.
- After this pipeline: 34,165 training rows (70% split, 24 exact duplicates removed) and 14,653 test rows (30%, untouched). Hashes of the processed files are in `results/dataset_manifest.yaml`.

## Collection and processing
- Every row is one person from the 1994 Current Population Survey. The original donors already filtered the rows to people older than 16 with more than 0 working hours.
- `fnlwgt` is a census sampling weight (how many people in the population this row represents). It is not a personal attribute.
- Our pipeline: split 70/30 stratified first, then impute (median / mode), clip numeric features to the 1st–99th percentile, and drop exact duplicates from train. All statistics are learned from the training split only. Details are in `report.md` and `config.yaml`.

## Known biases
- **Old data.** The data is from 1994, and the $50K threshold is not adjusted for inflation. A model trained on it describes the 1994 US labour market, not today's.
- **Sex imbalance.** 66.8% of rows are male. 30.4% of men earn >50K, compared with 10.9% of women. A model can learn sex as a strong predictor of income.
- **Race imbalance.** 85.5% of rows are White. Small groups have very few rows: Amer-Indian-Eskimo 1.0%, Other 0.8%. Results for these groups are less reliable.
- **Country imbalance.** 89.7% of rows have native-country United-States, so the model knows little about other countries.
- **Missing values are not random.** Rows with a missing value earn >50K much less often (13.2% vs 23.9% overall). Whenever workclass is missing, occupation is missing too. Filling them with the mode ("Private", "Prof-specialty") therefore hides a real pattern.
- **Top-coded values.** capital-gain is capped at 99,999 (244 rows). That is a recording limit, not a real amount, and the clipping step changes it further.
- **Label imbalance.** Only 23.9% of rows are `>50K`, so accuracy alone looks good even for a weak model. This is why F1 is also reported.

## Privacy concerns
- The data has no names or IDs. It is public and anonymised.
- However, it contains sensitive attributes: sex, race, marital status and native country. A rare combination of age, race, native-country and occupation could point to a small group of real people. The data should not be joined with other datasets to identify individuals.
- A model trained on this data should not be used to make real decisions about people, such as loans or hiring. It would repeat the 1994 inequalities listed above.

## Intended use
Teaching and benchmarking classification and data-pipeline methods.
