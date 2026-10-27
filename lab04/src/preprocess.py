"""Task 4-5: the cleaning transformer. Everything it learns comes from the training split.

fit(train):   median per numeric column, mode per categorical column, and the
              1st/99th percentile of each numeric column (after imputation).
transform(X): '?' -> missing -> impute -> clip -> optional derived features.
The same fitted object is applied unchanged to the test split, so no test
statistic ever reaches training.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AdultPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric, categorical, missing_token="?", clip_percentiles=(1, 99),
                 add_features=True, age_offset=16):
        self.numeric = numeric
        self.categorical = categorical
        self.missing_token = missing_token
        self.clip_percentiles = clip_percentiles
        self.add_features = add_features
        self.age_offset = age_offset

    def _to_missing(self, X):
        X = X.copy()
        for c in self.categorical:
            s = X[c].astype("object")
            X[c] = s.where(s != self.missing_token, np.nan)
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
        return X

    def fit(self, X, y=None):
        X = self._to_missing(X)
        self.medians_ = {c: float(X[c].median()) for c in self.numeric}
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.categorical}
        lo, hi = self.clip_percentiles
        filled = X[self.numeric].fillna(self.medians_)
        self.clip_low_ = {c: float(np.percentile(filled[c], lo)) for c in self.numeric}
        self.clip_high_ = {c: float(np.percentile(filled[c], hi)) for c in self.numeric}
        return self

    def transform(self, X):
        X = self._to_missing(X)
        X[self.numeric] = X[self.numeric].fillna(self.medians_)
        for c in self.categorical:
            X[c] = X[c].fillna(self.modes_[c]).astype("object")
        for c in self.numeric:
            X[c] = X[c].clip(self.clip_low_[c], self.clip_high_[c])
        if self.add_features:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            X["hours_per_age"] = X["hours-per-week"] / np.maximum(X["age"] - self.age_offset, 1)
        return X

    def report(self, X):
        """Per-column counts of what transform() would impute and clip (no data changed)."""
        X = self._to_missing(X)
        rows = []
        for c in self.numeric:
            v = X[c].fillna(self.medians_[c])
            rows.append({"column": c, "kind": "numeric", "imputed": int(X[c].isna().sum()),
                         "fill_value": self.medians_[c], "clip_low": self.clip_low_[c],
                         "clip_high": self.clip_high_[c],
                         "clipped_low": int((v < self.clip_low_[c]).sum()),
                         "clipped_high": int((v > self.clip_high_[c]).sum())})
        for c in self.categorical:
            rows.append({"column": c, "kind": "categorical", "imputed": int(X[c].isna().sum()),
                         "fill_value": self.modes_[c], "clip_low": "", "clip_high": "",
                         "clipped_low": "", "clipped_high": ""})
        return pd.DataFrame(rows)
