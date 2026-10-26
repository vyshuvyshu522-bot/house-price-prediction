import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import clean_data, split_train_test
from src.modeling import build_pipeline, regression_metrics
from sklearn.linear_model import LinearRegression


def make_row(**overrides):
    row = {"id": 1, "price": 400000, "sqft_living": 1500, "bedrooms": 3, "bathrooms": 2, "sqft_lot": 5000,
           "floors": 1, "waterfront": 0, "view": 0, "condition": 3, "grade": 7, "yr_built": 1990, "zipcode": 98103}
    row.update(overrides)
    return row


def test_clean_data_removes_bad_rows_and_duplicates():
    df = pd.DataFrame([
        make_row(id=1), make_row(id=1),                 # exact duplicate
        make_row(id=2, price=np.nan),                   # missing target
        make_row(id=3, price=-5),                       # impossible price
        make_row(id=4, bedrooms=33),                    # data-entry typo
        make_row(id=5, sqft_living=0),                  # impossible area
        make_row(id=6),                                 # good row
    ])
    clean, report = clean_data(df)
    assert sorted(clean["id"]) == [1, 6]
    assert report["duplicate_rows_removed"] == 1
    assert report["implausible_bedrooms_removed"] == 1


def test_clean_data_converts_zipcode_to_string_and_keeps_missing_features():
    df = pd.DataFrame([make_row(id=1, zipcode=98103.0), make_row(id=2, sqft_living=np.nan)])
    clean, _ = clean_data(df)
    assert clean.loc[0, "zipcode"] == "98103"
    assert clean["sqft_living"].isna().sum() == 1   # left for the imputer, not silently dropped


def test_clean_data_requires_expected_columns():
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({"price": [1]}))


def test_split_has_no_house_overlap(clean_df):
    train, test = split_train_test(clean_df)
    assert set(train["id"]).isdisjoint(set(test["id"]))
    assert len(train) + len(test) == len(clean_df)


def test_pipeline_handles_missing_values_and_unseen_zipcode(clean_df):
    sample = clean_df.sample(500, random_state=1)
    pipe = build_pipeline(LinearRegression()).fit(sample[config.FEATURES], sample[config.TARGET])
    row = sample[config.FEATURES].head(2).copy()
    row.loc[row.index[0], "sqft_living"] = np.nan   # missing number -> median imputed
    row.loc[row.index[1], "zipcode"] = "00000"      # unseen category -> ignored by encoder
    assert np.isfinite(pipe.predict(row)).all()


def test_regression_metrics_are_correct():
    m = regression_metrics([100, 200, 300], [110, 190, 330])
    assert m["mae"] == pytest.approx(50 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["mdape"] == pytest.approx(10.0)
