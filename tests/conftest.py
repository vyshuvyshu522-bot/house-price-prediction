"""Shared fixtures. A small, fast model is trained on a sample of the real dataset,
so the tests do not depend on (or overwrite) the full trained model."""
import joblib
import pytest
from sklearn.ensemble import RandomForestRegressor

from app import create_app
from src import config
from src.data import clean_data, load_raw
from src.modeling import build_pipeline

VALID_INPUT = {
    "sqft_living": "1800", "bedrooms": "3", "bathrooms": "2", "sqft_lot": "7500", "floors": "1",
    "waterfront": "0", "view": "0", "condition": "3", "grade": "7", "yr_built": "1980", "zipcode": "98103",
}


@pytest.fixture(scope="session")
def clean_df():
    df, _ = clean_data(load_raw())
    return df


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, clean_df):
    sample = clean_df.sample(2500, random_state=0)
    pipe = build_pipeline(RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=1))
    pipe.fit(sample[config.FEATURES], sample[config.TARGET])
    directory = tmp_path_factory.mktemp("model")
    joblib.dump({"pipeline": pipe, "model_name": "Random Forest", "features": config.FEATURES,
                 "zipcodes": sorted(clean_df["zipcode"].unique().tolist())}, directory / "model.joblib")
    return directory


@pytest.fixture
def valid_input():
    return dict(VALID_INPUT)


@pytest.fixture
def app(tmp_path, model_dir):
    return create_app({"TESTING": True, "DATABASE": str(tmp_path / "test.db"),
                       "MODEL_PATH": str(model_dir / "model.joblib"),
                       "METRICS_PATH": str(tmp_path / "missing_metrics.json")})


@pytest.fixture
def client(app):
    return app.test_client()
