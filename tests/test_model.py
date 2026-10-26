import joblib
import pytest

from src import config
from src.predictor import ModelNotAvailableError, Predictor
from src.validation import validate_input


def test_model_loads_and_features_match(model_dir):
    predictor = Predictor.load(model_dir / "model.joblib")
    assert predictor.features == config.FEATURES
    assert "98103" in predictor.zipcodes


def test_valid_prediction_is_plausible(model_dir, valid_input):
    predictor = Predictor.load(model_dir / "model.joblib")
    values, errors = validate_input(valid_input, predictor.zipcodes)
    assert not errors
    price = predictor.predict(values)
    assert isinstance(price, int) and price % 1000 == 0
    assert 50_000 < price < 5_000_000


def test_missing_model_file_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="train_model.py"):
        Predictor.load(tmp_path / "nope.joblib")


def test_corrupt_model_file_raises_clear_error(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelNotAvailableError):
        Predictor.load(bad)


def test_model_with_wrong_features_is_rejected(tmp_path, model_dir):
    bundle = joblib.load(model_dir / "model.joblib")
    bundle["features"] = bundle["features"][:-1]
    path = tmp_path / "wrong.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ModelNotAvailableError, match="features"):
        Predictor.load(path)
