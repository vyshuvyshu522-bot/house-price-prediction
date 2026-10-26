"""Loading the saved model bundle and making predictions."""
import joblib
import pandas as pd

from src import config


class ModelNotAvailableError(RuntimeError):
    """Raised when the model file is missing, unreadable or incompatible."""


class Predictor:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.model_name = bundle["model_name"]
        self.features = list(bundle["features"])
        self.zipcodes = set(bundle["zipcodes"])

    @classmethod
    def load(cls, path=config.MODEL_PATH) -> "Predictor":
        try:
            bundle = joblib.load(path)
        except FileNotFoundError as exc:
            raise ModelNotAvailableError("Model file not found. Run `python train_model.py` first.") from exc
        except Exception as exc:  # corrupt file, version mismatch, ...
            raise ModelNotAvailableError(f"Model file could not be loaded ({type(exc).__name__}).") from exc

        if not isinstance(bundle, dict) or not {"pipeline", "model_name", "features", "zipcodes"} <= bundle.keys():
            raise ModelNotAvailableError("Model file has an unexpected format. Retrain the model.")
        if list(bundle["features"]) != config.FEATURES:
            raise ModelNotAvailableError("Model features do not match the application. Retrain the model.")
        return cls(bundle)

    def predict(self, values: dict) -> int:
        """Predict a price (USD) from validated values; rounded to the nearest $1,000."""
        row = pd.DataFrame([{name: values[name] for name in self.features}], columns=self.features)
        row["zipcode"] = row["zipcode"].astype(str)
        price = float(self.pipeline.predict(row)[0])
        return int(round(price, -3))
