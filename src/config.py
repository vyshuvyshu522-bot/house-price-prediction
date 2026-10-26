"""Central configuration: paths, feature lists and the form/validation specs.

Keeping the feature specs in ONE place means the training code, the server-side
validator and the HTML form can never drift apart.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "kc_house_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "house_price_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
CHARTS_DIR = BASE_DIR / "static" / "img"
DATABASE_PATH = BASE_DIR / "instance" / "predictions.db"

TARGET = "price"
GROUP_COLUMN = "id"  # the same house can be sold twice; used to avoid leakage

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC_FEATURES = [
    "sqft_living", "bedrooms", "bathrooms", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "yr_built",
]
CATEGORICAL_FEATURES = ["zipcode"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Prediction form / validation specification (order = order shown in the form).
# kind: "int" or "float" (numeric) or "choice" (zipcode, filled from the model).
FEATURE_SPECS = {
    "sqft_living": {"label": "Living area (sq ft)", "kind": "int", "min": 300, "max": 15000, "step": 10,
                    "default": 1800, "help": "Interior living space."},
    "bedrooms": {"label": "Bedrooms", "kind": "int", "min": 1, "max": 10, "step": 1, "default": 3},
    "bathrooms": {"label": "Bathrooms", "kind": "float", "min": 0.5, "max": 8, "step": 0.25, "default": 2,
                  "help": "0.5 = toilet + sink only; 0.75 = adds a shower."},
    "sqft_lot": {"label": "Lot size (sq ft)", "kind": "int", "min": 500, "max": 1700000, "step": 100,
                 "default": 7500},
    "floors": {"label": "Floors", "kind": "float", "min": 1, "max": 3.5, "step": 0.5, "default": 1},
    "waterfront": {"label": "Waterfront", "kind": "int", "min": 0, "max": 1, "step": 1, "default": 0,
                   "choices": {0: "No", 1: "Yes"}},
    "view": {"label": "View quality (0-4)", "kind": "int", "min": 0, "max": 4, "step": 1, "default": 0,
             "help": "0 = no special view, 4 = excellent."},
    "condition": {"label": "Condition (1-5)", "kind": "int", "min": 1, "max": 5, "step": 1, "default": 3},
    "grade": {"label": "Build grade (1-13)", "kind": "int", "min": 1, "max": 13, "step": 1, "default": 7,
              "help": "King County grade: 7 = average construction and design."},
    "yr_built": {"label": "Year built", "kind": "int", "min": 1900, "max": 2015, "step": 1, "default": 1980,
                 "help": "The training data covers 1900-2015."},
    "zipcode": {"label": "ZIP code", "kind": "choice"},
}
